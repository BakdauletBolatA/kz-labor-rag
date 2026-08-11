"""Кодирование текста в векторы.

Единственное место в проекте, где к тексту приписываются префиксы ``query:``
и ``passage:``. Модели семейства e5 обучены на асимметричных префиксах:
запрос и документ кодируются по-разному. Перепутать их или забыть — тихая
деградация качества без единой ошибки в логах, поэтому наружу выставлены
именно ``encode_query`` и ``encode_passages``, а не общий ``encode(texts)``,
который легко вызвать не с той стороны.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

ENCODER_VERSION = "1.0"


@runtime_checkable
class Encoder(Protocol):
    """Что от энкодера нужно остальному коду."""

    @property
    def dimensions(self) -> int: ...

    @property
    def descriptor(self) -> dict[str, str]: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class EncoderParams:
    model: str
    dimensions: int
    query_prefix: str
    passage_prefix: str
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"

    @classmethod
    def from_config(cls, section: dict) -> EncoderParams:
        return cls(
            model=section["model"],
            dimensions=int(section["dimensions"]),
            query_prefix=section["query_prefix"],
            passage_prefix=section["passage_prefix"],
            normalize=bool(section["normalize"]),
            batch_size=int(section["batch_size"]),
            device=section["device"],
        )


class EmbeddingCache:
    """Кэш векторов на диске.

    Ключ — sha256(модель + подпись чанкинга + текст). Смена чанкинга меняет
    подпись и автоматически обесценивает старые векторы; смена модели пишет в
    отдельный файл и не затирает уже посчитанное, поэтому вернуться к
    предыдущей модели можно без переиндексации.
    """

    def __init__(self, directory: str | Path, model: str, chunking_signature: str) -> None:
        self.model = model
        self.chunking_signature = chunking_signature
        slug = model.replace("/", "__")
        self.path = Path(directory) / f"{slug}.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )
        self._db.commit()

    def key(self, text: str) -> str:
        payload = f"{self.model}|{self.chunking_signature}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_many(self, texts: Sequence[str]) -> dict[int, np.ndarray]:
        if not texts:
            return {}
        keys = [self.key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        # SQLite ограничивает число параметров, поэтому запрашиваем частями.
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            for key, dim, blob in self._db.execute(
                f"SELECT key, dim, vec FROM vectors WHERE key IN ({placeholders})", batch
            ):
                found[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return {i: found[k] for i, k in enumerate(keys) if k in found}

    def put_many(self, texts: Sequence[str], vectors: np.ndarray) -> None:
        rows = [
            (self.key(text), int(vec.shape[0]), vec.astype(np.float32).tobytes())
            for text, vec in zip(texts, vectors, strict=True)
        ]
        self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", rows)
        self._db.commit()

    @property
    def size(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        self._db.close()


class E5Encoder:
    """sentence-transformers поверх модели семейства e5."""

    def __init__(self, params: EncoderParams, cache: EmbeddingCache | None = None) -> None:
        self.params = params
        self.cache = cache
        self._model = None

    @property
    def dimensions(self) -> int:
        return self.params.dimensions

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "backend": "sentence-transformers",
            "model": self.params.model,
            "dimensions": str(self.params.dimensions),
            "query_prefix": self.params.query_prefix,
            "passage_prefix": self.params.passage_prefix,
            "normalize": str(self.params.normalize),
            "encoder_version": ENCODER_VERSION,
        }

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "нужен пакет sentence-transformers: pip install sentence-transformers"
                ) from exc
            log.info(
                "Загружается модель эмбеддингов %s (%s)", self.params.model, self.params.device
            )
            self._model = SentenceTransformer(self.params.model, device=self.params.device)
            actual = self._model.get_sentence_embedding_dimension()
            if actual != self.params.dimensions:
                raise ValueError(
                    f"в конфиге dimensions={self.params.dimensions}, а модель "
                    f"{self.params.model} выдаёт {actual}. Схема pgvector создаётся по "
                    "конфигу, поэтому расхождение обязано быть фатальным."
                )
        return self._model

    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.params.batch_size,
            normalize_embeddings=self.params.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Закодировать запрос. Префикс запроса ставится здесь и только здесь."""
        return self._encode_raw([self.params.query_prefix + text])[0]

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Закодировать документы, переиспользуя кэш.

        Кэш ключуется по тексту *без* префикса, но сам префикс входит в модель
        кодирования; менять префикс без смены модели нельзя, поэтому такой
        ключ безопасен.
        """
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        cached: dict[int, np.ndarray] = self.cache.get_many(texts) if self.cache else {}
        missing = [i for i in range(len(texts)) if i not in cached]
        if missing:
            log.info("Кодируется %d чанков (%d взято из кэша)", len(missing), len(cached))
            fresh = self._encode_raw([self.params.passage_prefix + texts[i] for i in missing])
            if self.cache:
                self.cache.put_many([texts[i] for i in missing], fresh)
            for slot, i in enumerate(missing):
                cached[i] = fresh[slot]

        return np.vstack([cached[i] for i in range(len(texts))]).astype(np.float32)


class HashEncoder:
    """Детерминированный энкодер без модели.

    Существует ради тестов и отладки пайплайна без скачивания весов: вектор
    выводится из хеша текста. Семантики в нём нет, поэтому для измерений он
    непригоден — но позволяет проверить индексацию, схему и выдачу.
    """

    def __init__(
        self,
        dimensions: int = 768,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        self._dimensions = dimensions
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.seen_prefixes: list[str] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def descriptor(self) -> dict[str, str]:
        return {"backend": "hash", "model": "hash", "dimensions": str(self._dimensions)}

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        vec = rng.standard_normal(self._dimensions).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def encode_query(self, text: str) -> np.ndarray:
        self.seen_prefixes.append(self.query_prefix)
        return self._vector(self.query_prefix + text)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        self.seen_prefixes.extend([self.passage_prefix] * len(texts))
        return np.vstack([self._vector(self.passage_prefix + t) for t in texts])
