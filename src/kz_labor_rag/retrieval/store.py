"""Хранилище чанков и векторов в PostgreSQL + pgvector.

Схема создаётся по конфигу: размерность вектора берётся из ``embeddings``,
метрика расстояния — из ``vector_store``. В таблице метаданных пишется,
чем именно построен текущий индекс, чтобы поиск не выполнялся поверх векторов
от другой модели или другого чанкинга — молчаливое несоответствие здесь
испортило бы все метрики разом.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kz_labor_rag.types import Chunk, RetrievedChunk

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Оператор расстояния pgvector и способ превратить его в человекочитаемый скор.
DISTANCE_OPS: dict[str, str] = {
    "cosine": "<=>",
    "l2": "<->",
    "inner_product": "<#>",
}


class StoreError(RuntimeError):
    """Проблема с хранилищем или несоответствие индекса конфигу."""


@dataclass(frozen=True)
class StoreParams:
    dsn: str
    table: str
    distance: str
    dimensions: int
    index: str = "none"

    @property
    def meta_table(self) -> str:
        """Таблица метаданных привязана к таблице чанков.

        Раньше она называлась просто ``index_meta`` и была одна на всю базу.
        Из-за этого фикстура тестов, работающая с таблицей ``chunks_test``,
        сносила метаданные боевого индекса: чанки оставались, а запись о том,
        чем они построены, исчезала. Проверка соответствия после этого молчала
        не потому, что всё в порядке, а потому, что сравнивать было не с чем.
        """
        return f"{self.table}_meta"

    @property
    def operator(self) -> str:
        if self.distance not in DISTANCE_OPS:
            known = ", ".join(DISTANCE_OPS)
            raise StoreError(f"неизвестная метрика '{self.distance}'. Известные: {known}")
        return DISTANCE_OPS[self.distance]


def to_similarity(distance: float, metric: str) -> float:
    """Расстояние pgvector → скор, где больше значит релевантнее.

    Наружу в ``/search`` и в eval-дампы уходит именно это число, и оно должно
    расти вместе с релевантностью, иначе разбирать промахи глазами невозможно.
    """
    if metric == "cosine":
        return 1.0 - distance
    if metric == "inner_product":
        return -distance
    return -distance


class PgVectorStore:
    """Таблица чанков с векторным столбцом."""

    def __init__(self, params: StoreParams) -> None:
        self.params = params
        self._conn = None

    # --- соединение -------------------------------------------------------

    def connect(self):
        if self._conn is None or self._conn.closed:
            try:
                import psycopg
                from pgvector.psycopg import register_vector
            except ImportError as exc:  # pragma: no cover
                raise StoreError(
                    "нужны пакеты psycopg и pgvector: pip install 'psycopg[binary]' pgvector"
                ) from exc
            try:
                self._conn = psycopg.connect(self.params.dsn, autocommit=True)
            except Exception as exc:  # noqa: BLE001
                raise StoreError(
                    f"не удалось подключиться к базе: {exc}\n"
                    "Если запускаете вне Docker, поднимите базу: docker compose up -d db"
                ) from exc
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    # --- схема ------------------------------------------------------------

    def existing_dimensions(self) -> int | None:
        """Размерность вектора в уже существующей таблице, если она есть."""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s AND a.attname = 'embedding' AND NOT a.attisdropped
            """,
            (self.params.table,),
        ).fetchone()
        if row is None:
            return None
        match = re.search(r"vector\((\d+)\)", row[0])
        return int(match.group(1)) if match else None

    def create_schema(self) -> None:
        conn = self.connect()
        table = self.params.table

        # CREATE TABLE IF NOT EXISTS молча оставит таблицу с другой
        # размерностью вектора, и расхождение вылезет глубоко внутри вставки
        # невнятным DataException. Проверяем заранее и объясняем, что делать.
        existing = self.existing_dimensions()
        if existing is not None and existing != self.params.dimensions:
            raise StoreError(
                f"таблица {table} уже существует с размерностью вектора {existing}, "
                f"а конфиг требует {self.params.dimensions}. "
                "Такое бывает после смены модели эмбеддингов. "
                "Пересоздайте индекс: kzrag-index build --rebuild"
            )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                chunk_id      TEXT PRIMARY KEY,
                text          TEXT NOT NULL,
                articles      TEXT[] NOT NULL,
                spans         JSONB NOT NULL,
                article_title TEXT NOT NULL DEFAULT '',
                section       TEXT NOT NULL DEFAULT '',
                chapter       TEXT NOT NULL DEFAULT '',
                char_start    INTEGER NOT NULL DEFAULT 0,
                char_end      INTEGER NOT NULL DEFAULT 0,
                embedding     vector({self.params.dimensions}) NOT NULL
            )
            """
        )
        # Поиск по номеру статьи нужен отладочному CLI, а не самому retrieval.
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_articles_idx ON {table} USING GIN (articles)"
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.params.meta_table} (
                id             INTEGER PRIMARY KEY DEFAULT 1,
                schema_version TEXT NOT NULL,
                payload        JSONB NOT NULL,
                built_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (id = 1)
            )
            """
        )

    def drop(self) -> None:
        conn = self.connect()
        conn.execute(f"DROP TABLE IF EXISTS {self.params.table}")
        conn.execute(f"DROP TABLE IF EXISTS {self.params.meta_table}")

    # --- запись -----------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Записать чанки с векторами.

        При конфликте обновляются **все** колонки. Раньше обновлялись только
        текст, статьи, пункты и вектор, а заголовок статьи, раздел, глава и
        границы оставались от прошлой сборки: строка получалась смешанной, а
        ``article_title`` не декоративен — он уходит в контекст модели.
        """
        if len(chunks) != len(vectors):
            raise StoreError(
                f"чанков {len(chunks)}, векторов {len(vectors)} — рассинхрон индексации"
            )
        if not chunks:
            return 0
        if vectors.shape[1] != self.params.dimensions:
            raise StoreError(
                f"размерность вектора {vectors.shape[1]} не совпадает со схемой "
                f"({self.params.dimensions}). Пересоздайте индекс."
            )

        conn = self.connect()
        rows = [
            (
                c.chunk_id,
                c.text,
                list(c.articles),
                json.dumps([list(s) for s in c.spans], ensure_ascii=False),
                c.article_title,
                c.section,
                c.chapter,
                c.char_start,
                c.char_end,
                vec,
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        with conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.params.table}
                    (chunk_id, text, articles, spans, article_title, section, chapter,
                     char_start, char_end, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    articles = EXCLUDED.articles,
                    spans = EXCLUDED.spans,
                    article_title = EXCLUDED.article_title,
                    section = EXCLUDED.section,
                    chapter = EXCLUDED.chapter,
                    char_start = EXCLUDED.char_start,
                    char_end = EXCLUDED.char_end,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return len(rows)

    def write_meta(self, payload: dict[str, Any]) -> None:
        conn = self.connect()
        conn.execute(
            f"""
            INSERT INTO {self.params.meta_table} (id, schema_version, payload, built_at)
            VALUES (1, %s, %s, now())
            ON CONFLICT (id) DO UPDATE SET
                schema_version = EXCLUDED.schema_version,
                payload = EXCLUDED.payload,
                built_at = now()
            """,
            (SCHEMA_VERSION, json.dumps(payload, ensure_ascii=False)),
        )

    def read_meta(self) -> dict[str, Any] | None:
        from psycopg import errors as pg_errors

        conn = self.connect()
        try:
            row = conn.execute(
                f"SELECT payload, built_at FROM {self.params.meta_table} WHERE id = 1"
            ).fetchone()
        except pg_errors.UndefinedTable:
            # Единственный законный случай: индекс ещё не строили. Всё
            # остальное — сломанные права, битая таблица, потерянное
            # соединение — раньше попадало под тот же except и превращалось
            # в «метаданных нет». Дальше по цепочке это давало пустой
            # provenance и None в ключах сопоставимости, а два прогона с
            # None сравнивались друг с другом как сопоставимые.
            return None
        if row is None:
            return None
        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return {**payload, "built_at": row[1].isoformat()}

    # --- чтение -----------------------------------------------------------

    def count(self) -> int:
        conn = self.connect()
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {self.params.table}").fetchone()[0]
        except Exception:  # noqa: BLE001 — таблицы может не быть вовсе
            return 0

    def _row_to_chunk(self, row) -> Chunk:
        chunk_id, text, articles, spans, title, section, chapter, cs, ce = row[:9]
        parsed = spans if isinstance(spans, list) else json.loads(spans)
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            articles=tuple(articles),
            spans=tuple((a, c) for a, c in parsed),
            article_title=title,
            section=section,
            chapter=chapter,
            char_start=cs,
            char_end=ce,
        )

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        conn = self.connect()
        rows = conn.execute(
            f"""
            SELECT chunk_id, text, articles, spans, article_title, section, chapter,
                   char_start, char_end, embedding {self.params.operator} %s AS distance
            FROM {self.params.table}
            ORDER BY embedding {self.params.operator} %s
            LIMIT %s
            """,
            (vector, vector, k),
        ).fetchall()
        return [
            RetrievedChunk(
                chunk=self._row_to_chunk(row),
                score=to_similarity(float(row[9]), self.params.distance),
                rank=i + 1,
            )
            for i, row in enumerate(rows)
        ]

    def all_chunks(self) -> list[Chunk]:
        """Все чанки индекса. Из них строится BM25 — по тем же текстам, что и dense."""
        conn = self.connect()
        rows = conn.execute(
            f"""
            SELECT chunk_id, text, articles, spans, article_title, section, chapter,
                   char_start, char_end
            FROM {self.params.table}
            ORDER BY chunk_id
            """
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def by_article(self, article: str) -> list[Chunk]:
        """Все чанки, в которые попал текст статьи. Нужно отладочному CLI."""
        conn = self.connect()
        rows = conn.execute(
            f"""
            SELECT chunk_id, text, articles, spans, article_title, section, chapter,
                   char_start, char_end
            FROM {self.params.table}
            WHERE %s = ANY(articles)
            ORDER BY chunk_id
            """,
            (article,),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]
