"""Лексический поиск Okapi BM25 по тем же чанкам, что лежат в pgvector.

Реализация своя, без зависимости: формула короткая, а так видно ровно то, что
считается. Индекс строится в памяти при прогреве: на сотнях и тысячах чанков
это доли секунды.

    score(D, Q) = Σ idf(t) · tf(t, D) · (k1 + 1) / (tf(t, D) + k1 · (1 − b + b · |D| / avgdl))
    idf(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

Вариант idf с «1 +» под логарифмом не уходит в минус для частых терминов:
иначе слово, встречающееся больше чем в половине чанков, штрафовало бы чанк.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from kz_labor_rag.types import Chunk, RetrievedChunk


class BM25Index:
    def __init__(self, documents: Sequence[list[str]], *, k1: float, b: float) -> None:
        if not documents:
            raise ValueError("BM25 по пустому набору документов")
        self.k1 = k1
        self.b = b
        self.tf = [Counter(doc) for doc in documents]
        lengths = np.array([len(doc) for doc in documents], dtype=float)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        df = Counter(term for doc in self.tf for term in doc)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([doc.get(term, 0) for doc in self.tf], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + self.norm)
        return out


class BM25Retriever:
    """BM25 за интерфейсом ``Retriever``.

    ``load_chunks`` отдаёт чанки индекса; обычно это ``store.all_chunks``,
    чтобы лексический и плотный поиск работали по одним и тем же текстам.
    """

    def __init__(
        self,
        load_chunks: Callable[[], list[Chunk]],
        analyzer: Callable[[str], list[str]],
        *,
        k1: float,
        b: float,
        version: str,
        provenance: Callable[[], dict] | None = None,
    ) -> None:
        self._load_chunks = load_chunks
        self.analyzer = analyzer
        self.k1 = k1
        self.b = b
        self._version = version
        self._provenance = provenance
        self._chunks: list[Chunk] | None = None
        self._index: BM25Index | None = None
        self.last_timings: dict[str, float] = {}

    @property
    def version(self) -> str:
        return self._version

    def provenance(self) -> dict:
        return self._provenance() if self._provenance else {}

    def warmup(self) -> None:
        if self._index is None:
            self._chunks = self._load_chunks()
            self._index = BM25Index(
                [self.analyzer(c.text) for c in self._chunks], k1=self.k1, b=self.b
            )

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]:
        self.warmup()
        t0 = time.perf_counter()
        scores = self._index.scores(self.analyzer(query))
        # Стабильная сортировка: при равных скорах порядок документа, а не случайный.
        order = np.argsort(-scores, kind="stable")[:k]
        hits = [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]
        self.last_timings = {"bm25": (time.perf_counter() - t0) * 1000}
        return hits
