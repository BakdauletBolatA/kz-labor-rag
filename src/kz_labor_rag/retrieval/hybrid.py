"""Гибридный поиск: dense и BM25, слитые через reciprocal rank fusion.

    rrf(d) = Σ_по_спискам 1 / (k + rank(d))

RRF берёт только ранги, а не скоры. Косинус dense лежит в узком диапазоне
около 0.8, BM25 — в произвольных единицах; складывать их с весами значило бы
подбирать нормировку, которая на другом корпусе развалится. Константа k
(обычно 60) гасит разницу между первым и пятым местом, чтобы один список не
диктовал результат.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from kz_labor_rag.types import RetrievedChunk, Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], *, k: int
) -> list[RetrievedChunk]:
    """Слить ранжированные списки чанков. При равенстве — порядок первого появления."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda cid: -scores[cid])
    return [
        RetrievedChunk(chunk=chunks[cid].chunk, score=scores[cid], rank=rank)
        for rank, cid in enumerate(order, start=1)
    ]


class HybridRetriever:
    """Достаёт ``candidate_k`` кандидатов из каждого поиска и сливает их RRF."""

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        candidate_k: int,
        rrf_k: int,
        version: str,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._version = version
        self.last_timings: dict[str, float] = {}

    @property
    def version(self) -> str:
        return self._version

    def provenance(self) -> dict:
        return self.dense.provenance()

    def warmup(self) -> None:
        self.dense.warmup()
        self.lexical.warmup()

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]:
        depth = max(k, self.candidate_k)
        dense_hits = self.dense.search(query, depth)
        lexical_hits = self.lexical.search(query, depth)
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([dense_hits, lexical_hits], k=self.rrf_k)[:k]
        self.last_timings = {
            **self.dense.last_timings,
            **self.lexical.last_timings,
            "fusion": (time.perf_counter() - t0) * 1000,
        }
        return fused
