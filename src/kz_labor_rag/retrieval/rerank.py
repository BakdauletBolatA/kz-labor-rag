"""Реранкинг кросс-энкодером.

Би-энкодер (e5) кодирует вопрос и чанк по отдельности и сравнивает векторы;
кросс-энкодер читает пару «вопрос + чанк» целиком и видит, отвечает ли текст на
вопрос. Это точнее и намного дороже: модель прогоняется на каждом кандидате,
поэтому реранкер получает только верх выдачи (``candidate_k``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from kz_labor_rag.types import RetrievedChunk, Retriever


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class CrossEncoderScorer:
    """Обёртка над sentence-transformers CrossEncoder с ленивой загрузкой."""

    def __init__(self, model: str, *, max_length: int, batch_size: int, device: str) -> None:
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model, max_length=self.max_length, device=self.device)
        return self._model

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        return self._load().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)


class RerankingRetriever:
    """Берёт ``candidate_k`` кандидатов у базового поиска и переупорядочивает их."""

    def __init__(
        self, base: Retriever, scorer: PairScorer, *, candidate_k: int, version: str
    ) -> None:
        self.base = base
        self.scorer = scorer
        self.candidate_k = candidate_k
        self._version = version
        self.last_timings: dict[str, float] = {}

    @property
    def version(self) -> str:
        return self._version

    def provenance(self) -> dict:
        return self.base.provenance()

    def warmup(self) -> None:
        self.base.warmup()
        self.scorer.predict([("прогрев", "прогрев")])

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]:
        candidates = list(self.base.search(query, max(k, self.candidate_k)))
        t0 = time.perf_counter()
        scores = self.scorer.predict([(query, c.text) for c in candidates]) if candidates else []
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))[:k]
        self.last_timings = {
            **getattr(self.base, "last_timings", {}),
            "rerank": (time.perf_counter() - t0) * 1000,
        }
        return [
            RetrievedChunk(chunk=candidates[i].chunk, score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]
