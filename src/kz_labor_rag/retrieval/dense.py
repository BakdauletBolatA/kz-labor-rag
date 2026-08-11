"""Плотный поиск по косинусному сходству — реализация baseline.

Намеренно минимален: закодировать запрос, взять ближайших соседей, отдать как
есть. Ни reranking, ни гибрида, ни расширения запроса. Всё это — отдельные
итерации за тем же интерфейсом ``Retriever``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from kz_labor_rag.embeddings.encoder import Encoder
from kz_labor_rag.retrieval.store import PgVectorStore, StoreError
from kz_labor_rag.types import RetrievedChunk

log = logging.getLogger(__name__)


class DenseRetriever:
    """Поиск ближайших соседей в pgvector."""

    def __init__(
        self,
        store: PgVectorStore,
        encoder: Encoder,
        *,
        version: str,
        candidate_k: int | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self._version = version
        # В baseline candidate_k равен top_k: добирать нечего, reranking нет.
        self.candidate_k = candidate_k

    @property
    def version(self) -> str:
        return self._version

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "kind": "dense",
            "version": self._version,
            "distance": self.store.params.distance,
            "reranker": "none",
            "hybrid": "off",
            **{f"encoder_{k}": v for k, v in self.encoder.descriptor.items()},
        }

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]:
        if self.store.count() == 0:
            raise StoreError(
                "индекс пуст. Постройте его: kzrag-index build "
                "(в Docker это делает точка входа при первом запуске)"
            )
        vector = self.encoder.encode_query(query)
        limit = max(k, self.candidate_k or k)
        hits = self.store.search(vector, limit)
        return hits[:k]
