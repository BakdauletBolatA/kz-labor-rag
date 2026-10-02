"""Сборка поискового пайплайна по конфигу.

Одно место, где конфиг превращается в объекты. Отладочный CLI, FastAPI и
eval-харнесс собирают поиск отсюда и потому гарантированно работают на
одинаковых настройках: разъехавшиеся параметры между «поискал руками» и
«прогнал eval» — самый неприятный способ потерять доверие к цифрам.
"""

from __future__ import annotations

import os

from kz_labor_rag.config import Config, ConfigError
from kz_labor_rag.corpus.chunker import ChunkingParams, chunking_signature
from kz_labor_rag.embeddings.encoder import E5Encoder, EmbeddingCache, Encoder, EncoderParams
from kz_labor_rag.retrieval.analyzers import Analyzer
from kz_labor_rag.retrieval.bm25 import BM25Retriever
from kz_labor_rag.retrieval.dense import DenseRetriever
from kz_labor_rag.retrieval.hybrid import HybridRetriever
from kz_labor_rag.retrieval.rerank import CrossEncoderScorer, RerankingRetriever
from kz_labor_rag.retrieval.store import PgVectorStore, StoreParams
from kz_labor_rag.types import Retriever

DEFAULT_DSN = "postgresql://kzrag:kzrag@localhost:5432/kzrag"


def build_chunking_params(config: Config) -> ChunkingParams:
    params = ChunkingParams.from_config(config.section("chunking"))
    params.validate()
    return params


def build_encoder(config: Config, *, use_cache: bool = True) -> Encoder:
    params = EncoderParams.from_config(config.section("embeddings"))
    provider = config.get("embeddings.provider")
    if provider != "huggingface":
        raise NotImplementedError(
            f"провайдер эмбеддингов '{provider}' пока не поддержан. "
            "Baseline работает на локальной multilingual-e5-base без платных ключей."
        )

    cache = None
    if use_cache and config.get("embeddings.cache.enabled"):
        cache = EmbeddingCache(
            directory=config.path_of("embeddings.cache.dir"),
            model=params.model,
            chunking_signature=chunking_signature(build_chunking_params(config)),
            passage_prefix=params.passage_prefix,
            normalize=params.normalize,
        )
    return E5Encoder(params, cache=cache)


def build_store(config: Config) -> PgVectorStore:
    dsn = config.get_or("vector_store.dsn", None) or os.environ.get(
        "KZRAG_DATABASE_URL", DEFAULT_DSN
    )
    return PgVectorStore(
        StoreParams(
            dsn=dsn,
            table=config.get("vector_store.table"),
            distance=config.get("vector_store.distance"),
            dimensions=int(config.get("embeddings.dimensions")),
            index=config.get("vector_store.index"),
        )
    )


IMPLEMENTATIONS = ("dense", "bm25", "hybrid")


def build_retriever(config: Config, encoder: Encoder | None = None) -> Retriever:
    """Поиск по конфигу: dense, bm25 или hybrid, опционально с реранкингом.

    BM25 строится по чанкам из той же таблицы pgvector, что и dense: иначе
    методы сравнивались бы на разных текстах.
    """
    implementation = config.get("retrieval.implementation")
    if implementation not in IMPLEMENTATIONS:
        raise ConfigError(
            f"retrieval.implementation='{implementation}', допустимы: {', '.join(IMPLEMENTATIONS)}"
        )
    version = config.version
    store = build_store(config)
    dense = DenseRetriever(
        store=store,
        encoder=encoder or build_encoder(config),
        version=version,
        candidate_k=int(config.get("retrieval.candidate_k")),
    )

    base: Retriever = dense
    if implementation in ("bm25", "hybrid"):
        lexical = BM25Retriever(
            store.all_chunks,
            Analyzer(config.get("retrieval.bm25.analyzer")),
            k1=float(config.get("retrieval.bm25.k1")),
            b=float(config.get("retrieval.bm25.b")),
            version=version,
            provenance=dense.provenance,
        )
        base = lexical
        if implementation == "hybrid":
            base = HybridRetriever(
                dense,
                lexical,
                candidate_k=int(config.get("retrieval.hybrid.candidate_k")),
                rrf_k=int(config.get("retrieval.hybrid.rrf_k")),
                version=version,
            )

    if config.get("retrieval.reranker.enabled"):
        scorer = CrossEncoderScorer(
            config.get("retrieval.reranker.model"),
            max_length=int(config.get("retrieval.reranker.max_length")),
            batch_size=int(config.get("retrieval.reranker.batch_size")),
            device=config.get("retrieval.reranker.device"),
        )
        base = RerankingRetriever(
            base,
            scorer,
            candidate_k=int(config.get("retrieval.reranker.candidate_k")),
            version=version,
        )
    return base
