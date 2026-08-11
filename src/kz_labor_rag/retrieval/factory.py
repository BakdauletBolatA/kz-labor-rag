"""Сборка поискового пайплайна по конфигу.

Одно место, где конфиг превращается в объекты. Отладочный CLI, FastAPI и
eval-харнесс собирают поиск отсюда и потому гарантированно работают на
одинаковых настройках: разъехавшиеся параметры между «поискал руками» и
«прогнал eval» — самый неприятный способ потерять доверие к цифрам.
"""

from __future__ import annotations

import os

from kz_labor_rag.config import Config
from kz_labor_rag.corpus.chunker import ChunkingParams, chunking_signature
from kz_labor_rag.embeddings.encoder import E5Encoder, EmbeddingCache, Encoder, EncoderParams
from kz_labor_rag.retrieval.dense import DenseRetriever
from kz_labor_rag.retrieval.store import PgVectorStore, StoreParams

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
            directory=config.get("embeddings.cache.dir"),
            model=params.model,
            chunking_signature=chunking_signature(build_chunking_params(config)),
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


def build_retriever(config: Config, encoder: Encoder | None = None) -> DenseRetriever:
    if config.get("retrieval.hybrid.enabled") or config.get("retrieval.reranker.enabled"):
        raise NotImplementedError(
            "гибридный поиск и reranking ещё не реализованы — это отдельные "
            "запланированные итерации. Выключите их в конфиге."
        )
    return DenseRetriever(
        store=build_store(config),
        encoder=encoder or build_encoder(config),
        version=config.version,
        candidate_k=int(config.get("retrieval.candidate_k")),
    )
