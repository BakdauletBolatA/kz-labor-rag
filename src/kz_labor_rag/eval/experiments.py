"""Варианты конфигурации для сравнительных прогонов.

Общие для ``evals/retrieval_eval.py`` и ``evals/answer_eval.py``: обе таблицы
обязаны говорить об одних и тех же «нарезке» и «методе», иначе строки одной
нельзя сопоставить со строками другой.
"""

from __future__ import annotations

import copy
import logging

from kz_labor_rag.config import Config
from kz_labor_rag.indexer import build_index, index_mismatch
from kz_labor_rag.retrieval.factory import build_store

log = logging.getLogger(__name__)


def chunking(strategy: str, overlap: int = 0, header: bool = False) -> dict:
    return {
        "chunking.strategy": strategy,
        "chunking.chunk_overlap_tokens": overlap,
        "chunking.prepend_article_header": header,
    }


CHUNKINGS: dict[str, dict] = {
    "fixed512": chunking("fixed_tokens"),
    "fixed512-overlap128": chunking("fixed_tokens", overlap=128),
    "clause": chunking("clause"),
    "article": chunking("article"),
    # Заголовок статьи — отдельная гипотеза, поэтому отдельной строкой.
    "clause+header": chunking("clause", header=True),
}

EMBEDDINGS: dict[str, dict] = {
    "e5-base": {},
    "bge-m3": {
        "embeddings.model": "BAAI/bge-m3",
        "embeddings.dimensions": 1024,
        "embeddings.query_prefix": "",
        "embeddings.passage_prefix": "",
        "embeddings.max_sequence_length": 8192,
    },
}


def method(implementation: str, *, rerank: bool = False, **extra) -> dict:
    """Переопределения конфига для метода. Реранкер включается или выключается
    явно: конфиг сервиса его включает, и метод без реранкера не должен его
    наследовать — иначе baseline таблицы тихо превращается в dense+rerank."""
    return {
        "retrieval.implementation": implementation,
        "retrieval.reranker.enabled": rerank,
        **extra,
    }


METHODS: dict[str, dict] = {
    "dense": method("dense"),
    "bm25-lemma": method("bm25", **{"retrieval.bm25.analyzer": "lemma"}),
    "bm25-stem": method("bm25", **{"retrieval.bm25.analyzer": "stem"}),
    "hybrid": method("hybrid"),
    "dense+rerank": method("dense", rerank=True),
    "hybrid+rerank": method("hybrid", rerank=True),
    # Реранкеру отдаётся 40 кандидатов вместо 20: вдруг нужный пункт стоял ниже.
    "dense+rerank-k40": method("dense", rerank=True, **{"retrieval.reranker.candidate_k": 40}),
    "hybrid+rerank-k40": method("hybrid", rerank=True, **{"retrieval.reranker.candidate_k": 40}),
}


def derive(base: Config, *overrides: dict, version: str, table: str) -> Config:
    data = copy.deepcopy(base.data)
    for block in overrides:
        for dotted, value in block.items():
            node = data
            *path, last = dotted.split(".")
            for key in path:
                node = node[key]
            node[last] = value
    data["version"] = version
    data["vector_store"]["table"] = table
    # Генерация и судья здесь не участвуют: таблица только про поиск.
    data["generation"]["enabled"] = False
    data["judge"]["enabled"] = False
    return Config(data=data, path=base.path, root=base.root)


def ensure_index(config: Config, budget_config: Config) -> None:
    store = build_store(config)
    if store.count() and not index_mismatch(config, store):
        return
    log.info("Строится индекс %s", config.get("vector_store.table"))
    build_index(config, store=store, budget_config=budget_config)


def table_name(chunking_name: str, embeddings: str) -> str:
    """Своя таблица pgvector на каждую пару «нарезка × модель»."""
    name = f"exp_{chunking_name}_{embeddings}"
    for ch in "-+.":
        name = name.replace(ch, "_")
    return name


def cell_config(base: Config, chunking_name: str, embeddings: str, method: str) -> Config:
    """Конфиг ячейки с построенным индексом.

    Границы чанков задаёт базовая модель: так разные модели эмбеддингов
    получают одинаковые тексты чанков.
    """
    overrides = CHUNKINGS[chunking_name]
    config = derive(
        base,
        overrides,
        EMBEDDINGS[embeddings],
        METHODS[method],
        version=f"{chunking_name}/{embeddings}/{method}",
        table=table_name(chunking_name, embeddings),
    )
    ensure_index(config, derive(base, overrides, version="budget", table="unused"))
    return config
