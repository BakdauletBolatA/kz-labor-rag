"""Конфиг → конвейер поиска. Модели и база не нужны: всё загружается лениво."""

from __future__ import annotations

import pytest

from kz_labor_rag.config import ConfigError, load_config
from kz_labor_rag.retrieval.bm25 import BM25Retriever
from kz_labor_rag.retrieval.dense import DenseRetriever
from kz_labor_rag.retrieval.factory import build_retriever
from kz_labor_rag.retrieval.hybrid import HybridRetriever
from kz_labor_rag.retrieval.rerank import RerankingRetriever


def config_with(**overrides):
    config = load_config(apply_env=False)
    for dotted, value in overrides.items():
        node = config.data
        *path, last = dotted.split("__")
        for key in path:
            node = node[key]
        node[last] = value
    return config


@pytest.mark.parametrize(
    "implementation,kind",
    [("dense", DenseRetriever), ("bm25", BM25Retriever), ("hybrid", HybridRetriever)],
)
def test_implementation_picks_the_retriever(implementation, kind):
    config = config_with(retrieval__implementation=implementation)
    assert isinstance(build_retriever(config), kind)


def test_reranker_wraps_the_base():
    config = config_with(retrieval__implementation="hybrid")
    config.data["retrieval"]["reranker"]["enabled"] = True
    retriever = build_retriever(config)
    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever.base, HybridRetriever)
    assert retriever.candidate_k == config.get("retrieval.reranker.candidate_k")


def test_bm25_uses_the_configured_analyzer():
    config = config_with(retrieval__implementation="bm25")
    config.data["retrieval"]["bm25"]["analyzer"] = "stem"
    assert build_retriever(config).analyzer.name == "stem"


def test_unknown_implementation_rejected():
    with pytest.raises(ConfigError, match="implementation"):
        build_retriever(config_with(retrieval__implementation="magic"))
