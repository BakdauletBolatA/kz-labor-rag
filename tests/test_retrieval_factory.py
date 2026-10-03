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
    # Тесты задают реранкер сами, а не наследуют его из конфига сервиса.
    config.data["retrieval"]["reranker"]["enabled"] = False
    # Кэш эмбеддингов создаётся при сборке поиска; боевой каталог тестам трогать нельзя.
    config.data["embeddings"]["cache"]["enabled"] = False
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


class TestForceAnswerWiring:
    def generation_config(self, threshold):
        config = config_with()
        config.data["generation"]["enabled"] = True
        config.data["generation"]["force_answer_above"] = threshold
        return config

    def test_off_by_default(self):
        from kz_labor_rag.eval.factory import build_generator
        from kz_labor_rag.eval.generator import ForceAnswerGenerator

        assert not isinstance(build_generator(config_with()), ForceAnswerGenerator)

    def test_threshold_wraps_the_generator_with_the_forced_prompt(self):
        from kz_labor_rag.eval.factory import build_generator
        from kz_labor_rag.eval.generator import ForceAnswerGenerator

        generator = build_generator(self.generation_config(0.28))
        assert isinstance(generator, ForceAnswerGenerator)
        assert generator.threshold == 0.28
        assert generator.forced.descriptor["prompt"] == "answer_force_ru@v1"
        assert generator.inner.descriptor["prompt"] != generator.forced.descriptor["prompt"]
