"""Пайплайн без ключей API.

Требование: отсутствие ключа отключает генерацию и судью, но не ломает прогон.
Метрики поиска обязаны посчитаться, faithfulness обязана оказаться null —
а не нулём и не traceback'ом.
"""

from __future__ import annotations

import logging

import pytest
from conftest import FakeRetriever, ranked

from kz_labor_rag.eval.dataset import EvalDataset, EvalQuestion
from kz_labor_rag.eval.factory import build_generator, build_judge, missing_key_env
from kz_labor_rag.eval.generator import AnthropicGenerator, DisabledGenerator
from kz_labor_rag.eval.judge import AnthropicJudge, DisabledJudge
from kz_labor_rag.eval.runner import EvalRunner


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def with_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


@pytest.fixture
def llm_config(config):
    """Конфиг с включёнными генерацией и судьёй."""
    config.data["generation"] = {
        "enabled": True,
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "prompt_id": "answer_ru",
        "prompt_version": "v1",
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    config.data["judge"] = {
        "enabled": True,
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "prompt_id": "faithfulness_ru",
        "prompt_version": "v1",
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    return config


def question(qid: str, text: str) -> EvalQuestion:
    return EvalQuestion.from_dict(
        {
            "id": qid,
            "question": text,
            "lang": "ru",
            "origin": "synthetic",
            "required_articles": ["54"],
            "evidence": [{"article": "54", "quote": "Не допускается расторжение"}],
            "reviewed_by_human": True,
        }
    )


class TestKeyDetection:
    def test_missing_key_reported(self, no_keys):
        assert missing_key_env("anthropic") == "ANTHROPIC_API_KEY"
        assert missing_key_env("openai") == "OPENAI_API_KEY"

    def test_present_key_not_reported(self, with_keys):
        assert missing_key_env("anthropic") is None

    def test_unknown_provider_needs_no_key(self, no_keys):
        assert missing_key_env("local") is None


class TestFactoryDegradation:
    def test_generator_disabled_without_key(self, llm_config, no_keys):
        generator = build_generator(llm_config)
        assert isinstance(generator, DisabledGenerator)
        assert "ANTHROPIC_API_KEY" in generator.reason

    def test_judge_disabled_without_key(self, llm_config, no_keys):
        judge = build_judge(llm_config)
        assert isinstance(judge, DisabledJudge)
        assert "ANTHROPIC_API_KEY" in judge.reason

    def test_warning_is_logged_not_raised(self, llm_config, no_keys, caplog):
        with caplog.at_level(logging.WARNING):
            build_generator(llm_config)
        messages = [r.getMessage() for r in caplog.records]
        assert any("ANTHROPIC_API_KEY" in m for m in messages)
        # Сообщение должно говорить, что делать, а не только что сломалось.
        assert any(".env" in m for m in messages)
        assert any("recall" in m for m in messages)

    def test_real_components_built_when_key_present(self, llm_config, with_keys):
        # Клиент создаётся лениво, поэтому фейковый ключ здесь безопасен:
        # ни одного сетевого вызова не происходит.
        assert isinstance(build_generator(llm_config), AnthropicGenerator)
        assert isinstance(build_judge(llm_config), AnthropicJudge)

    def test_judge_follows_disabled_generator(self, llm_config, with_keys):
        # Судить нечего — незачем тратить 60 вызовов API, чтобы это выяснить.
        llm_config.data["generation"]["enabled"] = False
        generator = build_generator(llm_config)
        judge = build_judge(llm_config, generator=generator)
        assert isinstance(judge, DisabledJudge)
        assert "судить нечего" in judge.reason


class TestFullRunWithoutKeys:
    def _dataset(self) -> EvalDataset:
        return EvalDataset(
            questions=tuple(
                EvalQuestion.from_dict(
                    {
                        **question(f"r{i}", f"вопрос {i}").to_dict(),
                        "origin": "real" if i < 15 else "synthetic",
                    }
                )
                for i in range(60)
            )
            + tuple(question(f"k{i}", f"сұрақ {i}") for i in range(15))
        )

    def _kk(self, ds: EvalDataset) -> EvalDataset:
        return EvalDataset(
            questions=tuple(
                EvalQuestion.from_dict(
                    {**q.to_dict(), "lang": "kk" if q.id.startswith("k") else "ru"}
                )
                for q in ds
            )
        )

    def test_pipeline_completes_and_reports_search_metrics(self, llm_config, no_keys):
        dataset = self._kk(self._dataset())
        responses = {q.question: ranked("54", "1", "2") for q in dataset}

        generator = build_generator(llm_config)
        judge = build_judge(llm_config, generator=generator)
        runner = EvalRunner(llm_config, FakeRetriever(responses), generator, judge)

        result = runner.run(dataset)  # не должно бросать
        primary = result["aggregates"]["primary"]

        # Метрики поиска посчитаны полностью.
        assert primary["n"] == 60
        assert primary["recall@5"] == 1.0
        assert primary["mrr"] == 1.0

        # А LLM-метрики честно отсутствуют, а не равны нулю.
        assert primary["faithfulness"] is None
        assert primary["citation_validity"] is None
        assert primary["n_judged"] == 0

    def test_result_records_why_llm_was_skipped(self, llm_config, no_keys):
        dataset = self._kk(self._dataset())
        responses = {q.question: ranked("54") for q in dataset}

        generator = build_generator(llm_config)
        judge = build_judge(llm_config, generator=generator)
        result = EvalRunner(llm_config, FakeRetriever(responses), generator, judge).run(dataset)

        # По JSON должно быть видно, почему faithfulness пустая:
        # «не измеряли из-за отсутствия ключа», а не «система плохая».
        assert result["components"]["generator"]["backend"] == "disabled"
        assert "ANTHROPIC_API_KEY" in result["components"]["generator"]["reason"]
        assert result["components"]["judge"]["backend"] == "disabled"

    def test_no_network_calls_are_attempted(self, llm_config, no_keys, monkeypatch):
        # Страховка от регрессии: без ключа не должно быть ни одной попытки
        # создать клиента — иначе прогон превратится в 60 таймаутов.
        def explode(*args, **kwargs):
            raise AssertionError("попытка создать API-клиент без ключа")

        monkeypatch.setattr(AnthropicGenerator, "_get_client", explode)
        monkeypatch.setattr(AnthropicJudge, "_get_client", explode)

        dataset = self._kk(self._dataset())
        generator = build_generator(llm_config)
        judge = build_judge(llm_config, generator=generator)
        EvalRunner(
            llm_config, FakeRetriever({q.question: ranked("54") for q in dataset}), generator, judge
        ).run(dataset)
