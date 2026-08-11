"""Прогон целиком: гейт готовности датасета, агрегаты, форма результата."""

from __future__ import annotations

import json

import pytest

from conftest import FakeGenerator, FakeJudge, FakeRetriever, ranked
from kz_labor_rag.eval.dataset import EvalDataset, EvalQuestion
from kz_labor_rag.eval.runner import DatasetNotReadyError, EvalRunner, save_result

BASE = {
    "question": "Может ли работодатель уволить работника в отпуске?",
    "lang": "ru",
    "origin": "synthetic",
    "required_articles": [54],
    "evidence": "Не допускается расторжение",
    "reviewed_by_human": True,
}


def q(qid: str, **overrides) -> EvalQuestion:
    return EvalQuestion.from_dict({"id": qid, **BASE, **overrides})


def complete_dataset(responses_ok: bool = True) -> EvalDataset:
    """Датасет, проходящий гейт: 60 ru (15 real) + 15 kk."""
    return EvalDataset(
        questions=tuple(
            q(f"r{i}", question=f"вопрос {i}", origin="real" if i < 15 else "synthetic")
            for i in range(60)
        )
        + tuple(q(f"k{i}", question=f"сұрақ {i}", lang="kk") for i in range(15)),
        path=None,
    )


class TestCompletenessGate:
    def test_incomplete_dataset_stops_the_run(self, config):
        runner = EvalRunner(config, FakeRetriever({}))
        tiny = EvalDataset(questions=(q("r1"),))
        with pytest.raises(DatasetNotReadyError) as exc:
            runner.run(tiny)
        # Сообщение обязано объяснять, почему запуск остановлен, а не просто падать.
        assert "не укомплектован" in str(exc.value)
        assert "русских вопросов 1" in str(exc.value)

    def test_gate_can_be_bypassed_explicitly(self, config):
        runner = EvalRunner(config, FakeRetriever({}))
        tiny = EvalDataset(questions=(q("r1", question="вопрос 1"),))
        result = runner.run(tiny, enforce_gate=False)
        assert result["aggregates"]["primary"]["n"] == 1

    def test_gate_disabled_in_config(self, config):
        config.data["eval"]["completeness"]["enforce"] = False
        runner = EvalRunner(config, FakeRetriever({}))
        runner.check_dataset_ready(EvalDataset(questions=(q("r1"),)))  # не должно падать


class TestAggregates:
    def test_perfect_retrieval(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked(54, 1, 2) for x in ds}
        runner = EvalRunner(config, FakeRetriever(responses))
        result = runner.run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["n"] == 60
        assert primary["recall@5"] == 1.0
        assert primary["mrr"] == 1.0
        assert primary["retrieval_failures"] == []

    def test_total_miss(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked(1, 2, 3) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["recall@5"] == 0.0
        assert primary["mrr"] == 0.0
        # Список проваленных вопросов — вход для обсуждения гипотез из пункта 5.
        assert len(primary["retrieval_failures"]) == 60

    def test_languages_are_reported_separately(self, config):
        ds = complete_dataset()
        responses = {x.question: (ranked(54) if x.lang == "ru" else ranked(1)) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        by_lang = result["aggregates"]["by_language"]
        assert by_lang["ru"]["recall@5"] == 1.0
        assert by_lang["kk"]["recall@5"] == 0.0
        # Главная цифра — только основной срез, казахский её не разбавляет.
        assert result["aggregates"]["primary"]["recall@5"] == 1.0

    def test_real_and_synthetic_reported_separately(self, config):
        ds = complete_dataset()
        responses = {x.question: (ranked(54) if x.origin == "real" else ranked(1)) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        by_origin = result["aggregates"]["by_origin"]
        assert by_origin["real"]["recall@5"] == 1.0
        assert by_origin["synthetic"]["recall@5"] == 0.0

    def test_unmeasured_metrics_are_none_not_zero(self, config):
        # Без генератора и судьи faithfulness не измерялась. Ноль здесь означал
        # бы «ответ не обоснован» и портил бы таблицу.
        ds = complete_dataset()
        responses = {x.question: ranked(54) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["faithfulness"] is None
        assert primary["n_judged"] == 0


class TestGenerationAndJudge:
    def test_faithfulness_and_citations_recorded(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked(54, 1) for x in ds}
        runner = EvalRunner(
            config,
            FakeRetriever(responses),
            generator=FakeGenerator("Нет, нельзя (ст. 54)."),
            judge=FakeJudge("supported", 1.0),
        )
        result = runner.run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["faithfulness"] == 1.0
        assert primary["citation_validity"] == 1.0
        assert primary["n_judged"] == 60

    def test_hallucinated_citation_lowers_validity(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked(54) for x in ds}
        runner = EvalRunner(
            config,
            FakeRetriever(responses),
            generator=FakeGenerator("Смотрите ст. 54 и ст. 300."),
            judge=FakeJudge("partially_supported", 0.5),
        )
        result = runner.run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["citation_validity"] == 0.5  # 54 показывалась, 300 — нет
        assert primary["faithfulness"] == 0.5


class TestResultShape:
    def test_result_is_self_describing(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked(54) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        # По этому JSON строка EVALUATION.md должна восстанавливаться целиком.
        for key in (
            "schema_version",
            "metrics_version",
            "timestamp",
            "version",
            "git",
            "config",
            "config_fingerprint",
            "chunking_signature",
            "dataset",
            "components",
            "aggregates",
            "questions",
        ):
            assert key in result, f"в результате прогона нет поля {key}"

        assert result["version"] == "test-v0"
        assert result["components"]["retriever"]["version"] == "fake-v0"
        assert len(result["questions"]) == 75

    def test_per_question_retrieval_is_debuggable(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked(1, 2, 54) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        first = result["questions"][0]["retrieved"][0]
        # Сырые скоры и id чанков нужны, чтобы глазами разбирать промахи.
        assert {"rank", "chunk_id", "article", "score", "preview"} <= set(first)
        assert isinstance(first["score"], float)

    def test_saved_file_names_carry_time_and_version(self, config, tmp_path):
        ds = complete_dataset()
        result = EvalRunner(config, FakeRetriever({x.question: ranked(54) for x in ds})).run(ds)
        path = save_result(result, tmp_path)

        assert path.name.endswith("__test-v0.json")
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == "test-v0"

    def test_existing_result_is_never_overwritten(self, config, tmp_path):
        ds = complete_dataset()
        result = EvalRunner(config, FakeRetriever({x.question: ranked(54) for x in ds})).run(ds)
        save_result(result, tmp_path)
        with pytest.raises(FileExistsError):
            save_result(result, tmp_path)
