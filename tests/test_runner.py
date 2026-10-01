"""Прогон целиком: гейт готовности датасета, агрегаты, форма результата."""

from __future__ import annotations

import json

import pytest
from conftest import FakeGenerator, FakeJudge, FakeRetriever, make_chunk, ranked

from kz_labor_rag.eval.dataset import EvalDataset, EvalQuestion
from kz_labor_rag.eval.runner import EvalRunner, NoVerifiedQuestionsError, save_result
from kz_labor_rag.types import RetrievedChunk

BASE = {
    "question": "Может ли работодатель уволить работника в отпуске?",
    "lang": "ru",
    "origin": "synthetic",
    "type": "condition",
    "required_articles": ["54"],
    "required_clauses": [{"article": "54", "clause": "1"}],
    "evidence": [{"article": "54", "quote": "Не допускается расторжение"}],
    "verified": True,
}


def q(qid: str, **overrides) -> EvalQuestion:
    return EvalQuestion.from_dict({"id": qid, **BASE, **overrides})


def complete_dataset(responses_ok: bool = True) -> EvalDataset:
    """60 ru (15 real) + 15 kk, все проверены."""
    return EvalDataset(
        questions=tuple(
            q(f"r{i}", question=f"вопрос {i}", origin="real" if i < 15 else "synthetic")
            for i in range(60)
        )
        + tuple(q(f"k{i}", question=f"сұрақ {i}", lang="kk") for i in range(15)),
        path=None,
    )


class TestOnlyVerifiedQuestionsAreCounted:
    """Метрики считаются только по вопросам, проверенным человеком.

    Непроверенная разметка может быть просто неверной, и метрика на ней мерила
    бы ошибки эталона, а не поиска. Сколько вопросов реально посчитано, пишется
    в каждый результат: цифра без n ничего не значит.
    """

    def test_unverified_questions_are_skipped(self, config):
        ds = EvalDataset(
            questions=(
                q("r1", question="в1"),
                q("r2", question="в2", origin="real"),
                q("r3", question="в3", verified=False),
            )
        )
        result = EvalRunner(config, FakeRetriever({})).run(ds)
        assert result["aggregates"]["primary"]["n"] == 2
        assert [x["id"] for x in result["questions"]] == ["r1", "r2"]
        assert result["dataset"]["evaluated"] == {
            "n": 2,
            "real": 1,
            "synthetic": 1,
            "unanswerable": 0,
            "skipped_unverified": 1,
        }

    def test_no_verified_questions_stops_the_run(self, config):
        ds = EvalDataset(questions=(q("r1", verified=False),))
        with pytest.raises(NoVerifiedQuestionsError, match="ни одного проверенного"):
            EvalRunner(config, FakeRetriever({})).run(ds)

    def test_unanswerable_questions_are_counted_but_not_scored(self, config):
        unanswerable = EvalQuestion.from_dict(
            {
                "id": "u1",
                "question": "Какая ставка ИПН?",
                "lang": "ru",
                "origin": "synthetic",
                "type": "unanswerable",
                "notes": "Налоговый кодекс",
                "verified": True,
            }
        )
        ds = EvalDataset(questions=(q("r1"), unanswerable))
        result = EvalRunner(config, FakeRetriever({})).run(ds)
        assert result["aggregates"]["primary"]["n"] == 1
        assert result["dataset"]["evaluated"]["unanswerable"] == 1

    def test_drafts_are_not_counted_as_skipped(self, config):
        draft = EvalQuestion.from_dict(
            {"id": "real_099", "lang": "ru", "origin": "real", "status": "draft"}
        )
        ds = EvalDataset(questions=(q("r1"), draft))
        result = EvalRunner(config, FakeRetriever({})).run(ds)
        assert result["dataset"]["evaluated"]["skipped_unverified"] == 0


class TestAggregates:
    def test_perfect_retrieval(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked("54/1", "1", "2") for x in ds}
        runner = EvalRunner(config, FakeRetriever(responses))
        result = runner.run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["n"] == 60
        assert primary["recall@5"] == 1.0
        assert primary["mrr"] == 1.0
        assert primary["retrieval_failures"] == []

    def test_total_miss(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked("1", "2", "3") for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["recall@5"] == 0.0
        assert primary["mrr"] == 0.0
        # Список проваленных вопросов — вход для обсуждения гипотез из пункта 5.
        assert len(primary["retrieval_failures"]) == 60

    def test_languages_are_reported_separately(self, config):
        ds = complete_dataset()
        responses = {x.question: (ranked("54/1") if x.lang == "ru" else ranked("1")) for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        by_lang = result["aggregates"]["by_language"]
        assert by_lang["ru"]["recall@5"] == 1.0
        assert by_lang["kk"]["recall@5"] == 0.0
        # Главная цифра — только основной срез, казахский её не разбавляет.
        assert result["aggregates"]["primary"]["recall@5"] == 1.0

    def test_real_and_synthetic_reported_separately(self, config):
        ds = complete_dataset()
        responses = {
            x.question: (ranked("54/1") if x.origin == "real" else ranked("1")) for x in ds
        }
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        by_origin = result["aggregates"]["by_origin"]
        assert by_origin["real"]["recall@5"] == 1.0
        assert by_origin["synthetic"]["recall@5"] == 0.0

    def test_unmeasured_metrics_are_none_not_zero(self, config):
        # Без генератора и судьи faithfulness не измерялась. Ноль здесь означал
        # бы «ответ не обоснован» и портил бы таблицу.
        ds = complete_dataset()
        responses = {x.question: ranked("54/1") for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        primary = result["aggregates"]["primary"]
        assert primary["faithfulness"] is None
        assert primary["n_judged"] == 0


class TestGenerationAndJudge:
    def test_faithfulness_and_citations_recorded(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked("54/1", "1") for x in ds}
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
        responses = {x.question: ranked("54/1") for x in ds}
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
        responses = {x.question: ranked("54/1") for x in ds}
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
            "comparability",
            "index",
            "dataset",
            "components",
            "aggregates",
            "questions",
        ):
            assert key in result, f"в результате прогона нет поля {key}"

        assert result["version"] == "test-v0"
        assert result["components"]["retriever"]["version"] == "fake-v0"
        assert len(result["questions"]) == 75

    def test_comparability_covers_window_and_context_format(self, config):
        """Оба ключа добавлены по итогам аудита.

        Без ``k`` смена окна 5 -> 10 не роняла сравнение: главные метрики
        просто выпадали из таблицы прочерками. Без ``context_format`` правка
        функции, собирающей промпт, меняла вход модели молча — реестр
        промптов стережёт только текстовые файлы.
        """
        ds = complete_dataset()
        responses = {x.question: ranked("54/1") for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        block = result["comparability"]
        assert block["k"] == config.get("eval.k")
        assert block["context_format"]

    def test_retrieved_dump_lists_every_article_of_a_chunk(self, config):
        # Поле "article" — головная статья, «для отладки и отображения».
        # Чанк регулярно накрывает несколько, и по дампу это должно быть видно.
        ds = complete_dataset()
        chunk = RetrievedChunk(
            chunk=make_chunk("45", extra_articles=("46", "47")), score=0.9, rank=1
        )
        responses = {x.question: [chunk] for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        first = result["questions"][0]["retrieved"][0]
        assert first["articles"] == ["45", "46", "47"]

    def test_retrieved_dump_shows_clauses_of_every_article(self, config):
        # chunk.clauses отдаёт пункты головной статьи. В дампе, по которому
        # разбирают промахи, это противоречило бы clause-метрике: пункт
        # найден, а в списке его нет, потому что он у соседней статьи.
        ds = complete_dataset()
        chunk = RetrievedChunk(
            chunk=make_chunk("45", ("1",), extra_articles=("46",), cid="c1"),
            score=0.9,
            rank=1,
        )
        responses = {x.question: [chunk] for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        assert result["questions"][0]["retrieved"][0]["clauses"] == ["45/1"]

    def test_per_question_retrieval_is_debuggable(self, config):
        ds = complete_dataset()
        responses = {x.question: ranked("1", "2", "54/1") for x in ds}
        result = EvalRunner(config, FakeRetriever(responses)).run(ds)

        first = result["questions"][0]["retrieved"][0]
        # Сырые скоры и id чанков нужны, чтобы глазами разбирать промахи.
        assert {"rank", "chunk_id", "article", "score", "preview"} <= set(first)
        assert isinstance(first["score"], float)

    def test_saved_file_names_carry_time_and_version(self, config, tmp_path):
        ds = complete_dataset()
        result = EvalRunner(config, FakeRetriever({x.question: ranked("54/1") for x in ds})).run(ds)
        path = save_result(result, tmp_path)

        assert path.name.endswith("__test-v0.json")
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == "test-v0"

    def test_existing_result_is_never_overwritten(self, config, tmp_path):
        ds = complete_dataset()
        result = EvalRunner(config, FakeRetriever({x.question: ranked("54/1") for x in ds})).run(ds)
        save_result(result, tmp_path)
        with pytest.raises(FileExistsError):
            save_result(result, tmp_path)


class TestReviewFlag:
    def test_default_is_unreviewed(self):
        # Дефолт false: вопрос считается непроверенным, пока не сказано обратное.
        raw = {k: v for k, v in BASE.items() if k != "verified"}
        assert EvalQuestion.from_dict({"id": "r1", **raw}).verified is False

    def test_review_flag_survives_roundtrip(self, tmp_path):
        from kz_labor_rag.eval.dataset import load_dataset, save_dataset

        ds = EvalDataset(
            questions=(
                q("r1", question="в", verified=True),
                q("r2", question="в2", verified=False),
            )
        )
        path = tmp_path / "ds.jsonl"
        save_dataset(ds, path)
        loaded = load_dataset(path)
        assert [x.verified for x in loaded] == [True, False]


class TestRecallIsCountedInClauses:
    """recall@k и MRR считаются по пунктам, а не по статьям.

    Ответ на вопрос лежит в конкретном пункте. Чанк с другим пунктом той же
    статьи генератору ответа не даёт, но статейная метрика засчитывала его как
    попадание: в ст. 52 больше двадцати оснований увольнения, и любой её кусок
    «находил» нужную статью.
    """

    def run_one(self, config, hits):
        question = q("r1", question="вопрос про ст. 54 п. 1")
        dataset = EvalDataset(questions=(question,))
        retriever = FakeRetriever({question.question: hits})
        return EvalRunner(config, retriever).run(dataset)

    def test_right_article_wrong_clause_is_a_miss(self, config):
        hits = [RetrievedChunk(chunk=make_chunk("54", ("2",)), score=0.9, rank=1)]
        primary = self.run_one(config, hits)["aggregates"]["primary"]
        assert primary["recall@5"] == 0.0
        assert primary["mrr"] == 0.0
        # Статейная метрика остаётся, но под своим именем.
        assert primary["article_recall@5"] == 1.0

    def test_mrr_ranks_the_chunk_with_the_clause(self, config):
        hits = [
            RetrievedChunk(chunk=make_chunk("54", ("2",), cid="c1"), score=0.9, rank=1),
            RetrievedChunk(chunk=make_chunk("54", ("1",), cid="c2"), score=0.8, rank=2),
        ]
        primary = self.run_one(config, hits)["aggregates"]["primary"]
        assert primary["recall@5"] == 1.0
        assert primary["mrr"] == 0.5
