"""Отказ сравнивать несопоставимые прогоны.

Самое опасное, что может сделать харнесс, — нарисовать таблицу «до/после» для
двух прогонов, которые сравнивать нельзя: числа выглядят осмысленно, разница
выглядит как эффект итерации, а объясняется другой нарезкой или другим
промптом судьи.

Случай не выдуманный: фикс окна модели перенарезал корпус целиком, из 143
чанков не совпал ни один. Если бы baseline уже был посчитан, сравнение с ним
продолжило бы рисоваться.
"""

from __future__ import annotations

import json

import pytest

from kz_labor_rag.cli import main as eval_main
from kz_labor_rag.eval.compare import (
    COMPARABILITY_KEYS,
    IncomparableRunsError,
    build_table,
    find_mismatches,
)

BASE_COMPARABILITY = {
    "metrics_version": "1.0",
    "chunking_signature": "4d4ae0162b8c9a49",
    "embeddings_model": "intfloat/multilingual-e5-base",
    "dataset_sha256": "abc123",
    "generator_prompt": "answer_ru@v1",
    "judge_prompt": "faithfulness_ru@v1",
}


def run(version: str, *, recall: float = 0.5, failures=None, **overrides) -> dict:
    return {
        "version": version,
        "comparability": {**BASE_COMPARABILITY, **overrides},
        "aggregates": {
            "primary": {
                "n": 60,
                "recall@5": recall,
                "mrr": recall,
                "strict_hit@5": recall / 2,
                "faithfulness": None,
                "retrieval_failures": failures or [],
            },
            "by_language": {"ru": {"n": 60, "recall@5": recall, "retrieval_failures": []}},
        },
    }


class TestRefusal:
    def test_identical_configuration_compares(self):
        table = build_table(run("baseline-v0"), run("iter-1", recall=0.7))
        assert "recall@5" in table
        assert "+0.200" in table

    @pytest.mark.parametrize("key", sorted(COMPARABILITY_KEYS))
    def test_every_key_blocks_comparison(self, key):
        """Каждый ключ сопоставимости обязан останавливать сравнение.

        Параметризация по самому словарю: добавят ключ — тест подхватит его
        автоматически, а не промолчит.
        """
        with pytest.raises(IncomparableRunsError):
            build_table(run("baseline-v0"), run("iter-1", **{key: "другое-значение"}))

    def test_message_names_the_culprit(self):
        with pytest.raises(IncomparableRunsError) as exc:
            build_table(run("baseline-v0"), run("iter-1", chunking_signature="b9e73c04"))
        message = str(exc.value)
        assert "chunking_signature" in message
        assert "4d4ae0162b8c9a49" in message and "b9e73c04" in message
        # Сообщение обязано говорить, что делать, а не только что сломалось.
        assert "перепрогон baseline" in message

    def test_chunking_change_explained_in_plain_words(self):
        with pytest.raises(IncomparableRunsError, match="тексты чанков не те же"):
            build_table(run("a"), run("b", chunking_signature="другая"))

    def test_judge_prompt_change_blocks(self):
        with pytest.raises(IncomparableRunsError, match="промптами судьи"):
            build_table(run("a"), run("b", judge_prompt="faithfulness_ru@v2"))

    def test_generator_prompt_change_blocks(self):
        with pytest.raises(IncomparableRunsError, match="разными промптами"):
            build_table(run("a"), run("b", generator_prompt="answer_ru@v2"))

    def test_all_mismatches_listed_at_once(self):
        mismatches = find_mismatches(
            run("a"), run("b", chunking_signature="x", judge_prompt="y", dataset_sha256="z")
        )
        assert {m.key for m in mismatches} == {
            "chunking_signature",
            "judge_prompt",
            "dataset_sha256",
        }

    def test_legacy_result_without_block_is_refused(self):
        # Прогон старого формата сравнивать не с чем: подставлять значения
        # по умолчанию значило бы придумать сопоставимость.
        legacy = {"version": "старый", "aggregates": {"primary": {}}}
        with pytest.raises(IncomparableRunsError):
            build_table(legacy, run("iter-1"))


class TestTable:
    def test_shows_delta_per_metric(self):
        table = build_table(run("baseline-v0", recall=0.40), run("iter-1", recall=0.55))
        assert "0.400" in table and "0.550" in table and "+0.150" in table

    def test_unmeasured_metric_is_dash_not_zero(self):
        # faithfulness = None означает «не измеряли», и в таблице это обязано
        # отличаться от нуля.
        table = build_table(run("a"), run("b"))
        assert "faithfulness" not in table

    def test_lists_fixed_and_broken_questions(self):
        table = build_table(
            run("a", failures=["syn_001", "syn_002"]),
            run("b", failures=["syn_002", "syn_009"]),
        )
        assert "починилось вопросов: 1  syn_001" in table
        assert "сломалось вопросов : 1  syn_009" in table

    def test_language_slice(self):
        table = build_table(run("a"), run("b", recall=0.9), language="ru")
        assert "recall@5" in table


class TestCli:
    def test_refuses_and_exits_nonzero(self, tmp_path, capsys):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps(run("baseline-v0")), encoding="utf-8")
        b.write_text(
            json.dumps(run("iter-1", chunking_signature="другая")), encoding="utf-8"
        )
        assert eval_main(["compare", str(a), str(b)]) == 1
        assert "несравнимы" in capsys.readouterr().err

    def test_prints_table_when_comparable(self, tmp_path, capsys):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps(run("baseline-v0", recall=0.4)), encoding="utf-8")
        b.write_text(json.dumps(run("iter-1", recall=0.6)), encoding="utf-8")
        assert eval_main(["compare", str(a), str(b)]) == 0
        assert "+0.200" in capsys.readouterr().out
