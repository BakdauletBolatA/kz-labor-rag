"""Схема датасета. Ошибка здесь — это молча испорченный эталон."""

from __future__ import annotations

import json

import pytest

from kz_labor_rag.eval.dataset import (
    CompletenessRule,
    DatasetError,
    EvalDataset,
    EvalQuestion,
    load_dataset,
    save_dataset,
    validate_against_corpus,
)

VALID = {
    "id": "q001",
    "question": "Может ли работодатель уволить работника, находящегося в отпуске?",
    "lang": "ru",
    "origin": "synthetic",
    "required_articles": [54],
    "acceptable_articles": [52],
    "evidence": "Не допускается расторжение трудового договора по инициативе работодателя",
    "preferred_clause": {"article": 54, "clause": "2"},
    "reviewed_by_human": True,
}


def q(**overrides) -> EvalQuestion:
    return EvalQuestion.from_dict({**VALID, **overrides})


class TestSchema:
    def test_valid_question_parses(self):
        parsed = q()
        assert parsed.required_articles == (54,)
        assert parsed.preferred_clause is not None
        assert parsed.preferred_clause.clause == "2"

    @pytest.mark.parametrize(
        "field", ["id", "question", "lang", "origin", "required_articles", "evidence"]
    )
    def test_missing_required_field_rejected(self, field):
        raw = {**VALID}
        raw[field] = "" if isinstance(raw[field], str) else []
        with pytest.raises(DatasetError, match=field):
            EvalQuestion.from_dict(raw)

    def test_evidence_is_mandatory(self):
        # Без цитаты разметку невозможно отревьюировать, не открывая кодекс.
        raw = {k: v for k, v in VALID.items() if k != "evidence"}
        with pytest.raises(DatasetError, match="evidence"):
            EvalQuestion.from_dict(raw)

    def test_article_cannot_be_required_and_acceptable(self):
        with pytest.raises(DatasetError, match="однозначным"):
            q(required_articles=[54], acceptable_articles=[54])

    def test_preferred_clause_must_point_into_required(self):
        with pytest.raises(DatasetError, match="preferred_clause"):
            q(preferred_clause={"article": 52, "clause": "1"})

    def test_unknown_lang_rejected(self):
        with pytest.raises(DatasetError, match="lang"):
            q(lang="en")

    def test_unknown_origin_rejected(self):
        with pytest.raises(DatasetError, match="origin"):
            q(origin="generated")


class TestIO:
    def test_roundtrip_preserves_fields(self, tmp_path):
        ds = EvalDataset(questions=(q(), q(id="q002", preferred_clause=None)))
        path = tmp_path / "ds.jsonl"
        save_dataset(ds, path)
        loaded = load_dataset(path)
        assert len(loaded) == 2
        assert loaded.questions[0].to_dict() == ds.questions[0].to_dict()

    def test_saved_file_is_sorted_by_id(self, tmp_path):
        ds = EvalDataset(questions=(q(id="q009"), q(id="q001")))
        path = tmp_path / "ds.jsonl"
        save_dataset(ds, path)
        ids = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]
        assert ids == ["q001", "q009"]

    def test_duplicate_ids_rejected(self, tmp_path):
        path = tmp_path / "ds.jsonl"
        path.write_text(
            json.dumps(VALID, ensure_ascii=False) + "\n" + json.dumps(VALID, ensure_ascii=False),
            encoding="utf-8",
        )
        with pytest.raises(DatasetError, match="повторяющиеся id"):
            load_dataset(path)

    def test_broken_json_reports_line_number(self, tmp_path):
        path = tmp_path / "ds.jsonl"
        path.write_text(json.dumps(VALID) + "\n{ сломано\n", encoding="utf-8")
        with pytest.raises(DatasetError, match=":2:"):
            load_dataset(path)


class TestSlicesAndStats:
    def test_language_slice_is_isolated(self):
        ds = EvalDataset(questions=(q(id="r1"), q(id="k1", lang="kk")))
        assert [x.id for x in ds.slice("ru")] == ["r1"]
        assert [x.id for x in ds.slice("kk")] == ["k1"]

    def test_stats(self):
        ds = EvalDataset(
            questions=(q(id="r1"), q(id="r2", origin="real"), q(id="k1", lang="kk"))
        )
        s = ds.stats
        assert s == {
            "total": 3,
            "ru": 2,
            "kk": 1,
            "real": 1,
            "synthetic": 2,
            "with_preferred_clause": 3,
            "reviewed_by_human": 3,
        }


class TestCompletenessGate:
    def test_incomplete_dataset_lists_every_problem(self):
        ds = EvalDataset(questions=(q(),))
        problems = CompletenessRule().violations(ds)
        assert len(problems) == 3  # мало ru, мало kk, мало real
        assert any("русских" in p for p in problems)
        assert any("казахских" in p for p in problems)
        assert any("real" in p for p in problems)

    def test_unreviewed_questions_block_the_gate(self):
        ds = EvalDataset(questions=(q(reviewed_by_human=False),))
        problems = CompletenessRule(min_ru=1, min_kk=0, min_real=0).violations(ds)
        assert problems == ["не отревьюировано человеком: q001"]

    def test_complete_dataset_passes(self):
        questions = (
            tuple(q(id=f"r{i}", origin="real") for i in range(15))
            + tuple(q(id=f"s{i}") for i in range(45))
            + tuple(q(id=f"k{i}", lang="kk") for i in range(15))
        )
        assert CompletenessRule().violations(EvalDataset(questions=questions)) == []


class TestCorpusValidation:
    CORPUS = {
        54: "Статья 54. Не допускается расторжение трудового договора по инициативе "
        "работодателя в период временной нетрудоспособности. 2. Положения настоящего пункта...",
        52: "Статья 52. Основания расторжения трудового договора.",
    }

    def test_clean_dataset_passes(self):
        report = validate_against_corpus(EvalDataset(questions=(q(),)), self.CORPUS)
        assert report.ok

    def test_nonexistent_article_caught(self):
        report = validate_against_corpus(
            EvalDataset(questions=(q(required_articles=[999], preferred_clause=None),)),
            self.CORPUS,
        )
        assert report.missing_articles == [("q001", 999)]

    def test_fabricated_quote_caught(self):
        # Ровно та ошибка, ради которой цитата обязательна: статья существует,
        # но обоснование к ней придумано.
        report = validate_against_corpus(
            EvalDataset(questions=(q(evidence="работодатель вправе уволить кого угодно"),)),
            self.CORPUS,
        )
        assert report.quote_not_found == ["q001"]

    def test_quote_matching_ignores_whitespace_and_case(self):
        report = validate_against_corpus(
            EvalDataset(
                questions=(
                    q(evidence="НЕ ДОПУСКАЕТСЯ   расторжение\nтрудового договора"),
                )
            ),
            self.CORPUS,
        )
        assert report.quote_not_found == []
