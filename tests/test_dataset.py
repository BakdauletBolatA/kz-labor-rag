"""Схема датасета. Ошибка здесь — это молча испорченный эталон."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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
    "required_articles": ["54"],
    "acceptable_articles": ["52"],
    "evidence": [
        {
            "article": "54",
            "quote": "Не допускается расторжение трудового договора по инициативе работодателя",
        }
    ],
    "preferred_clause": {"article": "54", "clause": "2"},
    "reviewed_by_human": True,
}


def q(**overrides) -> EvalQuestion:
    return EvalQuestion.from_dict({**VALID, **overrides})


class TestSchema:
    def test_valid_question_parses(self):
        parsed = q()
        assert parsed.required_articles == ("54",)
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
            q(required_articles=["54"], acceptable_articles=["54"])

    def test_preferred_clause_must_point_into_required(self):
        with pytest.raises(DatasetError, match="preferred_clause"):
            q(preferred_clause={"article": "52", "clause": "1"})

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
        ds = EvalDataset(questions=(q(id="r1"), q(id="r2", origin="real"), q(id="k1", lang="kk")))
        s = ds.stats
        assert s == {
            "total": 3,
            "ru": 2,
            "kk": 1,
            "real": 1,
            "synthetic": 2,
            "with_preferred_clause": 3,
            "reviewed_by_human": 3,
            "draft_slots": 0,
        }


class TestDraftSlots:
    """Пустые слоты под будущие вопросы живут в том же файле, но не считаются."""

    def test_draft_needs_only_id_lang_origin(self):
        draft = EvalQuestion.from_dict(
            {"id": "real_001", "lang": "ru", "origin": "real", "status": "draft"}
        )
        assert draft.is_draft
        assert draft.question == ""
        assert draft.required_articles == ()

    def test_draft_without_lang_is_rejected(self):
        with pytest.raises(DatasetError, match="черновика"):
            EvalQuestion.from_dict({"id": "real_001", "status": "draft"})

    def test_ready_question_still_requires_everything(self):
        with pytest.raises(DatasetError, match="evidence"):
            EvalQuestion.from_dict(
                {
                    "id": "x",
                    "question": "в",
                    "lang": "ru",
                    "origin": "real",
                    "required_articles": ["54"],
                }
            )

    def test_drafts_do_not_count_towards_the_gate(self):
        drafts = tuple(
            EvalQuestion.from_dict(
                {"id": f"real_{i:03d}", "lang": "ru", "origin": "real", "status": "draft"}
            )
            for i in range(1, 16)
        )
        ds = EvalDataset(questions=(q(id="s1"),) + drafts)
        # 15 черновых слотов с origin=real не должны закрывать требование
        # «минимум 15 реальных вопросов».
        problems = CompletenessRule(min_ru=1, min_kk=0, min_real=15).violations(ds)
        assert problems == ["вопросов с origin='real' 0, нужно минимум 15"]
        assert ds.stats["draft_slots"] == 15
        assert ds.stats["total"] == 1

    def test_drafts_are_excluded_from_slices(self):
        draft = EvalQuestion.from_dict(
            {"id": "real_001", "lang": "ru", "origin": "real", "status": "draft"}
        )
        ds = EvalDataset(questions=(q(id="s1"), draft))
        assert [x.id for x in ds.slice("ru")] == ["s1"]

    def test_draft_survives_roundtrip(self, tmp_path):
        draft = EvalQuestion.from_dict(
            {"id": "real_001", "lang": "ru", "origin": "real", "status": "draft"}
        )
        path = tmp_path / "ds.jsonl"
        save_dataset(EvalDataset(questions=(draft,)), path)
        assert load_dataset(path).questions[0].is_draft


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
        "54": "Статья 54. Не допускается расторжение трудового договора по инициативе "
        "работодателя в период временной нетрудоспособности. 2. Положения настоящего пункта...",
        "52": "Статья 52. Основания расторжения трудового договора.",
    }

    def test_clean_dataset_passes(self):
        report = validate_against_corpus(EvalDataset(questions=(q(),)), self.CORPUS)
        assert report.ok

    def test_nonexistent_article_caught(self):
        report = validate_against_corpus(
            EvalDataset(
                questions=(
                    q(
                        required_articles=["999"],
                        preferred_clause=None,
                        evidence=[{"article": "999", "quote": "текст несуществующей статьи"}],
                    ),
                )
            ),
            self.CORPUS,
        )
        assert report.missing_articles == [("q001", "999")]

    def test_fabricated_quote_caught(self):
        # Ровно та ошибка, ради которой цитата обязательна: статья существует,
        # но обоснование к ней придумано.
        report = validate_against_corpus(
            EvalDataset(
                questions=(
                    q(
                        evidence=[
                            {"article": "54", "quote": "работодатель вправе уволить кого угодно"}
                        ]
                    ),
                )
            ),
            self.CORPUS,
        )
        assert report.quote_not_found == ["q001 (цитата к ст. 54)"]

    def test_quote_matching_ignores_whitespace_and_case(self):
        report = validate_against_corpus(
            EvalDataset(
                questions=(
                    q(
                        evidence=[
                            {
                                "article": "54",
                                "quote": "НЕ ДОПУСКАЕТСЯ   расторжение\nтрудового договора",
                            }
                        ]
                    ),
                )
            ),
            self.CORPUS,
        )
        assert report.quote_not_found == []


class TestClauseValidation:
    """Наличие пункта проверяется по списку номеров, а не поиском по тексту.

    Парсер выносит номер пункта в отдельное поле, и в тексте статьи его больше
    нет. Прежняя эвристика «есть ли в теле подстрока '2.'» врала в обе стороны.
    """

    CORPUS = {"54": "Статья 54. Не допускается расторжение трудового договора."}

    def test_existing_clause_passes(self):
        report = validate_against_corpus(
            EvalDataset(
                questions=(q(evidence=[{"article": "54", "quote": "Не допускается расторжение"}]),)
            ),
            self.CORPUS,
            {"54": ("1", "2")},
        )
        assert report.missing_clauses == []

    def test_nonexistent_clause_caught(self):
        report = validate_against_corpus(
            EvalDataset(
                questions=(
                    q(
                        preferred_clause={"article": "54", "clause": "9"},
                        evidence=[{"article": "54", "quote": "Не допускается расторжение"}],
                    ),
                )
            ),
            self.CORPUS,
            {"54": ("1", "2")},
        )
        assert report.missing_clauses == [("q001", "ст. 54 п. 9")]

    def test_check_is_skipped_without_clause_map(self):
        # Лучше не проверять вовсе, чем угадывать по подстроке.
        report = validate_against_corpus(
            EvalDataset(
                questions=(
                    q(
                        preferred_clause={"article": "54", "clause": "9"},
                        evidence=[{"article": "54", "quote": "Не допускается расторжение"}],
                    ),
                )
            ),
            self.CORPUS,
        )
        assert report.missing_clauses == []


class TestShippedDataset:
    """Датасет, который лежит в репозитории. Регрессионная защита разметки."""

    PATH = Path("evals/datasets/kz_labor_v1.jsonl")

    @pytest.fixture(scope="class")
    def dataset(self):
        if not self.PATH.exists():
            pytest.skip("датасет ещё не собран")
        return load_dataset(self.PATH)

    def test_composition(self, dataset):
        s = dataset.stats
        assert s["synthetic"] == 45
        assert s["ru"] == 45
        assert s["draft_slots"] == 15

    def test_real_slots_are_reserved_but_empty(self, dataset):
        drafts = [x for x in dataset if x.is_draft]
        assert [x.id for x in drafts] == [f"real_{i:03d}" for i in range(1, 16)]
        assert all(x.origin == "real" and not x.question for x in drafts)

    def test_topic_limit_respected(self, dataset):
        # Лимит считается внутри языка: казахский срез намеренно повторяет темы
        # русского, и общий счётчик запретил бы это на ровном месте — на 8 ru
        # плюс 8 kk по одной теме. Сборщик проверяет ровно так же.
        counts = Counter((x.lang, tag) for x in dataset.ready for tag in x.tags)
        assert counts, "у вопросов должны быть теги"
        over = {key: n for key, n in counts.items() if n > 8}
        assert not over, f"превышен лимит 8 вопросов на тему: {over}"

    def test_required_topics_are_covered(self, dataset):
        counts = Counter(tag for x in dataset.ready for tag in x.tags)
        for tag in (
            "увольнение",
            "отпуск",
            "рабочее-время",
            "оплата",
            "испытательный-срок",
            "дисциплина",
            "изменение-условий",
            "срочный-договор",
            "совместительство",
        ):
            assert counts.get(tag, 0) > 0, f"тема '{tag}' не покрыта"

    def test_multi_article_questions_exist(self, dataset):
        # Интересные случаи — те, где ответ собирается из связки статей.
        multi = [x.id for x in dataset.ready if len(x.required_articles) > 1]
        assert len(multi) >= 5, f"многосоставных вопросов всего {len(multi)}"

    def test_every_question_has_evidence(self, dataset):
        assert all(x.evidence and all(len(e.quote) > 40 for e in x.evidence) for x in dataset.ready)

    def test_questions_avoid_code_language(self, dataset):
        # Вопрос, написанный терминами статьи, retrieval находит тривиально,
        # и baseline оказывается завышен.
        canned = (
            "каков порядок",
            "в соответствии с",
            "настоящего кодекса",
            "предусмотренных подпунктами",
            "регламентируется",
        )
        offenders = [
            x.id for x in dataset.ready if any(phrase in x.question.lower() for phrase in canned)
        ]
        assert not offenders, f"вопросы написаны языком кодекса: {offenders}"

    def test_preferred_clause_points_into_required(self, dataset):
        for x in dataset.ready:
            if x.preferred_clause:
                assert x.preferred_clause.article in x.required_articles

    @pytest.mark.skipif(
        not Path("data/raw/adilet_K1500000414_rus.html").exists(),
        reason="сырой HTML не скачан",
    )
    def test_markup_matches_the_real_corpus(self, dataset):
        from kz_labor_rag.corpus.parser import parse_file

        code = parse_file("data/raw/adilet_K1500000414_rus.html")
        report = validate_against_corpus(
            dataset, code.article_texts(), {a.number: a.clause_numbers for a in code}
        )
        assert report.missing_articles == []
        assert report.quote_not_found == []
        assert report.missing_clauses == []

        repealed = {a.number for a in code if a.is_repealed}
        cited = {n for x in dataset.ready for n in x.required_articles}
        assert not (cited & repealed), "эталон ссылается на исключённые статьи"
