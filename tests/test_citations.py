"""Ссылки в ответе проверяются кодом по тому, что модель реально видела."""

from __future__ import annotations

from conftest import make_chunk

from kz_labor_rag.eval.citations import (
    REFUSAL,
    Citation,
    ground,
    parse_citations,
)
from kz_labor_rag.types import RetrievedChunk

CONTEXT = [
    RetrievedChunk(chunk=make_chunk("54", ("1", "2"), cid="c1"), score=0.9, rank=1),
    RetrievedChunk(chunk=make_chunk("83", ("",), cid="c2"), score=0.8, rank=2),
    RetrievedChunk(chunk=make_chunk("73-1", ("2",), cid="c3"), score=0.7, rank=3),
]


class TestParse:
    def test_article_and_clause(self):
        assert parse_citations("см. ст. 54 п. 1") == [Citation("54", "1")]

    def test_spelled_out_and_compound_numbers(self):
        assert parse_citations("статья 73-1, пункт 2") == [Citation("73-1", "2")]

    def test_article_without_clause(self):
        assert parse_citations("(ст. 83)") == [Citation("83")]

    def test_sources_line_wins_over_the_body(self):
        text = "Ранее в ст. 99 было иначе.\nИсточники: ст. 54 п. 1; ст. 54 п. 2"
        assert parse_citations(text) == [Citation("54", "1"), Citation("54", "2")]

    def test_duplicates_collapse(self):
        assert parse_citations("ст. 54 п. 1 и снова ст. 54 п. 1") == [Citation("54", "1")]


class TestGround:
    def test_valid_answer_keeps_only_seen_citations(self):
        raw = "Нельзя.\nИсточники: ст. 54 п. 1; ст. 52 п. 1"
        g = ground(raw, CONTEXT)
        assert g.citations == (Citation("54", "1"),)
        assert g.invalid_citations == (Citation("52", "1"),)
        assert not g.refused
        assert g.text.endswith("Источники: ст. 54 п. 1")
        assert "52" not in g.text

    def test_clause_of_a_seen_article_that_was_not_shown_is_invalid(self):
        g = ground("Да.\nИсточники: ст. 54 п. 3", CONTEXT)
        assert g.withheld

    def test_answer_without_any_valid_citation_is_withheld(self):
        g = ground("Работодатель обязан заплатить вдвое.", CONTEXT)
        assert g.text == REFUSAL
        assert g.refused and g.withheld

    def test_honest_refusal_is_not_withheld(self):
        g = ground(REFUSAL, CONTEXT)
        assert g.refused and not g.withheld
        assert g.text == REFUSAL

    def test_unnumbered_clause_is_cited_by_article(self):
        g = ground("Не менее двенадцати часов.\nИсточники: ст. 83", CONTEXT)
        assert g.citations == (Citation("83"),)


class TestParaphrasedRefusal:
    """Модель отказывается своими словами и иногда добавляет «Источники».

    На вопросах без ответа в кодексе qwen2.5 писала «В Трудовом кодексе
    Республики Казахстан ответа на этот вопрос нет» со строкой источников —
    и такой отказ засчитывался как ответ, занижая correct_refusal.
    """

    def test_paraphrase_with_sources_is_a_refusal(self):
        raw = (
            "В Трудовом кодексе Республики Казахстан ответа на этот вопрос нет.\n"
            "Источники: ст. 54 п. 1"
        )
        g = ground(raw, CONTEXT)
        assert g.refused and not g.withheld
        assert g.text == REFUSAL

    def test_variant_wording_is_a_refusal(self):
        assert ground("В Трудовом кодексе ответа на этот вопрос не нашлось.", CONTEXT).refused

    def test_answer_mentioning_absence_later_is_not_a_refusal(self):
        raw = "Нельзя: ст. 54 запрещает увольнение. Других оснований нет.\nИсточники: ст. 54 п. 1"
        assert not ground(raw, CONTEXT).refused
