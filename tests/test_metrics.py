"""Метрики проверяются на выдаче с заранее известным ответом.

Если эти тесты врут, врут все цифры проекта. Поэтому проверяются в том числе
неприятные случаи: несколько обязательных пунктов, промах целиком, чанк нужной
статьи без нужного пункта, чанк, накрывший несколько статей.
"""

from __future__ import annotations

import pytest
from conftest import make_chunk, ranked

from kz_labor_rag.eval import metrics as M
from kz_labor_rag.types import ClauseRef, RetrievedChunk


def ref(text: str) -> ClauseRef:
    article, _, clause = text.partition("/")
    return ClauseRef(article, clause)


def spanning(cid: str, *refs: str, rank: int = 1) -> RetrievedChunk:
    """Чанк, накрывший несколько пунктов, возможно разных статей."""
    spans = tuple(r.partition("/")[::2] for r in refs)
    articles = tuple(dict.fromkeys(a for a, _ in spans))
    chunk = make_chunk(articles[0], extra_articles=articles[1:], cid=cid)
    chunk = type(chunk)(**{**chunk.__dict__, "spans": spans})
    return RetrievedChunk(chunk=chunk, score=1.0 - rank * 0.01, rank=rank)


class TestRankArticles:
    def test_dedupes_keeping_best_rank(self):
        chunks = [
            RetrievedChunk(chunk=make_chunk("52", cid="c1"), score=0.9, rank=1),
            RetrievedChunk(chunk=make_chunk("52", cid="c2"), score=0.8, rank=2),
            RetrievedChunk(chunk=make_chunk("54", cid="c3"), score=0.7, rank=3),
        ]
        assert M.rank_articles(chunks) == ["52", "54"]

    def test_empty(self):
        assert M.rank_articles([]) == []


class TestRecallAtK:
    def test_clause_found(self):
        assert M.recall_at_k([ref("54/1")], ranked("10/1", "54/1"), k=5) == 1.0

    def test_right_article_wrong_clause_is_a_miss(self):
        assert M.recall_at_k([ref("54/1")], ranked("54/2", "54/3"), k=5) == 0.0

    def test_same_clause_number_in_other_article_is_a_miss(self):
        assert M.recall_at_k([ref("54/1")], ranked("52/1"), k=5) == 0.0

    def test_partial_credit_on_multiple_clauses(self):
        # Знаменатель — число обязательных пунктов, а не k.
        assert M.recall_at_k([ref("99/1"), ref("100/1")], ranked("99/1", "5/1"), k=5) == 0.5

    def test_respects_k_cutoff(self):
        chunks = ranked("1/1", "2/1", "3/1", "4/1", "5/1", "54/1")
        assert M.recall_at_k([ref("54/1")], chunks, k=5) == 0.0

    def test_spanning_chunk_counts_for_every_clause_it_covers(self):
        chunks = [spanning("c1", "53/3", "54/1", "54/2")]
        assert M.recall_at_k([ref("54/1"), ref("54/2")], chunks, k=5) == 1.0

    def test_clause_number_normalization(self):
        # «2-1.» с точкой и « 2-1 » с пробелами — тот же пункт.
        assert M.recall_at_k([ClauseRef("54", " 2-1. ")], ranked("54/2-1"), k=5) == 1.0

    def test_empty_required_is_an_error(self):
        with pytest.raises(ValueError, match="required_clauses пуст"):
            M.recall_at_k([], ranked("1/1"), k=5)


class TestStrictHitAtK:
    def test_all_clauses_present(self):
        assert M.strict_hit_at_k([ref("52/1"), ref("53/1")], ranked("52/1", "53/1"), k=5) == 1.0

    def test_one_missing_scores_zero(self):
        assert M.strict_hit_at_k([ref("52/1"), ref("53/1")], ranked("52/1", "53/2"), k=5) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "refs,expected",
        [(("54/1", "1/1"), 1.0), (("1/1", "54/1"), 0.5), (("1/1", "2/1", "54/1"), 1 / 3)],
    )
    def test_position(self, refs, expected):
        assert M.reciprocal_rank([ref("54/1")], ranked(*refs)) == pytest.approx(expected)

    def test_chunk_of_the_article_without_the_clause_does_not_count(self):
        assert M.reciprocal_rank([ref("54/1")], ranked("54/2", "54/1")) == pytest.approx(0.5)

    def test_not_found_anywhere(self):
        assert M.reciprocal_rank([ref("54/1")], ranked("1/1", "2/1")) == 0.0

    def test_counts_beyond_k_deliberately(self):
        # «Нашлось в восьмом чанке» и «не нашлось вообще» должны различаться.
        chunks = ranked(*[f"{i}/1" for i in range(1, 8)], "54/1")
        assert M.reciprocal_rank([ref("54/1")], chunks) == pytest.approx(1 / 8)

    def test_takes_first_of_several_required(self):
        chunks = ranked("1/1", "53/1", "52/1")
        assert M.reciprocal_rank([ref("52/1"), ref("53/1")], chunks) == pytest.approx(0.5)


class TestArticleRecall:
    """Статейный recall — отдельная, более мягкая метрика."""

    def test_counts_article_even_without_the_clause(self):
        assert M.article_recall_at_k(["54"], ranked("54/2"), k=5) == 1.0

    def test_window_is_counted_in_chunks(self):
        # Первый чанк накрыл четыре статьи, нужная — во втором.
        chunks = [spanning("c1", "45/1", "46/1", "47/1", "48/1"), spanning("c2", "52/1", rank=2)]
        assert M.article_recall_at_k(["52"], chunks, k=5) == 1.0

    def test_empty_required_is_an_error(self):
        with pytest.raises(ValueError, match="required_articles пуст"):
            M.article_recall_at_k([], ranked("1"), k=5)


class TestCitationValidity:
    def test_all_cited_present_in_context(self):
        assert M.citation_validity(["52", "54"], ranked("52", "54", "1")) == 1.0

    def test_hallucinated_article(self):
        assert M.citation_validity(["54", "300"], ranked("54", "1")) == 0.5

    def test_answer_without_citations_is_invalid(self):
        assert M.citation_validity([], ranked("54")) == 0.0


def metrics(recall: float) -> M.QuestionMetrics:
    return M.QuestionMetrics(
        question_id="q1",
        recall_at_k=recall,
        strict_hit_at_k=0.0,
        reciprocal_rank=0.0,
        article_recall_at_k=1.0,
        citation_validity=None,
        faithfulness=None,
        retrieved_articles=["54"],
        required_articles=["54"],
        required_clauses=["ст. 54 п. 1"],
    )


class TestQuestionMetrics:
    def test_failure_is_judged_by_clauses_not_articles(self):
        # Статья нашлась (article_recall=1), пункт — нет: это провал поиска.
        assert metrics(0.0).is_retrieval_failure is True

    def test_partial_recall_is_not_a_failure(self):
        assert metrics(0.5).is_retrieval_failure is False
