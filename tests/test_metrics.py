"""Метрики проверяются на выдаче с заранее известным ответом.

Если эти тесты врут, врут все цифры в EVALUATION.md, и весь проект теряет
смысл. Поэтому проверяются в том числе неприятные случаи: несколько
обязательных статей, промах целиком, повторы статьи в выдаче.
"""

from __future__ import annotations

import pytest
from conftest import make_chunk, ranked

from kz_labor_rag.eval import metrics as M
from kz_labor_rag.types import ClauseRef, RetrievedChunk


class TestRankArticles:
    def test_dedupes_keeping_best_rank(self):
        chunks = [
            RetrievedChunk(chunk=make_chunk("52", cid="c1"), score=0.9, rank=1),
            RetrievedChunk(chunk=make_chunk("52", cid="c2"), score=0.8, rank=2),
            RetrievedChunk(chunk=make_chunk("54", cid="c3"), score=0.7, rank=3),
        ]
        # Статья 52 дала два чанка подряд; она обязана занять ранг 1 и не
        # вытеснять статью 54 на третье место в статейном ранжировании.
        assert M.rank_articles(chunks) == ["52", "54"]

    def test_empty(self):
        assert M.rank_articles([]) == []


def multi(cid: str, *articles: str, score: float = 0.9, rank: int = 1) -> RetrievedChunk:
    """Чанк, накрывший несколько статей, — обычное дело при нарезке по токенам."""
    return RetrievedChunk(chunk=make_chunk(articles[0], extra_articles=articles[1:], cid=cid),
                          score=score, rank=rank)


class TestWindowIsCountedInChunks:
    """Окно k — по чанкам, а не по статьям.

    Версия метрик 1.0 разворачивала k чанков в плоский список статей и брала
    первые k из него. При ~4 статьях на чанк recall@5 видел первый чанк с
    четвертью из пяти, то есть мерил четверть того контекста, который уходит
    в генератор.
    """

    def _spanning(self):
        # Первый чанк накрыл четыре статьи, ответ лежит во втором из двух.
        return [
            multi("c1", "45", "46", "47", "48", score=0.91, rank=1),
            multi("c2", "52", "53", score=0.88, rank=2),
        ]

    def test_required_in_second_chunk_is_found(self):
        assert M.recall_at_k(["52", "53"], self._spanning(), k=5) == 1.0

    def test_strict_hit_sees_the_whole_window(self):
        assert M.strict_hit_at_k(["52", "53"], self._spanning(), k=5) == 1.0

    def test_clause_and_article_metrics_agree_on_the_window(self):
        # Раньше здесь получалось противоречие: точный пункт ст.53 найден
        # (clause_hit=1.0), а сама ст.53 «не в топ-5» (strict_hit=0.0).
        chunks = [
            multi("c1", "45", "46", "47", "48", score=0.91, rank=1),
            RetrievedChunk(chunk=make_chunk("53", ("1",), cid="c2"), score=0.88, rank=2),
        ]
        assert M.clause_hit_at_k(ClauseRef("53", "1"), chunks, k=5) == 1.0
        assert M.strict_hit_at_k(["53"], chunks, k=5) == 1.0

    def test_beyond_the_window_is_still_a_miss(self):
        chunks = [multi(f"c{i}", str(i)) for i in range(1, 6)] + [multi("c6", "54")]
        assert M.recall_at_k(["54"], chunks, k=5) == 0.0


class TestRecallAtK:
    def test_single_required_found(self):
        assert M.recall_at_k(["54"], ranked("10", "54", "33"), k=5) == 1.0

    def test_single_required_missed(self):
        assert M.recall_at_k(["54"], ranked("10", "33", "12"), k=5) == 0.0

    def test_partial_credit_on_multiple_required(self):
        # Знаменатель — число обязательных статей, а не k.
        assert M.recall_at_k(["52", "54"], ranked("54", "10", "11"), k=5) == 0.5

    def test_respects_k_cutoff(self):
        # Статья 54 в шестом чанке: в топ-5 чанков не входит.
        assert M.recall_at_k(["54"], ranked("1", "2", "3", "4", "5", "54"), k=5) == 0.0

    def test_acceptable_articles_do_not_dilute(self):
        assert M.recall_at_k(["54"], ranked("7", "8", "54", "9", "10"), k=5) == 1.0

    def test_acceptable_articles_do_not_push_required_out(self):
        # syn_001: required ст.54, acceptable ст.52. Чанк с acceptable накрыл
        # ещё и соседей по документу. По статейному окну ст.54 вылетала за
        # край и получала ноль — то есть acceptable штрафовала, хотя
        # документация обещает обратное.
        chunks = [
            multi("c1", "52", "53", "55", "56", "57", score=0.90, rank=1),
            multi("c2", "54", score=0.87, rank=2),
        ]
        assert M.recall_at_k(["54"], chunks, k=5) == 1.0
        assert M.strict_hit_at_k(["54"], chunks, k=5) == 1.0

    def test_empty_required_is_an_error(self):
        with pytest.raises(ValueError, match="required_articles пуст"):
            M.recall_at_k([], ranked("1", "2"), k=5)


class TestStrictHitAtK:
    def test_all_required_present(self):
        assert M.strict_hit_at_k(["52", "54"], ranked("52", "54", "1"), k=5) == 1.0

    def test_one_missing_scores_zero(self):
        # Ровно тот случай, который средний recall@5 маскирует: половина
        # связки норм найдена, но ответ по такой выдаче не собрать.
        assert M.strict_hit_at_k(["52", "54"], ranked("52", "1", "2"), k=5) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "articles,expected",
        [(("54", "1", "2"), 1.0), (("1", "54", "2"), 0.5), (("1", "2", "54"), 1 / 3)],
    )
    def test_position(self, articles, expected):
        assert M.reciprocal_rank(["54"], ranked(*articles)) == pytest.approx(expected)

    def test_not_found_anywhere(self):
        assert M.reciprocal_rank(["54"], ranked("1", "2", "3")) == 0.0

    def test_counts_beyond_k_deliberately(self):
        # MRR намеренно не обрезается по k: «нашлось в восьмом чанке» и
        # «не нашлось вообще» должны различаться.
        chunks = ranked(*[str(i) for i in range(1, 8)], "54")
        assert M.reciprocal_rank(["54"], chunks) == pytest.approx(1 / 8)

    def test_takes_first_of_several_required(self):
        assert M.reciprocal_rank(["52", "54"], ranked("1", "54", "52")) == pytest.approx(0.5)

    def test_articles_of_one_chunk_share_its_rank(self):
        # Все статьи чанка найдены одним попаданием поиска. Раньше они
        # получали разные ранги по порядку в документе, и MRR зависел от
        # нарезки: тут ст.48 дала бы 1/4 вместо 1/1.
        chunks = [multi("c1", "45", "46", "47", "48")]
        assert M.reciprocal_rank(["48"], chunks) == 1.0


class TestClauseMetrics:
    def _chunks(self):
        return [
            RetrievedChunk(chunk=make_chunk("54", ("1",), cid="c1"), score=0.9, rank=1),
            RetrievedChunk(chunk=make_chunk("54", ("2", "3"), cid="c2"), score=0.8, rank=2),
            RetrievedChunk(chunk=make_chunk("52", ("1",), cid="c3"), score=0.7, rank=3),
        ]

    def test_hit_when_clause_present(self):
        assert M.clause_hit_at_k(ClauseRef("54", "2"), self._chunks(), k=5) == 1.0

    def test_miss_when_clause_in_other_article(self):
        # Пункт 1 есть и у 52-й статьи, но эталон указывает на 54-ю.
        assert M.clause_hit_at_k(ClauseRef("99", "1"), self._chunks(), k=5) == 0.0

    def test_precision_is_share_of_k_not_of_returned(self):
        # Один из пяти запрошенных чанков покрывает пункт -> 0.2.
        # Знаменатель именно k, иначе метрика подскочит на итерации,
        # которая просто начала возвращать меньше чанков.
        assert M.clause_precision_at_k(ClauseRef("54", "2"), self._chunks(), k=5) == pytest.approx(
            0.2
        )

    def test_clause_number_normalization(self):
        chunks = [RetrievedChunk(chunk=make_chunk("54", ("2-1",)), score=0.9, rank=1)]
        # «2-1.» с точкой и « 2-1 » с пробелами — тот же пункт.
        assert M.clause_hit_at_k(ClauseRef("54", " 2-1. "), chunks, k=5) == 1.0

    def test_rejects_nonpositive_k(self):
        with pytest.raises(ValueError):
            M.clause_precision_at_k(ClauseRef("54", "1"), self._chunks(), k=0)


class TestCitationValidity:
    def test_all_cited_present_in_context(self):
        assert M.citation_validity(["52", "54"], ranked("52", "54", "1")) == 1.0

    def test_hallucinated_article(self):
        # Статья 300 в контексте не показывалась — это грубая галлюцинация.
        assert M.citation_validity(["54", "300"], ranked("54", "1")) == 0.5

    def test_answer_without_citations_is_invalid(self):
        assert M.citation_validity([], ranked("54")) == 0.0


class TestQuestionMetrics:
    def test_retrieval_failure_flag(self):
        m = M.QuestionMetrics(
            question_id="q1",
            recall_at_k=0.0,
            strict_hit_at_k=0.0,
            reciprocal_rank=0.0,
            citation_validity=None,
            faithfulness=None,
            clause_precision_at_k=None,
            clause_hit_at_k=None,
            retrieved_articles=["1", "2"],
            required_articles=["54"],
        )
        assert m.is_retrieval_failure is True

    def test_partial_recall_is_not_a_failure(self):
        m = M.QuestionMetrics(
            question_id="q2",
            recall_at_k=0.5,
            strict_hit_at_k=0.0,
            reciprocal_rank=1.0,
            citation_validity=None,
            faithfulness=None,
            clause_precision_at_k=None,
            clause_hit_at_k=None,
            retrieved_articles=["52"],
            required_articles=["52", "54"],
        )
        assert m.is_retrieval_failure is False
