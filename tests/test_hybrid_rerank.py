"""RRF и реранкинг на выдаче с известным ответом."""

from __future__ import annotations

import pytest
from conftest import FakeRetriever, ranked

from kz_labor_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from kz_labor_rag.retrieval.rerank import RerankingRetriever


def ids(hits):
    return [h.chunk_id for h in hits]


class Timed(FakeRetriever):
    def __init__(self, responses, step):
        super().__init__(responses)
        self.step = step
        self.last_timings = {}
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def provenance(self):
        return {"chunking_signature": "sig"}

    def search(self, query, k):
        self.last_timings = {self.step: 1.0}
        return super().search(query, k)


class TestRRF:
    def test_agreement_beats_a_single_first_place(self):
        dense = ranked("1/1", "2/1", "3/1")
        lexical = ranked("4/1", "2/1", "5/1")
        fused = reciprocal_rank_fusion([dense, lexical], k=60)
        # 2/1 второй в обоих списках: 2/62 > 1/61.
        assert fused[0].chunk_id == dense[1].chunk_id

    def test_score_is_sum_of_reciprocal_ranks(self):
        a, b = ranked("1/1"), ranked("1/1")
        b[0] = type(b[0])(chunk=a[0].chunk, score=0.0, rank=1)
        fused = reciprocal_rank_fusion([a, b], k=60)
        assert fused[0].score == pytest.approx(2 / 61)

    def test_ranks_are_renumbered(self):
        fused = reciprocal_rank_fusion([ranked("1/1", "2/1"), ranked("3/1")], k=60)
        assert [h.rank for h in fused] == [1, 2, 3]

    def test_ignores_raw_scores(self):
        # Ранги одинаковые — скоры разного масштаба не должны ничего решать.
        dense = ranked("1/1", "2/1")
        lexical = [type(h)(chunk=h.chunk, score=h.score * 1000, rank=h.rank) for h in dense]
        assert ids(reciprocal_rank_fusion([dense, lexical], k=60)) == ids(dense)


class TestHybridRetriever:
    def test_fuses_candidates_and_cuts_to_k(self):
        dense = Timed({"q": ranked("1/1", "2/1", "3/1")}, "dense")
        lexical = Timed({"q": ranked("3/1", "4/1")}, "bm25")
        hybrid = HybridRetriever(dense, lexical, candidate_k=10, rrf_k=60, version="h")
        hits = hybrid.search("q", k=2)
        assert len(hits) == 2
        assert dense.calls == [("q", 10)] and lexical.calls == [("q", 10)]
        assert set(hybrid.last_timings) == {"dense", "bm25", "fusion"}

    def test_warmup_reaches_both(self):
        dense, lexical = Timed({}, "dense"), Timed({}, "bm25")
        HybridRetriever(dense, lexical, candidate_k=10, rrf_k=60, version="h").warmup()
        assert dense.warmed and lexical.warmed


class FakeScorer:
    """Скор = длина пересечения слов вопроса и текста — достаточно для проверки порядка."""

    def __init__(self, favourite: str):
        self.favourite = favourite
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [1.0 if self.favourite in text else 0.0 for _, text in pairs]


class TestReranking:
    def test_reorders_candidates_by_the_scorer(self):
        base = Timed({"q": ranked("1/1", "2/1", "3/1")}, "dense")
        scorer = FakeScorer(favourite="статьи 3")
        r = RerankingRetriever(base, scorer, candidate_k=3, version="r")
        hits = r.search("q", k=2)
        assert hits[0].article == "3"
        assert [h.rank for h in hits] == [1, 2]

    def test_asks_base_for_candidate_k(self):
        base = Timed({"q": ranked("1/1", "2/1", "3/1")}, "dense")
        RerankingRetriever(base, FakeScorer("x"), candidate_k=20, version="r").search("q", k=5)
        assert base.calls == [("q", 20)]

    def test_records_rerank_latency_on_top_of_base(self):
        base = Timed({"q": ranked("1/1")}, "dense")
        r = RerankingRetriever(base, FakeScorer("x"), candidate_k=5, version="r")
        r.search("q", k=1)
        assert set(r.last_timings) == {"dense", "rerank"}

    def test_empty_candidates(self):
        r = RerankingRetriever(Timed({}, "dense"), FakeScorer("x"), candidate_k=5, version="r")
        assert r.search("q", k=5) == []
