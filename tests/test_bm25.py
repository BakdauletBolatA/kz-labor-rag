"""BM25 и русская морфология на корпусе с заранее известным ответом."""

from __future__ import annotations

import pytest
from conftest import make_chunk

from kz_labor_rag.retrieval.analyzers import Analyzer, tokenize
from kz_labor_rag.retrieval.bm25 import BM25Index, BM25Retriever


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self):
        assert tokenize("Статья 52. Работодатель, (вправе)!") == [
            "статья",
            "52",
            "работодатель",
            "вправе",
        ]

    def test_yo_is_folded(self):
        assert tokenize("Ещё") == ["еще"]


class TestAnalyzers:
    @pytest.mark.parametrize("name", ["lemma", "stem"])
    def test_inflected_forms_meet(self, name):
        analyze = Analyzer(name)
        assert analyze("работодателем") == analyze("работодателя") == analyze("работодатель")

    def test_lemma_gives_dictionary_form(self):
        assert Analyzer("lemma")("уволили") == ["уволить"]

    def test_stem_cuts_the_ending(self):
        assert Analyzer("stem")("уволили") == ["увол"]

    def test_unknown_analyzer_rejected(self):
        with pytest.raises(ValueError, match="неизвестный анализатор"):
            Analyzer("magic")


class TestBM25Index:
    def test_matching_document_scores_highest(self):
        index = BM25Index(
            [["отпуск", "день"], ["увольнение", "работник"], ["зарплата"]], k1=1.2, b=0.75
        )
        scores = index.scores(["увольнение"])
        assert scores.argmax() == 1
        assert scores[0] == scores[2] == 0

    def test_rare_term_outweighs_frequent_one(self):
        docs = [["работник", "отпуск"], ["работник", "вахта"], ["работник", "зарплата"]]
        index = BM25Index(docs, k1=1.2, b=0.75)
        scores = index.scores(["работник", "вахта"])
        assert scores.argmax() == 1

    def test_frequent_term_never_penalises(self):
        # idf с «1 +» под логарифмом не бывает отрицательным.
        index = BM25Index([["работник"]] * 9 + [["отпуск"]], k1=1.2, b=0.75)
        assert (index.scores(["работник"]) >= 0).all()

    def test_longer_document_is_normalised(self):
        index = BM25Index([["вахта"], ["вахта"] + ["шум"] * 20], k1=1.2, b=0.75)
        scores = index.scores(["вахта"])
        assert scores[0] > scores[1]

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError):
            BM25Index([], k1=1.2, b=0.75)


class TestRetriever:
    CHUNKS = [
        make_chunk(
            "88", ("",), "Отпуск предоставляется продолжительностью двадцать четыре дня.", "c1"
        ),
        make_chunk(
            "54", ("1",), "Не допускается расторжение договора в период нетрудоспособности.", "c2"
        ),
        make_chunk(
            "135", ("4",), "Продолжительность вахты не может превышать пятнадцать дней.", "c3"
        ),
    ]

    def retriever(self, analyzer="lemma"):
        return BM25Retriever(lambda: self.CHUNKS, Analyzer(analyzer), k1=1.2, b=0.75, version="t")

    @pytest.mark.parametrize("analyzer", ["lemma", "stem"])
    def test_finds_the_inflected_match(self, analyzer):
        hits = self.retriever(analyzer).search("Сколько длится вахта?", k=2)
        assert hits[0].chunk_id == "c3"
        assert [h.rank for h in hits] == [1, 2]

    def test_records_its_own_latency(self):
        r = self.retriever()
        r.search("отпуск", k=1)
        assert set(r.last_timings) == {"bm25"}

    def test_chunks_are_loaded_once(self):
        calls = []

        def load():
            calls.append(1)
            return self.CHUNKS

        r = BM25Retriever(load, Analyzer("stem"), k1=1.2, b=0.75, version="t")
        r.warmup()
        r.search("отпуск", k=1)
        r.search("вахта", k=1)
        assert len(calls) == 1
