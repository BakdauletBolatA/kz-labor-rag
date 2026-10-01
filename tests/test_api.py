"""FastAPI.

API — тонкая обёртка, поэтому проверяется не логика поиска (она в своих
тестах), а контракт наружу: что ручки отдают, как ведут себя при пустом
индексе и виден ли статус индекса в /health.
"""

from __future__ import annotations

import pytest
from conftest import FakeGenerator, FakeRetriever, ranked

from kz_labor_rag.retrieval.store import StoreError

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kz_labor_rag import api  # noqa: E402

QUESTION = "можно ли уволить в отпуске"


@pytest.fixture
def client(config, monkeypatch):
    config.data["retrieval"] = {"top_k": 5}
    retriever = FakeRetriever({QUESTION: ranked("54", "52", "30")})
    monkeypatch.setattr(
        api, "_components", lambda: (config, retriever, FakeGenerator("Нельзя (ст. 54)."))
    )
    return TestClient(api.app)


class TestSearch:
    def test_returns_raw_scores_and_chunk_ids(self, client):
        body = client.get("/search", params={"q": QUESTION}).json()
        assert body["query"] == QUESTION
        assert body["version"] == "fake-v0"
        first = body["hits"][0]
        # Сырые скоры и id чанков — обязательная часть контракта: без них
        # невозможно глазами разобрать, почему поиск промахнулся.
        assert first["chunk_id"]
        assert isinstance(first["score"], float)
        assert first["articles"] == ["54"]
        assert first["rank"] == 1

    def test_k_is_respected(self, client):
        body = client.get("/search", params={"q": QUESTION, "k": 2}).json()
        assert len(body["hits"]) == 2

    def test_empty_query_rejected(self, client):
        assert client.get("/search", params={"q": ""}).status_code == 422

    def test_unknown_query_returns_empty_list(self, client):
        body = client.get("/search", params={"q": "чего-то нет"}).json()
        assert body["hits"] == []

    def test_empty_index_answers_503(self, config, monkeypatch):
        class Broken:
            version = "broken"

            def search(self, query, k):
                raise StoreError("индекс пуст")

        config.data["retrieval"] = {"top_k": 5}
        monkeypatch.setattr(api, "_components", lambda: (config, Broken(), FakeGenerator()))
        response = TestClient(api.app).get("/search", params={"q": "x"})
        assert response.status_code == 503
        assert "индекс пуст" in response.json()["detail"]


class TestAsk:
    def test_returns_answer_and_citations(self, client):
        body = client.get("/ask", params={"q": QUESTION}).json()
        assert body["answer"] == "Нельзя (ст. 54)."
        assert body["cited_articles"] == ["54"]
        assert body["error"] is None
        # Выдача возвращается вместе с ответом: видно, на чём он построен.
        assert len(body["hits"]) == 3

    def test_generator_error_is_surfaced_not_hidden(self, config, monkeypatch):
        from kz_labor_rag.eval.generator import DisabledGenerator

        config.data["retrieval"] = {"top_k": 5}
        monkeypatch.setattr(
            api,
            "_components",
            lambda: (
                config,
                FakeRetriever({QUESTION: ranked("54")}),
                DisabledGenerator("ANTHROPIC_API_KEY не задан"),
            ),
        )
        body = TestClient(api.app).get("/ask", params={"q": QUESTION}).json()
        assert body["answer"] == ""
        assert "ANTHROPIC_API_KEY" in body["error"]
        # Поиск при этом отработал: отсутствие ключа не ломает retrieval.
        assert body["hits"]


class TestContext:
    def test_returns_exactly_what_the_model_sees(self, client):
        body = client.get("/context", params={"q": QUESTION}).json()
        assert "ст. 54" in body["context"]
        assert "Текст статьи 54." in body["context"]


class TestHealth:
    def test_reports_index_state(self, monkeypatch):
        class Store:
            def count(self):
                return 4242

            def read_meta(self):
                return {
                    "corpus_edition_date": "07.08.2026",
                    "embeddings_model": "intfloat/multilingual-e5-base",
                    "built_at": "2026-08-11T00:00:00+00:00",
                }

        monkeypatch.setattr(api, "build_store", lambda config: Store())
        monkeypatch.setattr(api, "index_mismatch", lambda config, store: None)
        body = TestClient(api.app).get("/health").json()
        assert body["status"] == "ok"
        assert body["chunks_indexed"] == 4242
        assert body["index"]["corpus_edition_date"] == "07.08.2026"

    def test_mismatch_degrades_status(self, monkeypatch):
        class Store:
            def count(self):
                return 10

            def read_meta(self):
                return {"embeddings_model": "другая-модель"}

        monkeypatch.setattr(api, "build_store", lambda config: Store())
        monkeypatch.setattr(
            api, "index_mismatch", lambda config, store: "индекс построен другой моделью"
        )
        body = TestClient(api.app).get("/health").json()
        # Несоответствие индекса конфигу обязано быть видно сразу, а не после
        # первого странного ответа.
        assert body["status"] == "degraded"
        assert "другой моделью" in body["index"]["problem"]

    def test_unreachable_database_degrades_not_crashes(self, monkeypatch):
        def explode(config):
            raise StoreError("не удалось подключиться к базе")

        monkeypatch.setattr(api, "build_store", explode)
        body = TestClient(api.app).get("/health").json()
        assert body["status"] == "degraded"
        assert body["index"]["built"] is False
