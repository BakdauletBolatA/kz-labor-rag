"""Хранилище pgvector.

Тесты идут против настоящей базы и пропускаются, если её нет: pgvector нечем
осмысленно замокать — проверять надо ровно то, как ведёт себя векторный поиск,
а не наш пересказ его поведения.

    docker compose up -d db
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from kz_labor_rag.embeddings.encoder import HashEncoder
from kz_labor_rag.retrieval.dense import DenseRetriever
from kz_labor_rag.retrieval.store import (
    PgVectorStore,
    StoreError,
    StoreParams,
    to_similarity,
)
from kz_labor_rag.types import Chunk, ClauseRef

DSN = os.environ.get("KZRAG_TEST_DSN", "postgresql://kzrag:kzrag@localhost:5432/kzrag")
DIM = 16


def db_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not db_available(), reason="нет базы: docker compose up -d db")


TEST_TABLE = "chunks_test"


@pytest.fixture
def store():
    # Прямая страховка: фикстура сносит свои таблицы, поэтому промах в имени
    # уничтожил бы боевой индекс.
    from conftest import PRODUCTION_TABLE

    assert TEST_TABLE != PRODUCTION_TABLE, "тест не имеет права работать с боевой таблицей"

    s = PgVectorStore(StoreParams(dsn=DSN, table=TEST_TABLE, distance="cosine", dimensions=DIM))
    s.connect()
    s.drop()
    s.create_schema()
    yield s
    s.drop()
    s.close()


def chunk(cid: str, articles: tuple[str, ...], spans: tuple[tuple[str, str], ...]) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"текст {cid} про статьи {', '.join(articles)}",
        articles=articles,
        spans=spans,
        article_title="Заголовок",
        chapter="Глава 4. ТРУДОВОЙ ДОГОВОР",
    )


def vectors(*rows: list[float]) -> np.ndarray:
    out = np.zeros((len(rows), DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
        out[i] /= np.linalg.norm(out[i])
    return out


class TestSchemaAndWrite:
    def test_upsert_and_count(self, store):
        chunks = [chunk("c1", ("54",), (("54", "1"),)), chunk("c2", ("52",), (("52", "1"),))]
        assert store.upsert(chunks, vectors([1, 0], [0, 1])) == 2
        assert store.count() == 2

    def test_upsert_is_idempotent(self, store):
        c = [chunk("c1", ("54",), (("54", "1"),))]
        store.upsert(c, vectors([1, 0]))
        store.upsert(c, vectors([1, 0]))
        assert store.count() == 1

    def test_dimension_mismatch_is_fatal(self, store):
        wrong = np.zeros((1, DIM + 4), dtype=np.float32)
        with pytest.raises(StoreError, match="размерность"):
            store.upsert([chunk("c1", ("54",), ())], wrong)

    def test_desync_between_chunks_and_vectors_is_fatal(self, store):
        with pytest.raises(StoreError, match="рассинхрон"):
            store.upsert([chunk("c1", ("54",), ())], vectors([1, 0], [0, 1]))


class TestSearch:
    def test_orders_by_similarity(self, store):
        store.upsert(
            [chunk("c1", ("54",), ()), chunk("c2", ("52",), ()), chunk("c3", ("30",), ())],
            vectors([1, 0], [0.9, 0.1], [0, 1]),
        )
        hits = store.search(vectors([1, 0])[0], 3)
        assert [h.chunk_id for h in hits] == ["c1", "c2", "c3"]
        assert hits[0].rank == 1

    def test_score_grows_with_relevance(self, store):
        store.upsert([chunk("c1", ("54",), ()), chunk("c2", ("52",), ())], vectors([1, 0], [0, 1]))
        hits = store.search(vectors([1, 0])[0], 2)
        # Наружу уходит скор, а не расстояние: больше значит релевантнее.
        assert hits[0].score > hits[1].score
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_limit_respected(self, store):
        store.upsert(
            [chunk(f"c{i}", ("54",), ()) for i in range(5)],
            np.vstack([vectors([1, i / 10]) for i in range(5)]),
        )
        assert len(store.search(vectors([1, 0])[0], 3)) == 3

    def test_multi_article_chunk_survives_roundtrip(self, store):
        # Чанк, перешедший границу статей, обязан вернуться из базы целым:
        # на этом держится recall при наивной нарезке.
        store.upsert(
            [chunk("c1", ("52", "53", "54"), (("52", "1"), ("53", "2"), ("54", "1")))],
            vectors([1, 0]),
        )
        hit = store.search(vectors([1, 0])[0], 1)[0]
        assert hit.articles == ("52", "53", "54")
        assert hit.article == "52"
        assert hit.covers(ClauseRef("53", "2"))
        assert not hit.covers(ClauseRef("53", "1"))

    def test_metadata_survives_roundtrip(self, store):
        store.upsert([chunk("c1", ("54",), (("54", "1"),))], vectors([1, 0]))
        hit = store.search(vectors([1, 0])[0], 1)[0]
        assert hit.chunk.article_title == "Заголовок"
        assert hit.chunk.chapter == "Глава 4. ТРУДОВОЙ ДОГОВОР"


class TestByArticle:
    def test_finds_chunks_containing_the_article(self, store):
        store.upsert(
            [
                chunk("c1", ("52", "53"), ()),
                chunk("c2", ("53", "54"), ()),
                chunk("c3", ("99",), ()),
            ],
            vectors([1, 0], [0, 1], [1, 1]),
        )
        assert [c.chunk_id for c in store.by_article("53")] == ["c1", "c2"]
        assert store.by_article("999") == []


class TestMeta:
    def test_roundtrip(self, store):
        store.write_meta({"version": "baseline-v0", "chunking_signature": "abc"})
        meta = store.read_meta()
        assert meta["version"] == "baseline-v0"
        assert "built_at" in meta

    def test_absent_meta_is_none(self, store):
        assert store.read_meta() is None


class TestRetriever:
    def test_empty_index_explains_itself(self, store):
        retriever = DenseRetriever(store, HashEncoder(DIM), version="test")
        with pytest.raises(StoreError, match="индекс пуст"):
            retriever.search("вопрос", 5)

    def test_returns_k_hits(self, store):
        encoder = HashEncoder(DIM)
        chunks = [chunk(f"c{i}", (str(50 + i),), ()) for i in range(8)]
        store.upsert(chunks, encoder.encode_passages([c.text for c in chunks]))

        retriever = DenseRetriever(store, encoder, version="test")
        hits = retriever.search("можно ли уволить в отпуске", 5)
        assert len(hits) == 5
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]

    def test_scores_are_monotonic(self, store):
        encoder = HashEncoder(DIM)
        chunks = [chunk(f"c{i}", (str(50 + i),), ()) for i in range(8)]
        store.upsert(chunks, encoder.encode_passages([c.text for c in chunks]))
        scores = [h.score for h in DenseRetriever(store, encoder, version="t").search("x", 8)]
        assert scores == sorted(scores, reverse=True)


class TestSimilarityConversion:
    def test_cosine(self):
        assert to_similarity(0.0, "cosine") == 1.0
        assert to_similarity(1.0, "cosine") == 0.0

    def test_l2_is_negated_so_bigger_is_better(self):
        assert to_similarity(2.0, "l2") < to_similarity(1.0, "l2")

    def test_unknown_metric_rejected(self):
        params = StoreParams(dsn="x", table="t", distance="manhattan", dimensions=8)
        with pytest.raises(StoreError, match="неизвестная метрика"):
            _ = params.operator


class TestSchemaGuards:
    """Схема обязана ловить рассогласование сама.

    Иначе таблица, оставшаяся от прошлой модели эмбеддингов, молча
    переживает CREATE TABLE IF NOT EXISTS, и расхождение вылезает глубоко
    внутри вставки как невнятный DataException.
    """

    def test_existing_table_with_other_dimensions_is_rejected(self, store):
        other = PgVectorStore(
            StoreParams(
                dsn=DSN, table="chunks_test", distance="cosine", dimensions=DIM * 2
            )
        )
        try:
            with pytest.raises(StoreError, match="уже существует с размерностью"):
                other.create_schema()
        finally:
            other.close()

    def test_message_tells_how_to_fix(self, store):
        other = PgVectorStore(
            StoreParams(dsn=DSN, table=TEST_TABLE, distance="cosine", dimensions=999)
        )
        try:
            with pytest.raises(StoreError, match="--rebuild"):
                other.create_schema()
        finally:
            other.close()

    def test_matching_dimensions_pass(self, store):
        store.create_schema()  # повторный вызов на своей же схеме безвреден
        assert store.existing_dimensions() == DIM

    def test_absent_table_reports_none(self, store):
        store.drop()
        assert store.existing_dimensions() is None


class TestMetaIsolation:
    """Метаданные привязаны к своей таблице чанков.

    Раньше таблица называлась ``index_meta`` и была одна на всю базу: фикстура
    тестов с таблицей ``chunks_test`` сносила метаданные боевого индекса.
    Чанки при этом оставались, и проверка соответствия молчала не потому, что
    всё сошлось, а потому, что сравнивать стало не с чем.
    """

    def test_meta_table_name_follows_the_chunks_table(self):
        assert (
            StoreParams(dsn="x", table="chunks", distance="cosine", dimensions=8).meta_table
            == "chunks_meta"
        )

    def test_dropping_one_store_keeps_the_others_meta(self, store):
        other = PgVectorStore(
            StoreParams(dsn=DSN, table="chunks_other", distance="cosine", dimensions=DIM)
        )
        try:
            other.create_schema()
            other.write_meta({"version": "боевой"})
            store.write_meta({"version": "тестовый"})

            store.drop()  # роняем «тестовое» хранилище

            assert other.read_meta()["version"] == "боевой"
        finally:
            other.drop()
            other.close()

    def test_missing_meta_table_reads_as_none(self, store):
        store.drop()
        assert store.read_meta() is None
