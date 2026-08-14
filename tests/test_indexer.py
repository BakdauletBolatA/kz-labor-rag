"""Соответствие построенного индекса тому, что сейчас в конфиге и на диске.

Идентификатор чанка позиционный, а обычная сборка делает upsert без удаления.
Значит, любое изменение, от которого меняются тексты чанков, обязано приводить
к пересозданию с нуля: иначе от прошлого индекса остаётся хвост строк, которым
в корпусе уже ничего не соответствует, и поиск продолжает их возвращать.

Проверяется на фейковом хранилище: ``index_mismatch`` ходит только в
``read_meta`` и ``count``, базы для этого не нужно.
"""

from __future__ import annotations

import hashlib

import pytest

from kz_labor_rag.config import Config
from kz_labor_rag.corpus.chunker import chunking_signature
from kz_labor_rag.corpus.parser import PARSER_VERSION
from kz_labor_rag.indexer import corpus_sha256, index_mismatch
from kz_labor_rag.retrieval.factory import build_chunking_params

CORPUS_BYTES = b"<html><body>redaktsiya 07.08.2026</body></html>"
MODEL = "intfloat/multilingual-e5-base"


@pytest.fixture
def raw(tmp_path):
    path = tmp_path / "adilet.html"
    path.write_bytes(CORPUS_BYTES)
    return path


@pytest.fixture
def config(raw) -> Config:
    return Config(
        data={
            "version": "test-v0",
            "corpus": {"raw_html": str(raw)},
            "embeddings": {"model": MODEL},
            "chunking": {
                "strategy": "fixed_tokens",
                "chunk_size_tokens": 512,
                "chunk_overlap_tokens": 0,
            },
        }
    )


def healthy_meta(config) -> dict:
    return {
        "chunking_signature": chunking_signature(build_chunking_params(config)),
        "embeddings_model": MODEL,
        "corpus_sha256": hashlib.sha256(CORPUS_BYTES).hexdigest(),
        "parser_version": PARSER_VERSION,
        "chunks": 143,
    }


class FakeStore:
    """Хранилище, от которого index_mismatch нужны только метаданные и счёт."""

    def __init__(self, meta: dict | None, rows: int | None = None) -> None:
        self._meta = meta
        self._rows = rows if rows is not None else (meta or {}).get("chunks", 0)

    def read_meta(self) -> dict | None:
        return self._meta

    def count(self) -> int:
        return self._rows


class TestMatchingIndex:
    def test_no_problem_when_everything_agrees(self, config):
        assert index_mismatch(config, FakeStore(healthy_meta(config))) is None

    def test_missing_meta_is_reported(self, config):
        assert "не построен" in index_mismatch(config, FakeStore(None))


class TestCorpusEdition:
    """Главный незакрытый путь: корпус обновили, а индекс остался прежним."""

    def test_new_corpus_edition_forces_rebuild(self, config, raw):
        meta = healthy_meta(config)
        raw.write_bytes(b"<html><body>redaktsiya 01.01.2027</body></html>")
        problem = index_mismatch(config, FakeStore(meta))
        assert "другой редакции корпуса" in problem
        assert "--rebuild" in problem

    def test_hash_catches_edits_that_keep_the_edition_label(self, config, raw):
        # Метка редакции в подвале могла не поменяться, а текст статьи —
        # поменяться. Хеш файла строже разобранной edition_date.
        meta = healthy_meta(config)
        raw.write_bytes(CORPUS_BYTES + "<!-- правка одной статьи -->".encode())
        assert "другой редакции корпуса" in index_mismatch(config, FakeStore(meta))

    def test_missing_corpus_is_not_silently_ok(self, config, raw):
        meta = healthy_meta(config)
        raw.unlink()
        assert "не найден" in index_mismatch(config, FakeStore(meta))

    def test_corpus_sha256_is_none_without_file(self, config, raw):
        raw.unlink()
        assert corpus_sha256(config) is None


class TestParserVersion:
    def test_parser_bump_forces_rebuild(self, config):
        meta = {**healthy_meta(config), "parser_version": "0.9"}
        assert "другой версией парсера" in index_mismatch(config, FakeStore(meta))


class TestLeftoverRows:
    def test_extra_rows_are_caught(self, config):
        # Новая редакция дала 140 чанков, в таблице осталось 143: хвост от
        # прошлой сборки продолжает находиться поиском.
        meta = {**healthy_meta(config), "chunks": 140}
        problem = index_mismatch(config, FakeStore(meta, rows=143))
        assert "143 строк" in problem
        assert "140" in problem

    def test_matching_counts_pass(self, config):
        meta = {**healthy_meta(config), "chunks": 140}
        assert index_mismatch(config, FakeStore(meta, rows=140)) is None


class TestPreviouslyCoveredChecks:
    def test_chunking_change_still_caught(self, config):
        meta = {**healthy_meta(config), "chunking_signature": "0000000000000000"}
        assert "другой нарезкой" in index_mismatch(config, FakeStore(meta))

    def test_model_change_still_caught(self, config):
        meta = {**healthy_meta(config), "embeddings_model": "другая-модель"}
        assert "другой моделью" in index_mismatch(config, FakeStore(meta))
