"""Нарезка на чанки.

Проверяется на потоке с известной структурой и токенизатором по пробелам:
здесь важна логика окон и восстановление принадлежности к статьям, а не
точное совпадение с токенизатором модели.
"""

from __future__ import annotations

import pytest

from kz_labor_rag.corpus.chunker import (
    ChunkingParams,
    WhitespaceTokenizer,
    build_chunks,
    build_stream,
    chunk_fixed_tokens,
    chunking_signature,
)
from kz_labor_rag.corpus.parser import Article, Clause, LaborCode
from kz_labor_rag.types import ClauseRef


def article(number: str, clauses: list[tuple[str, str]], title: str = "Заголовок") -> Article:
    return Article(
        number=number,
        title=title,
        chapter="Глава 4. ТРУДОВОЙ ДОГОВОР",
        clauses=tuple(Clause(number=n, text=t) for n, t in clauses),
        text="\n".join(t for _, t in clauses),
    )


def words(n: int, prefix: str) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


PARAMS = ChunkingParams(strategy="fixed_tokens", chunk_size_tokens=10, chunk_overlap_tokens=0)
TOK = WhitespaceTokenizer()


class TestParams:
    def test_overlap_must_be_smaller_than_window(self):
        # Иначе окно не сдвигается и нарезка зацикливается.
        with pytest.raises(ValueError, match="не сдвигается"):
            ChunkingParams("fixed_tokens", 512, 512).validate()

    def test_zero_window_rejected(self):
        with pytest.raises(ValueError, match="chunk_size_tokens"):
            ChunkingParams("fixed_tokens", 0, 0).validate()

    def test_signature_changes_with_any_parameter(self):
        base = ChunkingParams("fixed_tokens", 512, 0)
        assert chunking_signature(base) != chunking_signature(
            ChunkingParams("fixed_tokens", 256, 0)
        )
        assert chunking_signature(base) != chunking_signature(
            ChunkingParams("fixed_tokens", 512, 64)
        )
        assert chunking_signature(base) != chunking_signature(ChunkingParams("article", 512, 0))
        assert chunking_signature(base) == chunking_signature(
            ChunkingParams("fixed_tokens", 512, 0)
        )


class TestStream:
    def test_headers_excluded_by_default(self):
        stream = build_stream(
            [article("54", [("1", "текст пункта")])], prepend_article_header=False
        )
        # Baseline не получает заголовок статьи бесплатно: это отдельная гипотеза.
        assert "Статья 54" not in stream.text
        assert "текст пункта" in stream.text

    def test_headers_included_when_configured(self):
        stream = build_stream([article("54", [("1", "текст пункта")])], prepend_article_header=True)
        assert "Статья 54. Заголовок" in stream.text

    def test_segments_map_back_to_clauses(self):
        stream = build_stream(
            [article("54", [("1", "первый пункт"), ("2", "второй пункт")])],
            prepend_article_header=False,
        )
        assert stream.spans_in(0, len(stream.text)) == (("54", "1"), ("54", "2"))

    def test_span_lookup_respects_boundaries(self):
        stream = build_stream(
            [article("54", [("1", "аааа"), ("2", "бббб")])], prepend_article_header=False
        )
        first_end = stream.text.index("бббб")
        assert stream.spans_in(0, first_end) == (("54", "1"),)


class TestFixedTokenChunking:
    def test_window_size_respected(self):
        stream = build_stream(
            [article("54", [("1", words(45, "w"))])], prepend_article_header=False
        )
        chunks = chunk_fixed_tokens(stream, TOK, PARAMS)
        assert len(chunks) == 5
        assert all(len(c.text.split()) <= 10 for c in chunks)

    def test_chunks_cover_the_whole_stream(self):
        stream = build_stream(
            [article("54", [("1", words(37, "w"))])], prepend_article_header=False
        )
        chunks = chunk_fixed_tokens(stream, TOK, PARAMS)
        joined = " ".join(c.text for c in chunks).split()
        assert joined == [f"w{i}" for i in range(37)]

    def test_overlap_repeats_tail_tokens(self):
        stream = build_stream(
            [article("54", [("1", words(20, "w"))])], prepend_article_header=False
        )
        params = ChunkingParams("fixed_tokens", 10, 3)
        chunks = chunk_fixed_tokens(stream, TOK, params)
        first, second = chunks[0].text.split(), chunks[1].text.split()
        assert first[-3:] == second[:3]

    def test_chunk_crossing_article_boundary_counts_for_both(self):
        # Ровно то, ради чего у чанка список статей, а не одна:
        # окно на 10 токенов накрывает конец 52-й и начало 53-й.
        stream = build_stream(
            [article("52", [("1", words(6, "a"))]), article("53", [("1", words(6, "b"))])],
            prepend_article_header=False,
        )
        chunks = chunk_fixed_tokens(stream, TOK, PARAMS)
        crossing = [c for c in chunks if len(c.articles) > 1]
        assert crossing, "ни один чанк не пересёк границу статей"
        assert crossing[0].articles == ("52", "53")
        # Головная статья — та, с которой чанк начинается.
        assert crossing[0].article == "52"

    def test_spans_carry_clause_level_coverage(self):
        stream = build_stream(
            [article("54", [("1", words(5, "a")), ("2", words(5, "b"))])],
            prepend_article_header=False,
        )
        chunk = chunk_fixed_tokens(stream, TOK, PARAMS)[0]
        assert chunk.covers(ClauseRef("54", "1"))
        assert chunk.covers(ClauseRef("54", "2"))
        assert not chunk.covers(ClauseRef("54", "3"))
        assert not chunk.covers(ClauseRef("52", "1"))

    def test_chunk_ids_are_unique_and_stable(self):
        stream = build_stream(
            [article("54", [("1", words(50, "w"))])], prepend_article_header=False
        )
        first = chunk_fixed_tokens(stream, TOK, PARAMS)
        second = chunk_fixed_tokens(stream, TOK, PARAMS)
        ids = [c.chunk_id for c in first]
        assert len(set(ids)) == len(ids)
        assert ids == [c.chunk_id for c in second]

    def test_empty_stream_yields_nothing(self):
        stream = build_stream([], prepend_article_header=False)
        assert chunk_fixed_tokens(stream, TOK, PARAMS) == []


class TestArticleChunking:
    def test_one_chunk_per_article(self):
        code = LaborCode(
            articles=(
                article("52", [("1", "текст пятьдесят два")]),
                article("53", [("1", "текст пятьдесят три")]),
            )
        )
        chunks = build_chunks(code, TOK, ChunkingParams("article", 512, 0))
        assert [c.article for c in chunks] == ["52", "53"]
        assert all(len(c.articles) == 1 for c in chunks)

    def test_article_strategy_keeps_metadata(self):
        code = LaborCode(articles=(article("54", [("1", "текст")], title="Ограничение"),))
        chunk = build_chunks(code, TOK, ChunkingParams("article", 512, 0))[0]
        assert chunk.article_title == "Ограничение"
        assert chunk.chapter == "Глава 4. ТРУДОВОЙ ДОГОВОР"


class TestBuildChunks:
    def test_repealed_articles_are_not_indexed(self):
        repealed = Article(
            number="117",
            title="Исключённая",
            amendment_notes=("Сноска. Статья 117 исключена Законом РК.",),
        )
        code = LaborCode(articles=(article("54", [("1", words(5, "w"))]), repealed))
        chunks = build_chunks(code, TOK, PARAMS)
        assert all("117" not in c.articles for c in chunks)

    def test_unknown_strategy_is_fatal(self):
        code = LaborCode(articles=(article("54", [("1", "текст")]),))
        with pytest.raises(ValueError, match="неизвестная стратегия"):
            build_chunks(code, TOK, ChunkingParams("magic", 512, 0))
