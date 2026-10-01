"""Нарезка на чанки.

Проверяется на потоке с известной структурой и токенизатором по пробелам:
здесь важна логика окон и восстановление принадлежности к статьям, а не
точное совпадение с токенизатором модели.
"""

from __future__ import annotations

import pytest

from kz_labor_rag.corpus.chunker import (
    ChunkingParams,
    EncodingBudget,
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

    def test_header_alone_covers_no_clause(self):
        # У ст. 83 единственный пункт без номера, то есть с номером "". Заголовок
        # тоже помечался пунктом "", и окно с одним заголовком «покрывало» пункт.
        stream = build_stream([article("83", [("", "текст пункта")])], prepend_article_header=True)
        header_end = stream.text.index("текст пункта")
        assert stream.spans_in(0, header_end) == ()

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


def structured(strategy, articles, size=10, header=False, budget=None):
    params = ChunkingParams(strategy, size, 0, prepend_article_header=header)
    return build_chunks(LaborCode(articles=tuple(articles)), TOK, params, budget)


class TestClauseChunking:
    def test_one_chunk_per_clause(self):
        chunks = structured("clause", [article("54", [("1", "аа бб"), ("2", "вв гг")])])
        assert [c.spans for c in chunks] == [(("54", "1"),), (("54", "2"),)]
        assert [c.text for c in chunks] == ["аа бб", "вв гг"]

    def test_long_clause_is_split_inside_its_borders(self):
        chunks = structured(
            "clause", [article("52", [("1", words(25, "w")), ("2", words(3, "x"))])]
        )
        first = [c for c in chunks if c.spans == (("52", "1"),)]
        assert len(first) == 3
        assert all("x0" not in c.text for c in first)
        assert chunks[-1].spans == (("52", "2"),)

    def test_chunk_ids_are_unique(self):
        chunks = structured("clause", [article("52", [("1", words(25, "w")), ("2", "x")])])
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_header_goes_into_every_window(self):
        chunks = structured("clause", [article("52", [("1", words(25, "w"))])], header=True)
        assert len(chunks) > 1
        assert all(c.text.startswith("Статья 52. Заголовок\n") for c in chunks)

    def test_header_is_counted_against_the_budget(self):
        budget = EncodingBudget(measure=lambda t: len(t.split()), limit=10)
        chunks = structured(
            "clause", [article("52", [("1", words(25, "w"))])], header=True, budget=budget
        )
        assert max(len(c.text.split()) for c in chunks) <= 10


class TestArticleChunking:
    def test_one_chunk_per_article(self):
        chunks = structured(
            "article",
            [article("52", [("1", "текст"), ("2", "ещё")]), article("53", [("1", "другой")])],
        )
        assert [c.articles for c in chunks] == [("52",), ("53",)]
        assert chunks[0].spans == (("52", "1"), ("52", "2"))

    def test_long_article_never_crosses_into_the_next(self):
        chunks = structured(
            "article", [article("52", [("1", words(25, "w"))]), article("53", [("1", "x")])]
        )
        assert all(c.articles in (("52",), ("53",)) for c in chunks)
        assert len([c for c in chunks if c.article == "52"]) == 3

    def test_keeps_metadata(self):
        chunk = structured("article", [article("54", [("1", "текст")], title="Ограничение")])[0]
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


class TestEncodingBudget:
    """Окно ужимается под предел модели, стратегия нарезки не меняется.

    Работает на токенизаторе по пробелам, поэтому проверка живёт и в CI,
    где корпуса и весов модели нет.
    """

    def _stream(self, n: int = 60):
        return build_stream([article("54", [("1", words(n, "w"))])], prepend_article_header=False)

    def _budget(self, limit: int, prefix_tokens: int = 3):
        # Имитируем реальное поведение: содержимое плюс префикс плюс два
        # служебных токена.
        return EncodingBudget(
            measure=lambda text: len(text.split()) + prefix_tokens + 2, limit=limit
        )

    def test_without_budget_window_is_exactly_chunk_size(self):
        chunks = chunk_fixed_tokens(self._stream(), TOK, PARAMS)
        assert len(chunks[0].text.split()) == 10

    def test_budget_shrinks_the_window(self):
        # Предел 10, оверхед 5 -> содержимого может быть максимум 5.
        chunks = chunk_fixed_tokens(self._stream(), TOK, PARAMS, self._budget(limit=10))
        assert all(len(c.text.split()) <= 5 for c in chunks)

    def test_no_chunk_exceeds_the_limit(self):
        budget = self._budget(limit=10)
        chunks = chunk_fixed_tokens(self._stream(), TOK, PARAMS, budget)
        assert all(budget.measure(c.text) <= budget.limit for c in chunks)

    def test_roomy_budget_leaves_window_untouched(self):
        # Если предел не жмёт, нарезка обязана остаться прежней.
        wide = chunk_fixed_tokens(self._stream(), TOK, PARAMS, self._budget(limit=1000))
        plain = chunk_fixed_tokens(self._stream(), TOK, PARAMS)
        assert [c.text for c in wide] == [c.text for c in plain]

    def test_strategy_is_unchanged_only_window_size(self):
        # Чанки по-прежнему режутся подряд по потоку и покрывают его целиком,
        # без оглядки на структуру: меняется только длина окна.
        budget = self._budget(limit=10)
        chunks = chunk_fixed_tokens(self._stream(37), TOK, PARAMS, budget)
        joined = " ".join(c.text for c in chunks).split()
        assert joined == [f"w{i}" for i in range(37)]

    def test_shrinking_produces_more_chunks(self):
        plain = chunk_fixed_tokens(self._stream(), TOK, PARAMS)
        shrunk = chunk_fixed_tokens(self._stream(), TOK, PARAMS, self._budget(limit=10))
        assert len(shrunk) > len(plain)

    def test_impossible_budget_is_fatal_not_silent(self):
        # Оверхед больше предела — подогнать нечего. Молча отдавать
        # переполненные чанки нельзя.
        with pytest.raises(ValueError, match="не удалось подобрать размер окна"):
            chunk_fixed_tokens(self._stream(), TOK, PARAMS, self._budget(limit=4))

    def test_build_chunks_passes_budget_through(self):
        code = LaborCode(articles=(article("54", [("1", words(60, "w"))]),))
        budget = self._budget(limit=10)
        chunks = build_chunks(code, TOK, PARAMS, budget)
        assert chunks and all(budget.measure(c.text) <= budget.limit for c in chunks)

    def test_whitespace_tokenizer_reports_encoded_length(self):
        assert TOK.encoded_length("одно два три") == 5  # три слова плюс два служебных


class TestChunkerVersionInSignature:
    def test_version_bump_invalidates_old_chunks(self):
        """Подпись чанкинга включает версию чанкера.

        Иначе исправление нарезки не обесценило бы ни старый индекс, ни кэш
        эмбеддингов, и поиск продолжил бы работать на векторах от прежних,
        усечённых чанков.
        """
        import kz_labor_rag.corpus.chunker as ch

        params = ChunkingParams("fixed_tokens", 512, 0)
        before = chunking_signature(params)
        original = ch.CHUNKER_VERSION
        try:
            ch.CHUNKER_VERSION = "9.9"
            assert chunking_signature(params) != before
        finally:
            ch.CHUNKER_VERSION = original
