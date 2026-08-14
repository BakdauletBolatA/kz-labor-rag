"""Контекст, который видит модель, и разбор ссылок из её ответа.

Оба механизма определяют, что покажут метрики генерации, но метриками не
являются и потому долго оставались без тестов. Оба дефекта ниже найдены
аудитом до первого прогона baseline.
"""

from __future__ import annotations

from kz_labor_rag.eval import metrics as M
from kz_labor_rag.eval.generator import extract_cited_articles
from kz_labor_rag.eval.judge import context_format_fingerprint, format_context
from kz_labor_rag.types import Chunk, RetrievedChunk


def chunk(cid: str, *articles: str, title: str = "", text: str = "текст нормы") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, text=text, articles=articles, article_title=title),
        score=0.9,
        rank=1,
    )


class TestContextLabelling:
    """Фрагмент подписывается всеми статьями, текст которых в него попал.

    Подпись одной головной статьёй делала неверную ссылку предписанным
    поведением: заголовки статей в baseline не приклеиваются, номеров в теле
    фрагмента нет, а промпт требует ссылаться только на статьи из фрагментов.
    """

    def test_single_article_chunk_keeps_title(self):
        out = format_context([chunk("c1", "54", title="Ограничение расторжения")])
        assert "[Статья 54. Ограничение расторжения]" in out

    def test_multi_article_chunk_lists_every_article(self):
        out = format_context([chunk("c1", "45", "46", "47", "48", title="Перемещение")])
        for number in ("45", "46", "47", "48"):
            assert number in out

    def test_multi_article_chunk_does_not_claim_a_single_article(self):
        out = format_context([chunk("c1", "45", "46", "47", "48", title="Перемещение")])
        assert "[Статья 45" not in out
        # Заголовок принадлежит одной статье из четырёх и так же указывал бы
        # на неё ложно.
        assert "Перемещение" not in out

    def test_boundary_crossing_is_stated_explicitly(self):
        out = format_context([chunk("c1", "45", "46")])
        assert "пересекает границы статей" in out

    def test_text_is_kept_verbatim(self):
        out = format_context([chunk("c1", "54", text="  Не допускается расторжение.  ")])
        assert "Не допускается расторжение." in out


class TestContextFingerprint:
    """Формат контекста — фактическая часть промпта, реестром не покрытая."""

    def test_is_stable_across_calls(self):
        assert context_format_fingerprint() == context_format_fingerprint()

    def test_looks_like_a_digest(self):
        fingerprint = context_format_fingerprint()
        assert len(fingerprint) == 12
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestCitationExtraction:
    def test_single_reference(self):
        assert extract_cited_articles("Нельзя (ст. 54).") == ["54"]

    def test_enumeration_after_one_keyword(self):
        # Самый частый способ сослаться на связку норм. Раньше отдавал одну
        # статью из двух, и выдуманная вторая ссылка была невидима для
        # citation_validity.
        assert extract_cited_articles("Согласно статьям 52 и 53 Кодекса") == ["52", "53"]

    def test_comma_separated_enumeration(self):
        assert extract_cited_articles("См. ст. 52, 53 и 54.") == ["52", "53", "54"]

    def test_clause_number_is_not_taken_for_an_article(self):
        assert extract_cited_articles("ст. 52 п. 1 и ст. 53") == ["52", "53"]

    def test_composite_article_number(self):
        assert extract_cited_articles("По ст. 73-1 график скользящий.") == ["73-1"]

    def test_duplicates_collapse_keeping_order(self):
        assert extract_cited_articles("(ст. 54) и ещё раз ст. 54, затем ст. 52") == ["54", "52"]

    def test_answer_without_references(self):
        assert extract_cited_articles("В предоставленных фрагментах ответа нет.") == []


class TestCitationValidityCatchesMislabelling:
    def test_reference_outside_context_is_invalid(self):
        chunks = [chunk("c1", "45", "46")]
        assert M.citation_validity(["99"], chunks) == 0.0

    def test_partial_credit_over_all_cited(self):
        chunks = [chunk("c1", "45", "46")]
        assert M.citation_validity(["45", "99"], chunks) == 0.5
