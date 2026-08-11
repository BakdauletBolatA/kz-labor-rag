"""Парсер ТК РК.

Основные проверки идут на маленьких синтетических фрагментах разметки — так
видно, какое именно свойство HTML проверяется. В конце есть интеграционные
проверки на настоящем корпусе; они пропускаются, если файла нет, потому что
сырой HTML не коммитится.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kz_labor_rag.corpus.parser import ParseError, parse_labor_code
from kz_labor_rag.types import article_sort_key

RAW = Path("data/raw/adilet_K1500000414_rus.html")


def page(body: str) -> str:
    """Обернуть фрагмент в контейнер документа плюс обвязку сайта."""
    return f"""
    <html><body><div class="main">
    {body}
    <div class="container_omega aftertext">Если Вы обнаружили ошибку...</div>
    <div class="png_bg"><h4>Последние документы</h4>О районном бюджете на 2025-2027 годы</div>
    </div></body></html>
    """


ARTICLE_54 = """
<h3><a name="z206"></a>РАЗДЕЛ 2. ТРУДОВЫЕ ОТНОШЕНИЯ<br><a name="z207"></a>Глава 4. ТРУДОВОЙ ДОГОВОР</h3>
<p><b><a name="z54"></a>Статья 54. Ограничение возможности расторжения</b></p>
<p id="z439">&nbsp;&nbsp; 1. Не допускается расторжение трудового договора в период отпуска.</p>
<p id="z440">&nbsp;&nbsp; 2. Расторжение допускается в случаях, предусмотренных статьей 52.</p>
"""


class TestArticleDetection:
    def test_basic_article(self):
        code = parse_labor_code(page(ARTICLE_54))
        assert len(code) == 1
        a = code.by_number["54"]
        assert a.number == "54"
        assert a.title == "Ограничение возможности расторжения"
        assert a.section == "РАЗДЕЛ 2. ТРУДОВЫЕ ОТНОШЕНИЯ"
        assert a.chapter == "Глава 4. ТРУДОВОЙ ДОГОВОР"

    def test_heading_without_anchor_is_still_found(self):
        # Случай статьи 140: якорь <a name> перехвачен блоком «Примечание ИЗПИ!»
        # перед заголовком, поэтому опираться на якорь нельзя.
        html = page(
            '<font color="#FF0000"><a name="z140"></a>Примечание ИЗПИ!</font>'
            "<p><b>Статья 140. Особенности регулирования труда руководителя</b></p>"
            "<p>&nbsp; 1. Текст пункта.</p>"
        )
        code = parse_labor_code(html)
        assert code.by_number["140"].title.startswith("Особенности регулирования")

    def test_cross_reference_is_not_mistaken_for_heading(self):
        # «предусмотренных статьей 52» в теле не должно порождать статью 52.
        code = parse_labor_code(page(ARTICLE_54))
        assert set(code.by_number) == {"54"}
        assert "статьей 52" in code.by_number["54"].text

    def test_compound_article_numbers(self):
        html = page(
            "<p><b>Статья 73. Режим рабочего времени</b></p><p>&nbsp; 1. Текст.</p>"
            "<p><b>Статья 73-1. Скользящий график работы</b></p><p>&nbsp; 1. Другой текст.</p>"
            "<p><b>Статья 74. Сменная работа</b></p><p>&nbsp; 1. Третий текст.</p>"
        )
        code = parse_labor_code(html)
        assert list(code.by_number) == ["73", "73-1", "74"]
        # Составной номер — отдельная статья, а не пункт 73-й.
        assert code.by_number["73-1"].title == "Скользящий график работы"
        assert code.by_number["73-1"].text == "Другой текст."

    def test_natural_sort_of_article_numbers(self):
        assert sorted(["73-1", "8", "73", "203-1", "20"], key=article_sort_key) == [
            "8",
            "20",
            "73",
            "73-1",
            "203-1",
        ]


class TestNotesAreStripped:
    def test_paragraph_note_goes_to_metadata(self):
        html = page(
            "<p><b>Статья 54. Заголовок</b></p>"
            "<p>&nbsp; 1. Нормативный текст.</p>"
            '<p class="note">Сноска. Статья 54 с изменениями, внесенными Законом РК от 04.05.2020.</p>'
        )
        code = parse_labor_code(html)
        a = code.by_number["54"]
        assert a.text == "Нормативный текст."
        assert "Сноска" not in a.text
        assert len(a.amendment_notes) == 1

    def test_bare_span_note_between_paragraphs_is_collected(self):
        # Сноски приходят и голым <span class="note"> вне <p>.
        html = page(
            "<p><b>Статья 54. Заголовок</b></p>"
            "<p>&nbsp; 1. Нормативный текст.</p>"
            '<span class="note">Сноска. Статья 54 в редакции Закона РК.</span>'
        )
        code = parse_labor_code(html)
        a = code.by_number["54"]
        assert "Сноска" not in a.text
        assert a.amendment_notes == ("Сноска. Статья 54 в редакции Закона РК.",)

    def test_inline_note_removal_does_not_glue_words(self):
        # Вырезание узла склеивало соседний текст: «пунктом 1-1<note/>статьи»
        # превращалось в «1-1статьи», и дословная цитата переставала находиться.
        html = page(
            "<p><b>Статья 54. Заголовок</b></p>"
            '<p>&nbsp; 1. Согласно пункту 1-1<span class="note">Сноска.</span>'
            "статьи 52 настоящего Кодекса.</p>"
        )
        code = parse_labor_code(html)
        assert "1-1 статьи 52" in code.by_number["54"].text

    def test_notes_can_be_kept_when_configured(self):
        html = page(
            "<p><b>Статья 54. Заголовок</b></p>"
            '<p>&nbsp; 1. Текст.<span class="note"> Сноска. Изменена.</span></p>'
        )
        kept = parse_labor_code(html, strip_amendment_notes=False)
        assert "Сноска" in kept.by_number["54"].text

    def test_izpi_future_edition_note_is_separated(self):
        html = page(
            "<p><b>Статья 132. Заголовок</b></p>"
            '<font color="#FF0000">Примечание ИЗПИ! В статью 132 предусматривается '
            "изменение Законом РК от 12.06.2026 (вводится в действие с 01.01.2027).</font>"
            "<p>&nbsp; 1. Действующий текст.</p>"
        )
        code = parse_labor_code(html)
        a = code.by_number["132"]
        assert a.text == "Действующий текст."
        assert a.has_future_edition
        assert "01.01.2027" in a.izpi_notes[0]


class TestClauseSplitting:
    def test_numbered_clauses(self):
        code = parse_labor_code(page(ARTICLE_54))
        a = code.by_number["54"]
        assert a.clause_numbers == ("1", "2")
        assert a.clauses[0].text.startswith("Не допускается")

    def test_subitems_stay_inside_their_clause(self):
        # Подпункты «1)» не приравниваются к пунктам «1.»: нумерация
        # пересекается, и смешивание сломало бы preferred_clause в датасете.
        html = page(
            "<p><b>Статья 52. Заголовок</b></p>"
            "<p>&nbsp; 1. Договор расторгается по основаниям:</p>"
            "<p>&nbsp; 1) ликвидация работодателя;</p>"
            "<p>&nbsp; 2) сокращение численности;</p>"
            "<p>&nbsp; 2. Не допускается расторжение в отпуске.</p>"
        )
        a = parse_labor_code(html).by_number["52"]
        assert a.clause_numbers == ("1", "2")
        assert "ликвидация работодателя" in a.clauses[0].text
        assert "сокращение численности" in a.clauses[0].text
        assert a.clauses[1].text == "Не допускается расторжение в отпуске."

    def test_compound_clause_number(self):
        html = page(
            "<p><b>Статья 52. Заголовок</b></p>"
            "<p>&nbsp; 1. Первый пункт.</p>"
            "<p>&nbsp; 1-1. Дополнительный пункт.</p>"
        )
        assert parse_labor_code(html).by_number["52"].clause_numbers == ("1", "1-1")

    def test_unnumbered_text_is_not_lost(self):
        html = page(
            "<p><b>Статья 60. Заголовок</b></p>"
            "<p>&nbsp; Текст без нумерации пунктов.</p>"
        )
        a = parse_labor_code(html).by_number["60"]
        assert a.clause_numbers == ("",)
        assert a.text == "Текст без нумерации пунктов."


class TestRepealedArticles:
    def test_repealed_article_detected(self):
        html = page(
            "<p><b>Статья 117. Профессиональные стандарты</b></p>"
            '<p class="note">Сноска. Статья 117 исключена Законом РК от 04.07.2023.</p>'
        )
        a = parse_labor_code(html).by_number["117"]
        assert a.is_repealed
        assert a.text == ""

    def test_normal_article_is_not_repealed(self):
        assert not parse_labor_code(page(ARTICLE_54)).by_number["54"].is_repealed

    def test_in_force_excludes_repealed(self):
        html = page(
            ARTICLE_54
            + "<p><b>Статья 117. Заголовок</b></p>"
            + '<p class="note">Сноска. Статья 117 исключена Законом РК.</p>'
        )
        code = parse_labor_code(html)
        assert len(code) == 2
        assert [a.number for a in code.in_force] == ["54"]


class TestStructure:
    def test_chapter_marked_up_as_paragraph(self):
        # Глава 18 размечена <p>, а не <h3>, в отличие от остальных глав.
        html = page(
            "<h3>РАЗДЕЛ 4. БЕЗОПАСНОСТЬ ТРУДА<br>Глава 17. ОХРАНА ТРУДА</h3>"
            "<p><b>Статья 180. Первая</b></p><p>&nbsp; 1. Текст.</p>"
            "<p><b>Глава 18. РАССЛЕДОВАНИЕ НЕСЧАСТНЫХ СЛУЧАЕВ</b></p>"
            "<p><b>Статья 186. Вторая</b></p><p>&nbsp; 1. Текст.</p>"
        )
        code = parse_labor_code(html)
        assert code.by_number["180"].chapter == "Глава 17. ОХРАНА ТРУДА"
        assert code.by_number["186"].chapter == "Глава 18. РАССЛЕДОВАНИЕ НЕСЧАСТНЫХ СЛУЧАЕВ"

    def test_br_inside_heading_keeps_word_boundary(self):
        html = page(
            "<h3>Глава 2. ГОСУДАРСТВЕННОЕ РЕГУЛИРОВАНИЕ ТРУДОВЫХ<br>ОТНОШЕНИЙ</h3>"
            "<p><b>Статья 15. Заголовок</b></p><p>&nbsp; 1. Текст.</p>"
        )
        chapter = parse_labor_code(html).by_number["15"].chapter
        assert "ТРУДОВЫХОТНОШЕНИЙ" not in chapter
        assert "ТРУДОВЫХ ОТНОШЕНИЙ" in chapter

    def test_part_resets_section_and_chapter(self):
        html = page(
            "<h3>ОБЩАЯ ЧАСТЬ<br>РАЗДЕЛ 1. ОБЩИЕ ПОЛОЖЕНИЯ<br>Глава 1. ОСНОВНЫЕ ПОЛОЖЕНИЯ</h3>"
            "<p><b>Статья 1. Понятия</b></p><p>&nbsp; 1. Текст.</p>"
            "<h3>ОСОБЕННАЯ ЧАСТЬ</h3>"
            "<p><b>Статья 21. Другая</b></p><p>&nbsp; 1. Текст.</p>"
        )
        code = parse_labor_code(html)
        assert code.by_number["1"].chapter == "Глава 1. ОСНОВНЫЕ ПОЛОЖЕНИЯ"
        assert code.by_number["21"].part == "ОСОБЕННАЯ ЧАСТЬ"
        assert code.by_number["21"].chapter == ""


class TestChromeAndFailures:
    def test_site_chrome_is_cut_off(self):
        code = parse_labor_code(page(ARTICLE_54))
        joined = " ".join(a.text for a in code)
        assert "Последние документы" not in joined
        assert "районном бюджете" not in joined

    def test_missing_container_is_fatal(self):
        with pytest.raises(ParseError, match="контейнер"):
            parse_labor_code("<html><body><p>что угодно</p></body></html>")

    def test_document_without_articles_is_fatal(self):
        with pytest.raises(ParseError, match="ни одной статьи"):
            parse_labor_code(page("<p>Просто текст без статей.</p>"))


@pytest.mark.skipif(not RAW.exists(), reason="сырой HTML не скачан (data/raw в .gitignore)")
class TestRealCorpus:
    """Проверки на настоящем документе. Числа взяты из редакции 07.08.2026."""

    @pytest.fixture(scope="class")
    def code(self):
        from kz_labor_rag.corpus.parser import parse_file

        return parse_file(RAW)

    def test_article_count(self, code):
        # 204 простых номера (1–204 без пропусков) + 19 составных.
        assert len(code) == 223
        simple = [a.number for a in code if "-" not in a.number]
        assert len(simple) == 204
        assert len([a for a in code if "-" in a.number]) == 19

    def test_no_gaps_in_simple_numbering(self, code):
        simple = sorted(int(a.number) for a in code if "-" not in a.number)
        assert simple == list(range(1, 205))

    def test_edition_date_is_captured(self, code):
        assert code.edition_date == "07.08.2026"

    def test_every_article_has_text_or_is_repealed(self, code):
        assert code.stats["empty_not_repealed"] == 0
        assert {a.number for a in code if a.is_repealed} == {"117", "197", "199"}

    def test_amendment_notes_never_leak_into_text(self, code):
        # Самая опасная поломка: история правок в теле статьи попадает в чанк
        # и в дословную цитату эталона.
        assert [a.number for a in code if "Сноска" in a.text] == []
        assert [a.number for a in code if "Законом РК от" in a.text] == []

    def test_site_chrome_absent(self, code):
        joined = " ".join(a.text for a in code)
        for junk in ("Последние документы", "Состояние базы", "РГП на ПХВ", "Популярные документы"):
            assert junk not in joined

    def test_structure_is_populated(self, code):
        assert code.stats["sections"] == 5
        assert code.stats["chapters"] == 23
        assert all(a.chapter for a in code.in_force)

    def test_known_article_parsed_correctly(self, code):
        a = code.by_number["54"]
        assert a.title == "Ограничение возможности расторжения трудового договора по инициативе работодателя"
        assert a.chapter == "Глава 4. ТРУДОВОЙ ДОГОВОР"
        assert a.clause_numbers == ("1", "2")
        assert a.clauses[0].text.startswith(
            "Не допускается расторжение трудового договора по инициативе работодателя"
        )

    def test_compound_article_parsed_correctly(self, code):
        a = code.by_number["73-1"]
        assert a.title == "Скользящий график работы"
        assert a.chapter == "Глава 6. РАБОЧЕЕ ВРЕМЯ"
