"""Вырезание цитат.

Обоснование разметки держится на том, что цитата — дословная подстрока текста
статьи. Оба дефекта, найденных на ревью 42 вопросов, жили именно здесь и
тестами покрыты не были: разметка выглядела обоснованной, а цитата обрывалась
раньше того места, ради которого вопрос написан.
"""

from __future__ import annotations

import pytest

from kz_labor_rag.corpus.evidence import extract_evidence


class TestVerbatim:
    def test_quote_is_a_substring_of_the_article(self):
        text = "Первое предложение. Второе предложение про срок. Третье."
        quote = extract_evidence(text, "про срок")
        assert quote in text

    def test_missing_anchor_raises(self):
        with pytest.raises(ValueError, match="якорь не найден"):
            extract_evidence("Какой-то текст.", "такого тут нет")

    def test_fragment_starts_at_sentence_boundary(self):
        text = "Первое предложение. Второе предложение про срок. Третье."
        assert extract_evidence(text, "про срок") == "Второе предложение про срок."


class TestSemicolonIsNotASentenceEnd:
    """Первый дефект: «;» разделяет пункты перечня, а не предложения.

    На вопросе про беременную и ночные смены цитата обрывалась ровно перед
    словами «беременные женщины».
    """

    TEXT = (
        "К работе в ночное время не допускаются: работники, не достигшие "
        "восемнадцатилетнего возраста; беременные женщины, предоставившие "
        "работодателю справку о беременности. Следующее предложение."
    )

    def test_list_survives_the_semicolon(self):
        quote = extract_evidence(self.TEXT, "К работе в ночное время не допускаются")
        assert "беременные женщины" in quote

    def test_stops_at_the_period(self):
        quote = extract_evidence(self.TEXT, "К работе в ночное время не допускаются")
        assert "Следующее предложение" not in quote


class TestAnchorInTheLastSentence:
    """Второй дефект: без точки после якоря расширение молча не происходило.

    Цитата оставалась равной якорю: «в повышенном размере» без «но не ниже чем
    в полуторном размере», то есть без самого ответа.
    """

    def test_fragment_reaches_the_end_of_the_article(self):
        text = "Оплата производится в повышенном размере, но не ниже чем в полуторном размере"
        quote = extract_evidence(text, "в повышенном размере")
        assert quote.endswith("в полуторном размере")

    def test_final_period_is_kept(self):
        text = "Оплата производится в повышенном размере, но не ниже полуторного."
        assert extract_evidence(text, "в повышенном размере").endswith("полуторного.")

    def test_anchor_alone_is_not_the_answer(self):
        text = "Оплата производится в повышенном размере, но не ниже полуторного."
        assert extract_evidence(text, "в повышенном размере") != "в повышенном размере"


class TestLongEnumerations:
    def test_long_fragment_is_cut_from_the_anchor(self):
        # Перечни в кодексе бывают на тысячу символов: обрезка идёт от якоря,
        # потому что доказывает разметку именно он.
        text = "Начало не про то. " + "Основания: " + "; ".join(f"пункт {i}" for i in range(200))
        quote = extract_evidence(text, "Основания:", max_len=100)
        assert quote.startswith("Основания:")
        assert len(quote) <= 100

    def test_cut_respects_word_boundary(self):
        text = "Основания: " + "; ".join(f"пункт номер {i}" for i in range(200))
        quote = extract_evidence(text, "Основания:", max_len=100)
        assert quote in text
        assert not quote.endswith(" ")


class TestTruncationStaysInsideTheClause:
    """Обрезка длинной цитаты не должна перетекать в следующий пункт.

    syn_043: пункт 2 ст. 51 длиннее 600 символов, обрезка от якоря захватывала
    перевод строки и обрывок пункта 3 — «Датой истечения срока трудового
    договора, заключенного на». Цитата при этом «доказывала» пункт, к вопросу
    не относящийся.
    """

    def test_truncated_quote_does_not_cross_a_line_break(self):
        # Как в syn_043: длинное предложение без точки внутри, якорь ближе к
        # концу пункта, чем max_len, — окно от якоря дотягивается до соседа.
        clause = "Вступление " + "длинная норма " * 50 + "якорь нормы " + "хвост " * 20 + "конец."
        text = clause + "\nДатой истечения срока трудового договора является день."
        quote = extract_evidence(text, "якорь нормы", max_len=600)
        assert "\n" not in quote
        assert "Датой истечения" not in quote
