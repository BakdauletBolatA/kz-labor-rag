"""Вырезание дословной цитаты из текста статьи.

Цитата в эталоне не пишется руками: указывается якорь, а фрагмент вокруг него
вырезается отсюда. Правило одно на весь проект — им пользуются и сборка
синтетической части (``scripts/build_synthetic_dataset.py``), и команда
``kzrag-corpus quote``, которой размечают реальные и казахские вопросы. Две
копии этого правила разошлись бы, и цитата, показанная человеку при разметке,
перестала бы совпадать с той, что окажется в датасете.

Модуль работает с текстом, а не с разбором: на вход строка, поэтому eval-часть
остаётся независимой от парсера.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

MAX_QUOTE_LEN = 600


def extract_evidence(article_text: str, anchor: str, *, max_len: int = MAX_QUOTE_LEN) -> str:
    """Вырезать из текста статьи дословный фрагмент вокруг якоря.

    Два дефекта, найденных на ревью, исправлены здесь.

    Первый: точка с запятой считалась концом предложения. В юридическом тексте
    «;» разделяет пункты перечня, и цитата обрывалась на первом же элементе.
    На вопросе про беременную и ночные смены это отрезало фрагмент ровно перед
    словами «беременные женщины» — то есть перед тем, ради чего вопрос написан.
    Концом предложения считается только точка.

    Второй: если после якоря точки с пробелом не встречалось (якорь в последнем
    предложении статьи), расширение молча не происходило вовсе и цитата
    оставалась равной якорю. Отсюда «в повышенном размере» без «но не ниже чем
    в полуторном размере». Теперь при отсутствии точки фрагмент доводится до
    конца текста статьи.

    Возвращается подстрока исходного текста — ни одного символа не дописывается.
    """
    pos = article_text.find(anchor)
    if pos < 0:
        raise ValueError(f"якорь не найден в тексте статьи: {anchor!r}")

    left = max(
        article_text.rfind(". ", 0, pos) + 2,
        article_text.rfind("\n", 0, pos) + 1,
        0,
    )
    tail_start = pos + len(anchor)
    end = re.search(r"\.(?:\s|$)", article_text[tail_start:])
    right = tail_start + (end.start() + 1 if end else len(article_text) - tail_start)

    fragment = article_text[left:right].strip()
    if len(fragment) > max_len:
        # Перечни в кодексе бывают на тысячу символов. Режем по границе слова,
        # начиная от самого якоря: он и есть то, что доказывает разметку.
        # Перевод строки — граница пункта: обрезка не должна уносить в цитату
        # обрывок соседнего пункта.
        cut = article_text[pos : pos + max_len].split("\n", 1)[0]
        fragment = cut[: cut.rfind(" ")].strip() if " " in cut else cut.strip()
    return fragment


def clauses_of_quote(clauses: Sequence[tuple[str, str]], quote: str) -> list[str]:
    """Номера пунктов, текст которых накрывает цитата.

    ``clauses`` — пары (номер, текст) в порядке статьи; склеиваются через
    перевод строки ровно так же, как парсер собирает ``Article.text``. Цитата
    ищется с точностью до пробельных символов: в рукописных записях перенос
    строки мог превратиться в пробел.
    """
    text = "\n".join(body for _, body in clauses)
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text) if pattern else None
    if match is None:
        raise ValueError(f"цитата не найдена в тексте пунктов: {quote[:80]!r}")

    found: list[str] = []
    cursor = 0
    for number, body in clauses:
        start, end = cursor, cursor + len(body)
        if start < match.end() and match.start() < end:
            found.append(number)
        cursor = end + 1
    return found
