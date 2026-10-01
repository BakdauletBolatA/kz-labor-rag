"""Приведение русского текста к терминам для BM25.

Без морфологии BM25 на русском почти бесполезен: «работодатель»,
«работодателя», «работодателем» — три разных термина, и вопрос «уволили на
больничном» не совпадает с нормой «в период временной нетрудоспособности» ни
одним словом, но и «уволить» с «увольнения» тоже не совпадёт.

Два варианта за одним интерфейсом, чтобы их можно было сравнить замером:

- ``lemma`` — словарная лемматизация pymorphy3: «уволили» → «уволить».
  Точнее, но медленнее и зависит от словаря.
- ``stem`` — стеммер Snowball: «уволили» → «увол». Быстрее, без словаря, но
  иногда склеивает разные слова и не склеивает чередования.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре, «ё» приведена к «е»: в кодексе её нет, в вопросах бывает."""
    return TOKEN.findall(text.lower().replace("ё", "е"))


def _lemma() -> Callable[[str], str]:
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()

    @lru_cache(maxsize=200_000)
    def normal_form(word: str) -> str:
        return morph.parse(word)[0].normal_form.replace("ё", "е")

    return normal_form


def _stem() -> Callable[[str], str]:
    import snowballstemmer

    stemmer = snowballstemmer.stemmer("russian")

    @lru_cache(maxsize=200_000)
    def stem(word: str) -> str:
        return stemmer.stemWord(word)

    return stem


ANALYZERS: dict[str, Callable[[], Callable[[str], str]]] = {"lemma": _lemma, "stem": _stem}


class Analyzer:
    """Текст → список терминов."""

    def __init__(self, name: str) -> None:
        if name not in ANALYZERS:
            raise ValueError(f"неизвестный анализатор '{name}'. Известные: {', '.join(ANALYZERS)}")
        self.name = name
        self._normalize = ANALYZERS[name]()

    def __call__(self, text: str) -> list[str]:
        return [self._normalize(token) for token in tokenize(text)]
