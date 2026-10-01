"""Ссылки на статьи и пункты в ответе и проверка их по показанному контексту.

Правило ответа: каждое утверждение опирается на пункт, который модель
действительно видела. Проверяется это кодом, а не просьбой в промпте:

- ссылки разбираются из строки «Источники: …» (если её нет — из всего текста);
- ссылка, которой нет среди пунктов показанных фрагментов, отбрасывается;
- ответ, у которого не осталось ни одной верной ссылки, не показывается —
  вместо него возвращается отказ. Это и есть «не придумывает»: ответ без
  опоры на текст кодекса хуже честного «не нашлось».
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from kz_labor_rag.types import RetrievedChunk, normalize_article, normalize_clause

REFUSAL = "В Трудовом кодексе ответа на это не нашлось."

_NUMBER = r"\d{1,3}(?:-\d{1,2})?"
_CITATION = re.compile(
    rf"\b(?:ст\.?|стать[яиеюй]\w*)\s*({_NUMBER})"
    rf"(?:\s*,?\s*(?:п\.?|пункт\w*)\s*({_NUMBER}))?",
    re.IGNORECASE,
)
_SOURCES = re.compile(r"^\s*источники\s*:(.*)$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Citation:
    """Ссылка «ст. N» или «ст. N п. M». ``clause=None`` — ссылка на статью целиком."""

    article: str
    clause: str | None = None

    def __str__(self) -> str:
        return f"ст. {self.article}" + (f" п. {self.clause}" if self.clause else "")


def cite(article: str, clause: str) -> str:
    """Подпись пункта в контексте. Пункт без номера подписывается статьёй."""
    return f"ст. {article}" + (f" п. {clause}" if clause else "")


def parse_citations(text: str) -> list[Citation]:
    """Ссылки из ответа: из строки «Источники:», а без неё — из всего текста."""
    sources = _SOURCES.findall(text)
    haystack = " ".join(sources) if sources else text
    seen: dict[Citation, None] = {}
    for article, clause in _CITATION.findall(haystack):
        ref = Citation(normalize_article(article), normalize_clause(clause) if clause else None)
        seen.setdefault(ref, None)
    return list(seen)


def strip_sources(text: str) -> str:
    """Текст ответа без служебной строки «Источники:»."""
    return _SOURCES.sub("", text).strip()


# Модель отказывается и своими словами: «ответа на этот вопрос нет», «…не
# нашлось», порой со строкой «Источники». Отказом считается только первое
# предложение такого вида — фраза «других оснований нет» внутри ответа им не является.
_REFUSAL = re.compile(
    r"^в трудовом кодексе[^.]{0,60}?"
    r"(?:ответа[^.]{0,40}?(?:не нашлось|нет|не найдено|отсутствует)|нет ответа)"
)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.match(" ".join(text.lower().split())))


@dataclass(frozen=True)
class GroundedAnswer:
    """Ответ после проверки ссылок.

    ``withheld`` — модель ответила, но ни одна её ссылка не нашлась в
    показанных фрагментах; пользователю вместо ответа показан отказ.
    """

    text: str
    citations: tuple[Citation, ...]
    invalid_citations: tuple[Citation, ...]
    refused: bool
    withheld: bool
    raw: str


def ground(raw: str, context: Sequence[RetrievedChunk]) -> GroundedAnswer:
    """Проверить ответ модели по показанному контексту."""
    spans = {(a, c) for hit in context for a, c in hit.chunk.spans}
    articles = {a for hit in context for a in hit.articles}

    def valid(ref: Citation) -> bool:
        if ref.clause is None:
            return ref.article in articles
        return (ref.article, ref.clause) in spans

    cited = parse_citations(raw)
    good = tuple(r for r in cited if valid(r))
    bad = tuple(r for r in cited if not valid(r))

    if is_refusal(raw):
        return GroundedAnswer(REFUSAL, (), bad + good, refused=True, withheld=False, raw=raw)
    if not good:
        return GroundedAnswer(REFUSAL, (), bad, refused=True, withheld=True, raw=raw)
    body = strip_sources(raw)
    sources = "Источники: " + "; ".join(str(r) for r in good)
    return GroundedAnswer(f"{body}\n\n{sources}", good, bad, refused=False, withheld=False, raw=raw)
