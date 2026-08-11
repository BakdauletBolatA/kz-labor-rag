"""Метрики качества поиска.

Единственное место, где определены формулы. Если формула меняется — это ломает
сравнимость всех прошлых строк EVALUATION.md, поэтому такое изменение обязано
сопровождаться бампом ``metrics_version`` ниже и пересчётом старых прогонов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from kz_labor_rag.types import ClauseRef, RetrievedChunk

METRICS_VERSION = "1.0"


def rank_articles(chunks: Sequence[RetrievedChunk]) -> list[int]:
    """Свернуть ранжированный список чанков в ранжированный список статей.

    Поиск возвращает чанки, а эталон размечен статьями. Одна статья обычно
    порождает несколько чанков, поэтому статья занимает ранг своего лучшего
    чанка, а её повторы выбрасываются. Порядок чанков считается уже
    отсортированным по убыванию релевантности.
    """
    seen: set[int] = set()
    ranked: list[int] = []
    for chunk in chunks:
        if chunk.article not in seen:
            seen.add(chunk.article)
            ranked.append(chunk.article)
    return ranked


def recall_at_k(required: Iterable[int], ranked_articles: Sequence[int], k: int) -> float:
    """Доля обязательных статей, попавших в топ-k статей.

    Знаменатель — число обязательных статей, а не k. Вопрос с двумя
    обязательными статьями, из которых нашлась одна, даёт 0.5.
    """
    required_set = set(required)
    if not required_set:
        raise ValueError("required_articles пуст — такой вопрос не должен попадать в датасет")
    found = required_set & set(ranked_articles[:k])
    return len(found) / len(required_set)


def strict_hit_at_k(required: Iterable[int], ranked_articles: Sequence[int], k: int) -> float:
    """1.0, если в топ-k попали *все* обязательные статьи, иначе 0.0.

    Жёсткая версия recall@k. Нужна отдельно, потому что средний recall@5 умеет
    расти за счёт вопросов с одной обязательной статьёй, маскируя полный провал
    на вопросах, где ответ собирается из связки норм.
    """
    required_set = set(required)
    if not required_set:
        raise ValueError("required_articles пуст — такой вопрос не должен попадать в датасет")
    return 1.0 if required_set <= set(ranked_articles[:k]) else 0.0


def reciprocal_rank(required: Iterable[int], ranked_articles: Sequence[int]) -> float:
    """Обратный ранг первой обязательной статьи в выдаче.

    Ранги считаются с единицы. Если ни одна обязательная статья не найдена во
    всей выдаче — 0.0. Обрезки по k нет намеренно: MRR должен отличать
    «нашлось на 6-м месте» от «не нашлось вообще», иначе после каждой итерации
    непонятно, промах это или почти попадание.
    """
    required_set = set(required)
    if not required_set:
        raise ValueError("required_articles пуст — такой вопрос не должен попадать в датасет")
    for rank, article in enumerate(ranked_articles, start=1):
        if article in required_set:
            return 1.0 / rank
    return 0.0


def clause_precision_at_k(
    preferred: ClauseRef, chunks: Sequence[RetrievedChunk], k: int
) -> float:
    """Доля чанков в топ-k, реально покрывающих эталонный пункт.

    Метрика справочная и считается только для вопросов с заполненным
    ``preferred_clause``. Её смысл — «сколько из выданного контекста бьёт в
    точку», поэтому она напрямую показывает пользу structure-aware чанкинга
    даже тогда, когда статейный recall уже упёрся в потолок.

    Важное свойство: потолок метрики ниже 1.0 и зависит от чанкинга. Один пункт
    физически не может лежать во всех k чанках, поэтому абсолютное значение
    интерпретировать бессмысленно — сравнивать можно только между итерациями
    при одинаковых k и датасете. Поэтому рядом всегда считается
    ``clause_hit_at_k``.
    """
    if k <= 0:
        raise ValueError("k должно быть положительным")
    top = chunks[:k]
    if not top:
        return 0.0
    covering = sum(1 for chunk in top if chunk.covers(preferred))
    return covering / k


def clause_hit_at_k(preferred: ClauseRef, chunks: Sequence[RetrievedChunk], k: int) -> float:
    """1.0, если хотя бы один чанк из топ-k покрывает эталонный пункт."""
    return 1.0 if any(chunk.covers(preferred) for chunk in chunks[:k]) else 0.0


def citation_validity(cited: Iterable[int], chunks: Sequence[RetrievedChunk]) -> float:
    """Доля статей, процитированных генератором, которые есть в выданном контексте.

    Детерминированная проверка без LLM: ловит самый грубый вид галлюцинации —
    ссылку на статью, которой поиск вообще не показывал. Дополняет
    faithfulness от судьи, но не заменяет её: сослаться можно и на реально
    выданную статью, переврав при этом её содержание.

    Ответ без единой ссылки на статью считается невалидным (0.0): для
    справочника по кодексу ответ без ссылки бесполезен.
    """
    cited_set = set(cited)
    if not cited_set:
        return 0.0
    available = {chunk.article for chunk in chunks}
    return len(cited_set & available) / len(cited_set)


@dataclass(frozen=True)
class QuestionMetrics:
    """Метрики по одному вопросу. Агрегаты собираются из них в runner."""

    question_id: str
    recall_at_k: float
    strict_hit_at_k: float
    reciprocal_rank: float
    citation_validity: float | None
    faithfulness: float | None
    clause_precision_at_k: float | None
    clause_hit_at_k: float | None
    retrieved_articles: list[int]
    required_articles: list[int]

    @property
    def is_retrieval_failure(self) -> bool:
        """Вопрос, на котором поиск не достал ни одной обязательной статьи.

        Пункт 5 плана требует смотреть, где baseline проваливается, до
        обсуждения гипотез. Это тот самый флаг.
        """
        return self.recall_at_k == 0.0
