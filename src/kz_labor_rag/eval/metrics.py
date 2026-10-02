"""Метрики качества поиска.

Единственное место, где определены формулы. Если формула меняется — это ломает
сравнимость всех прошлых прогонов, поэтому такое изменение обязано
сопровождаться бампом ``METRICS_VERSION``: ``kzrag-eval compare`` откажется
ставить рядом прогоны разных версий.

Окно ``k`` отсчитывается по чанкам: топ-k чанков — ровно то, что уходит в
генератор.

Единица попадания — **пункт**, а не статья (версия 3.0). Ответ лежит в
конкретном пункте, и чанк с другим пунктом той же статьи генератору его не
даёт. Статейная метрика засчитывала такой чанк как попадание: в ст. 52 больше
двадцати оснований увольнения, и любой её кусок «находил» нужную статью.
Статейный recall остаётся отдельной метрикой ``article_recall@k`` — разница
между ним и ``recall@k`` показывает, сколько попаданий были «в ту статью, но не
туда».

Чанк засчитывается за все пункты, текст которых в него попал. Поэтому длинные
чанки покрывают больше пунктов при том же k, и сравнивать стратегии чанкинга по
recall можно только вместе с размером выданного контекста.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kz_labor_rag.types import ClauseRef, RetrievedChunk

METRICS_VERSION = "3.0"


def rank_articles(chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Свернуть ранжированный список чанков в ранжированный список статей.

    Статья занимает ранг своего лучшего чанка, повторы выбрасываются. Чанк
    засчитывается за все статьи, текст которых в него попал. Нужна для
    отладочных дампов и статейного recall.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for chunk in chunks:
        for article in chunk.articles:
            if article not in seen:
                seen.add(article)
                ranked.append(article)
    return ranked


def _required_set[T](required: Iterable[T], what: str) -> set[T]:
    required_set = set(required)
    if not required_set:
        raise ValueError(f"{what} пуст — такой вопрос не должен попадать в датасет")
    return required_set


def _covered(required: set[ClauseRef], chunks: Sequence[RetrievedChunk]) -> set[ClauseRef]:
    return {ref for ref in required if any(chunk.covers(ref) for chunk in chunks)}


def recall_at_k(
    required: Iterable[ClauseRef], chunks: Sequence[RetrievedChunk], k: int
) -> float:
    """Доля обязательных пунктов, покрытых топ-k чанками.

    Знаменатель — число обязательных пунктов, а не k: вопрос с двумя пунктами,
    из которых нашёлся один, даёт 0.5.
    """
    required_set = _required_set(required, "required_clauses")
    return len(_covered(required_set, chunks[:k])) / len(required_set)


def strict_hit_at_k(
    required: Iterable[ClauseRef], chunks: Sequence[RetrievedChunk], k: int
) -> float:
    """1.0, если топ-k чанков покрыли *все* обязательные пункты, иначе 0.0.

    Средний recall растёт за счёт вопросов с одним пунктом и маскирует провал
    там, где ответ собирается из нескольких норм.
    """
    required_set = _required_set(required, "required_clauses")
    return 1.0 if _covered(required_set, chunks[:k]) == required_set else 0.0


def reciprocal_rank(required: Iterable[ClauseRef], chunks: Sequence[RetrievedChunk]) -> float:
    """Обратный ранг первого чанка, покрывающего хотя бы один обязательный пункт.

    Обрезки по k нет намеренно: «нашлось в шестом чанке» и «не нашлось вообще»
    должны различаться. Если ничего не нашлось во всей выдаче — 0.0.
    """
    required_set = _required_set(required, "required_clauses")
    for rank, chunk in enumerate(chunks, start=1):
        if any(chunk.covers(ref) for ref in required_set):
            return 1.0 / rank
    return 0.0


def article_recall_at_k(
    required: Iterable[str], chunks: Sequence[RetrievedChunk], k: int
) -> float:
    """Доля обязательных *статей*, попавших в топ-k чанков.

    Справочная метрика, более мягкая, чем ``recall_at_k``: засчитывает чанк
    нужной статьи, даже если в нём нет нужного пункта.
    """
    required_set = _required_set(required, "required_articles")
    found = required_set & set(rank_articles(chunks[:k]))
    return len(found) / len(required_set)


def citation_validity(cited: Iterable[str], chunks: Sequence[RetrievedChunk]) -> float:
    """Доля статей, процитированных генератором, которые есть в выданном контексте.

    Детерминированная проверка без LLM: ловит ссылку на статью, которой поиск
    вообще не показывал. Ответ без единой ссылки считается невалидным (0.0).
    """
    cited_set = set(cited)
    if not cited_set:
        return 0.0
    available = {article for chunk in chunks for article in chunk.articles}
    return len(cited_set & available) / len(cited_set)


@dataclass(frozen=True)
class QuestionMetrics:
    """Метрики по одному вопросу. Агрегаты собираются из них в runner."""

    question_id: str
    recall_at_k: float
    strict_hit_at_k: float
    reciprocal_rank: float
    article_recall_at_k: float
    citation_validity: float | None
    faithfulness: float | None
    retrieved_articles: list[str]
    required_articles: list[str]
    required_clauses: list[str]

    @property
    def is_retrieval_failure(self) -> bool:
        """Поиск не покрыл ни одного обязательного пункта в топ-k."""
        return self.recall_at_k == 0.0
