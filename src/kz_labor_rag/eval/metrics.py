"""Метрики качества поиска.

Единственное место, где определены формулы. Если формула меняется — это ломает
сравнимость всех прошлых строк EVALUATION.md, поэтому такое изменение обязано
сопровождаться бампом ``metrics_version`` ниже и пересчётом старых прогонов.

Окно ``k`` во всех метриках отсчитывается **по чанкам**, а не по статьям.
Версия 1.0 считала его по статьям: свёртка разворачивала k чанков в плоский
список статей, и метрика брала первые k из него. При наивной нарезке один чанк
накрывает около четырёх статей, поэтому recall@5 фактически видел первый чанк с
четвертью из пяти, а остальные три с лишним чанка — те самые, что уходят в
генератор, — в метрику не попадали вовсе. Побочные следствия были ровно те же:
``acceptable_articles`` занимали места в окне и вытесняли обязательные статьи,
хотя документация обещает, что они не штрафуются; статьи внутри одного чанка
получали разные ранги по порядку в документе, то есть MRR зависел от нарезки;
а clause-метрики, считавшие окно по чанкам, стояли в одной таблице с recall и
меряли окно другого размера — до противоречия «точный пункт найден, а его
статья не в топ-5».
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kz_labor_rag.types import ClauseRef, RetrievedChunk

METRICS_VERSION = "2.0"


def rank_articles(chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Свернуть ранжированный список чанков в ранжированный список статей.

    Поиск возвращает чанки, а эталон размечен статьями. Одна статья обычно
    порождает несколько чанков, поэтому статья занимает ранг своего лучшего
    чанка, а её повторы выбрасываются. Порядок чанков считается уже
    отсортированным по убыванию релевантности.

    Чанк засчитывается за все статьи, текст которых в него попал, а не только
    за головную: при наивном чанкинге по 512 токенов чанк регулярно пересекает
    границу статей, и учитывать только первую значило бы штрафовать поиск за
    способ нарезки, а не за качество выдачи.

    Обрезать результат по k нельзя: внутри чанка статьи идут в порядке
    документа, и такая обрезка резала бы по порядку, не имеющему отношения к
    релевантности. Окно применяется к чанкам до вызова.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for chunk in chunks:
        for article in chunk.articles:
            if article not in seen:
                seen.add(article)
                ranked.append(article)
    return ranked


def _required_set(required: Iterable[str]) -> set[str]:
    required_set = set(required)
    if not required_set:
        raise ValueError("required_articles пуст — такой вопрос не должен попадать в датасет")
    return required_set


def recall_at_k(required: Iterable[str], chunks: Sequence[RetrievedChunk], k: int) -> float:
    """Доля обязательных статей, найденных в топ-k чанков.

    Знаменатель — число обязательных статей, а не k. Вопрос с двумя
    обязательными статьями, из которых нашлась одна, даёт 0.5.

    Топ-k чанков — это ровно то, что система показывает генератору, поэтому
    метрика измеряет выданный контекст, а не его первую четверть.
    """
    required_set = _required_set(required)
    found = required_set & set(rank_articles(chunks[:k]))
    return len(found) / len(required_set)


def strict_hit_at_k(required: Iterable[str], chunks: Sequence[RetrievedChunk], k: int) -> float:
    """1.0, если в топ-k чанков попали *все* обязательные статьи, иначе 0.0.

    Жёсткая версия recall@k. Нужна отдельно, потому что средний recall@k умеет
    расти за счёт вопросов с одной обязательной статьёй, маскируя полный провал
    на вопросах, где ответ собирается из связки норм.
    """
    required_set = _required_set(required)
    return 1.0 if required_set <= set(rank_articles(chunks[:k])) else 0.0


def reciprocal_rank(required: Iterable[str], chunks: Sequence[RetrievedChunk]) -> float:
    """Обратный ранг первого чанка, содержащего обязательную статью.

    Ранги считаются с единицы и по чанкам: все статьи одного чанка найдены
    одновременно, одним попаданием поиска, и раздавать им разные ранги значило
    бы мерить порядок статей в документе. Если ни одна обязательная статья не
    найдена во всей выдаче — 0.0.

    Обрезки по k нет намеренно: MRR должен отличать «нашлось в шестом чанке» от
    «не нашлось вообще», иначе после итерации непонятно, промах это или почти
    попадание.
    """
    required_set = _required_set(required)
    for rank, chunk in enumerate(chunks, start=1):
        if required_set & set(chunk.articles):
            return 1.0 / rank
    return 0.0


def clause_precision_at_k(preferred: ClauseRef, chunks: Sequence[RetrievedChunk], k: int) -> float:
    """Доля чанков в топ-k, реально покрывающих эталонный пункт.

    Метрика справочная и считается только для вопросов с заполненным
    ``preferred_clause``. Её смысл — «сколько из выданного контекста бьёт в
    точку», поэтому она напрямую показывает пользу structure-aware чанкинга
    даже тогда, когда статейный recall уже упёрся в потолок.

    Знаменатель — именно k, а не число реально выданных чанков: иначе метрика
    подскочит на итерации, которая просто начала возвращать меньше чанков.

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


def citation_validity(cited: Iterable[str], chunks: Sequence[RetrievedChunk]) -> float:
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
    available = {article for chunk in chunks for article in chunk.articles}
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
    retrieved_articles: list[str]
    required_articles: list[str]

    @property
    def is_retrieval_failure(self) -> bool:
        """Вопрос, на котором поиск не достал ни одной обязательной статьи.

        Пункт 5 плана требует смотреть, где baseline проваливается, до
        обсуждения гипотез. Это тот самый флаг.
        """
        return self.recall_at_k == 0.0
