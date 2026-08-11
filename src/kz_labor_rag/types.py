"""Базовые типы, общие для индексации, поиска и eval.

Живут отдельно от реализаций, чтобы eval-харнесс не зависел ни от одной
конкретной реализации поиска: харнесс знает только про ``Retriever`` и
``RetrievedChunk``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def normalize_clause(clause: str) -> str:
    """Привести номер пункта к каноническому виду.

    В ТК РК встречаются пункты вида ``1``, ``2-1``, ``15.``. Сравнение номеров
    обязано быть устойчиво к точкам и пробелам, иначе clause-метрика начнёт
    врать из-за форматирования, а не из-за качества поиска.
    """
    return clause.strip().rstrip(".").replace(" ", "")


def normalize_article(article: str | int) -> str:
    """Привести номер статьи к каноническому виду.

    Номер статьи — строка, а не число: помимо статей 1–204 в кодексе есть 19
    статей с составными номерами (``20-1``, ``73-1``, ``126-1``, ``203-1``),
    добавленных поправками. Целое число их не выражает, а выбросить их нельзя —
    среди них нормы про скользящий график, гарантии беременным и работу во
    вредных условиях.
    """
    return str(article).strip().rstrip(".").replace(" ", "")


def article_sort_key(article: str) -> tuple[int, int]:
    """Естественный порядок: 20 < 20-1 < 21."""
    base, _, sub = normalize_article(article).partition("-")
    try:
        return (int(base), int(sub) if sub else 0)
    except ValueError:
        return (10**6, 0)


@dataclass(frozen=True)
class ClauseRef:
    """Ссылка на конкретный пункт конкретной статьи."""

    article: str
    clause: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "clause", normalize_clause(self.clause))
        object.__setattr__(self, "article", normalize_article(self.article))

    def __str__(self) -> str:
        return f"ст. {self.article} п. {self.clause}"


@dataclass(frozen=True)
class Chunk:
    """Единица индексации.

    Наивный baseline режет текст по 512 токенов без учёта структуры, поэтому
    один чанк легко перекрывает несколько пунктов и даже границу между
    статьями. Отсюда два поля вместо одного:

    ``spans`` — все пары (статья, пункт), текст которых попал в чанк.
    ``articles`` — статьи из ``spans`` в порядке документа.

    Чанк засчитывается за **каждую** статью, текст которой в нём есть, а не
    только за ту, с которой он начинается. Иначе recall падал бы из-за того,
    как мы решили подписывать чанки, а не из-за качества поиска, и разница
    между baseline и structure-aware чанкингом оказалась бы артефактом
    разметки. Слабость наивного чанкинга и так видна — в clause-метриках и
    в том, что один чанк тащит за собой посторонние статьи.
    """

    chunk_id: str
    text: str
    articles: tuple[str, ...]
    spans: tuple[tuple[str, str], ...] = ()
    article_title: str = ""
    section: str = ""
    chapter: str = ""
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.articles:
            raise ValueError("чанк обязан относиться хотя бы к одной статье")
        object.__setattr__(self, "articles", tuple(normalize_article(a) for a in self.articles))
        object.__setattr__(
            self,
            "spans",
            tuple((normalize_article(a), normalize_clause(c)) for a, c in self.spans),
        )

    @property
    def article(self) -> str:
        """Статья, с которой чанк начинается. Для отладки и отображения."""
        return self.articles[0]

    @property
    def clauses(self) -> tuple[str, ...]:
        """Пункты головной статьи чанка."""
        return tuple(c for a, c in self.spans if a == self.article)

    def covers(self, ref: ClauseRef) -> bool:
        return (ref.article, normalize_clause(ref.clause)) in self.spans


@dataclass(frozen=True)
class RetrievedChunk:
    """Чанк вместе со скором, каким его вернул поиск.

    ``score`` отдаётся наружу как есть, без нормализации: в ``/search`` и в
    отладочных дампах нужен именно сырой скор, иначе разбирать промахи глазами
    невозможно.
    """

    chunk: Chunk
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def article(self) -> str:
        return self.chunk.article

    @property
    def articles(self) -> tuple[str, ...]:
        return self.chunk.articles

    @property
    def text(self) -> str:
        return self.chunk.text

    def covers(self, ref: ClauseRef) -> bool:
        return self.chunk.covers(ref)


@runtime_checkable
class Retriever(Protocol):
    """Всё, что eval-харнесс знает о поиске.

    Любая итерация — смена чанкинга, гибридный поиск, reranking — обязана
    уместиться за этим интерфейсом, чтобы харнесс не переписывался вместе с
    пайплайном и цифры оставались сравнимыми.
    """

    @property
    def version(self) -> str:
        """Метка версии пайплайна, попадающая в результат прогона."""

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]: ...
