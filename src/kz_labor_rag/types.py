"""Базовые типы, общие для индексации, поиска и eval.

Живут отдельно от реализаций, чтобы eval-харнесс не зависел ни от одной
конкретной реализации поиска: харнесс знает только про ``Retriever`` и
``RetrievedChunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


def normalize_clause(clause: str) -> str:
    """Привести номер пункта к каноническому виду.

    В ТК РК встречаются пункты вида ``1``, ``2-1``, ``15.``. Сравнение номеров
    обязано быть устойчиво к точкам и пробелам, иначе clause-метрика начнёт
    врать из-за форматирования, а не из-за качества поиска.
    """
    return clause.strip().rstrip(".").replace(" ", "")


@dataclass(frozen=True)
class ClauseRef:
    """Ссылка на конкретный пункт конкретной статьи."""

    article: int
    clause: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "clause", normalize_clause(self.clause))

    def __str__(self) -> str:
        return f"ст. {self.article} п. {self.clause}"


@dataclass(frozen=True)
class Chunk:
    """Единица индексации.

    ``clauses`` — номера пунктов, которые чанк задевает целиком или частично.
    Наивный baseline режет текст по 512 токенов без учёта структуры, поэтому
    один чанк там легко перекрывает несколько пунктов и даже несколько статей.
    Поле ``article`` в таком случае хранит статью, которой принадлежит начало
    чанка: это сознательная слабость baseline, и она обязана быть видна в
    метриках, а не замазана на этапе разметки.
    """

    chunk_id: str
    text: str
    article: int
    article_title: str = ""
    clauses: tuple[str, ...] = ()
    section: str = ""
    chapter: str = ""
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def covers(self, ref: ClauseRef) -> bool:
        return self.article == ref.article and normalize_clause(ref.clause) in self.clauses


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
    def article(self) -> int:
        return self.chunk.article

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

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]:
        ...
