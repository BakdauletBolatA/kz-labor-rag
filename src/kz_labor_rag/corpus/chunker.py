"""Нарезка кодекса на чанки.

Baseline намеренно наивен: текст кодекса склеивается в один поток и режется
окнами фиксированной длины в токенах, структура документа игнорируется. Из-за
этого чанк регулярно пересекает границу пункта и границу статьи. Это не
недосмотр, а точка отсчёта — structure-aware нарезка станет отдельной
измеряемой итерацией, и разница между ними и есть предмет проекта.

Все параметры приходят из конфига. Стратегия выбирается по имени, чтобы
следующая итерация добавляла функцию, а не правила существующую.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kz_labor_rag.corpus.parser import Article, LaborCode
from kz_labor_rag.types import Chunk

# 1.1 — окно чанка ужимается под предел модели: раньше чанк на 512 токенов
# содержимого вместе с префиксом «passage: » и служебными токенами давал 516–517
# при пределе 512, и хвост молча отбрасывался при кодировании. Версия входит в
# подпись чанкинга, поэтому старый индекс и старый кэш эмбеддингов
# обесцениваются автоматически.
CHUNKER_VERSION = "1.1"

# Сколько раз ужимать окно, прежде чем сдаться. На практике хватает одной-двух.
MAX_BUDGET_ATTEMPTS = 5


@runtime_checkable
class Tokenizer(Protocol):
    """Токенизатор, умеющий возвращать границы токенов в исходном тексте.

    Границы нужны, чтобы по окну токенов восстановить, какие статьи и пункты
    в него попали. Без офсетов пришлось бы считать длину «на глаз», и «512
    токенов» перестали бы означать 512 токенов той модели, которой мы кодируем.
    """

    def offsets(self, text: str) -> list[tuple[int, int]]: ...

    def encoded_length(self, text: str) -> int:
        """Сколько токенов получит модель, включая служебные."""


@dataclass(frozen=True)
class EncodingBudget:
    """Предел модели и способ измерить, сколько токенов она реально получит.

    ``measure`` обязан мерить текст ровно в том виде, в каком он уйдёт в
    модель, — то есть вместе с префиксом ``passage: `` и служебными токенами.
    Оценка «содержимое плюс длина префикса плюс два» здесь не годится: она
    ошибается на токен, и ошибка проявляется молчаливым усечением.
    """

    measure: Callable[[str], int]
    limit: int


@dataclass(frozen=True)
class ChunkingParams:
    """Параметры нарезки. Собираются из секции ``chunking`` конфига."""

    strategy: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    prepend_article_header: bool = False

    @classmethod
    def from_config(cls, section: dict) -> ChunkingParams:
        return cls(
            strategy=section["strategy"],
            chunk_size_tokens=int(section["chunk_size_tokens"]),
            chunk_overlap_tokens=int(section["chunk_overlap_tokens"]),
            prepend_article_header=bool(section.get("prepend_article_header", False)),
        )

    def validate(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens должен быть положительным")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens не может быть отрицательным")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens должен быть меньше chunk_size_tokens, "
                "иначе окно не сдвигается и нарезка зацикливается"
            )


@dataclass(frozen=True)
class Segment:
    """Кусок потока, принадлежащий одному пункту одной статьи."""

    article: str
    clause: str
    start: int
    end: int


@dataclass(frozen=True)
class CorpusStream:
    """Весь кодекс одной строкой плюс карта «где какой пункт лежит»."""

    text: str
    segments: tuple[Segment, ...]

    def spans_in(self, start: int, end: int) -> tuple[tuple[str, str], ...]:
        """Пары (статья, пункт), попавшие в диапазон символов.

        Порядок документа сохраняется, дубли убираются: один пункт может
        встретиться в диапазоне только один раз, но статья — несколько.
        """
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for seg in self.segments:
            if seg.start >= end:
                break
            if seg.end <= start:
                continue
            key = (seg.article, seg.clause)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return tuple(out)


def build_stream(articles: Iterable[Article], *, prepend_article_header: bool) -> CorpusStream:
    """Склеить статьи в один поток, запомнив границы пунктов.

    Заголовок статьи добавляется в поток только если это включено в конфиге.
    В baseline он выключен намеренно: приклеивание заголовка к чанку —
    очевидная и дешёвая гипотеза улучшения, и она должна быть измерена
    отдельной итерацией, а не достаться baseline'у бесплатно.
    """
    parts: list[str] = []
    segments: list[Segment] = []
    cursor = 0

    for article in articles:
        if prepend_article_header:
            header = article.heading + "\n"
            parts.append(header)
            # Заголовок относится к статье целиком, поэтому пункт пустой.
            segments.append(Segment(article.number, "", cursor, cursor + len(header)))
            cursor += len(header)

        for clause in article.clauses:
            body = clause.text + "\n"
            parts.append(body)
            segments.append(Segment(article.number, clause.number, cursor, cursor + len(body)))
            cursor += len(body)

    return CorpusStream(text="".join(parts), segments=tuple(segments))


def _cut(
    stream: CorpusStream,
    offsets: list[tuple[int, int]],
    content_budget: int,
    overlap: int,
    id_prefix: str,
) -> list[Chunk]:
    """Разрезать поток окнами по ``content_budget`` токенов."""
    step = content_budget - overlap
    chunks: list[Chunk] = []

    for index, start_token in enumerate(range(0, len(offsets), step)):
        window = offsets[start_token : start_token + content_budget]
        if not window:
            break

        char_start, char_end = window[0][0], window[-1][1]
        text = stream.text[char_start:char_end].strip()
        if not text:
            continue

        spans = stream.spans_in(char_start, char_end)
        if not spans:
            continue

        articles: list[str] = []
        for article, _ in spans:
            if article not in articles:
                articles.append(article)

        chunks.append(
            Chunk(
                chunk_id=f"{id_prefix}{index:05d}",
                text=text,
                articles=tuple(articles),
                spans=spans,
                char_start=char_start,
                char_end=char_end,
            )
        )

        if start_token + content_budget >= len(offsets):
            break

    return chunks


def chunk_fixed_tokens(
    stream: CorpusStream,
    tokenizer: Tokenizer,
    params: ChunkingParams,
    budget: EncodingBudget | None = None,
    *,
    id_prefix: str = "c",
) -> list[Chunk]:
    """Нарезать поток окнами фиксированной длины, игнорируя структуру.

    Стратегия нарезки не меняется: те же окна фиксированной длины по всему
    потоку, структура документа по-прежнему игнорируется. Меняется только
    величина окна — если задан ``budget``, содержимое ужимается ровно
    настолько, чтобы вместе с префиксом и служебными токенами уложиться в
    предел модели.

    Почему подгонка итеративная, а не «вычесть длину префикса»: sentencepiece
    склеивает префикс с началом текста иначе, чем токенизирует их по
    отдельности, поэтому оверхед не постоянен — на реальном корпусе он гуляет
    между 4 и 5 токенами. Вычитание константы оставило бы часть чанков за
    пределом, а обнаружилось бы это опять молча.
    """
    params.validate()
    offsets = tokenizer.offsets(stream.text)
    if not offsets:
        return []

    content_budget = params.chunk_size_tokens
    if budget is None:
        return _cut(stream, offsets, content_budget, params.chunk_overlap_tokens, id_prefix)

    for _ in range(MAX_BUDGET_ATTEMPTS):
        chunks = _cut(stream, offsets, content_budget, params.chunk_overlap_tokens, id_prefix)
        if not chunks:
            return chunks
        worst = max(budget.measure(c.text) for c in chunks)
        if worst <= budget.limit:
            return chunks
        content_budget -= worst - budget.limit
        if content_budget <= params.chunk_overlap_tokens:
            break

    raise ValueError(
        f"не удалось подобрать размер окна под предел модели ({budget.limit} токенов) "
        f"за {MAX_BUDGET_ATTEMPTS} попыток. Последний бюджет содержимого: {content_budget}."
    )


def chunk_by_article(
    articles: Sequence[Article], params: ChunkingParams, *, id_prefix: str = "a"
) -> list[Chunk]:
    """Structure-aware нарезка: один чанк — одна статья.

    В baseline не используется. Лежит здесь, чтобы соответствующая итерация
    сводилась к смене одного значения в конфиге, а не к правке кода.
    """
    chunks: list[Chunk] = []
    for index, article in enumerate(articles):
        text = article.full_text if params.prepend_article_header else article.text
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{id_prefix}{index:05d}",
                text=text.strip(),
                articles=(article.number,),
                spans=tuple((article.number, c.number) for c in article.clauses),
                article_title=article.title,
                section=article.section,
                chapter=article.chapter,
            )
        )
    return chunks


STRATEGIES: dict[str, str] = {
    "fixed_tokens": "окна фиксированной длины по всему потоку, структура игнорируется",
    "article": "один чанк — одна статья",
}


def build_chunks(
    code: LaborCode,
    tokenizer: Tokenizer,
    params: ChunkingParams,
    budget: EncodingBudget | None = None,
) -> list[Chunk]:
    """Точка входа: нарезать кодекс согласно конфигу.

    Исключённые статьи в индекс не попадают: текста у них нет, а место в
    выдаче они занимали бы.
    """
    articles = code.in_force

    if params.strategy == "fixed_tokens":
        stream = build_stream(articles, prepend_article_header=params.prepend_article_header)
        return chunk_fixed_tokens(stream, tokenizer, params, budget)
    if params.strategy == "article":
        return chunk_by_article(articles, params)

    known = ", ".join(f"'{name}'" for name in STRATEGIES)
    raise ValueError(f"неизвестная стратегия чанкинга '{params.strategy}'. Известные: {known}")


def chunking_signature(params: ChunkingParams) -> str:
    """Отпечаток параметров нарезки — часть ключа кэша эмбеддингов."""
    payload = (
        f"{CHUNKER_VERSION}|{params.strategy}|{params.chunk_size_tokens}|"
        f"{params.chunk_overlap_tokens}|{int(params.prepend_article_header)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class WhitespaceTokenizer:
    """Токенизация по пробелам.

    Нужна там, где важна только логика нарезки, а не точное соответствие
    токенизатору модели: в тестах и в отладке без скачивания весов.
    Для индексации не годится — «512 токенов» тогда не те 512 токенов.
    """

    def offsets(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        pos = 0
        for word in text.split():
            start = text.index(word, pos)
            out.append((start, start + len(word)))
            pos = start + len(word)
        return out

    def encoded_length(self, text: str) -> int:
        # Два служебных токена — как у моделей семейства BERT/XLM-R.
        return len(self.offsets(text)) + 2


class HFTokenizer:
    """Токенизатор модели эмбеддингов.

    Длина чанка обязана считаться тем же токенизатором, которым текст потом
    кодируется, иначе «512 токенов» — не те 512 токенов, и часть чанков молча
    обрезается при кодировании.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("нужен пакет transformers: pip install transformers") from exc
        self.model_name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self._tok.is_fast:  # pragma: no cover
            raise RuntimeError(f"токенизатор {model_name} не fast, а без него нет офсетов токенов")

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tok(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        # Токенизатор может вернуть пустые интервалы для служебных токенов.
        return [(s, e) for s, e in encoded["offset_mapping"] if e > s]

    def encoded_length(self, text: str) -> int:
        """Ровно то число токенов, которое получит модель.

        Считается на целом тексте, а не как сумма частей: sentencepiece
        по-разному режет границу между префиксом и началом текста.
        """
        return len(self._tok(text, add_special_tokens=True, verbose=False)["input_ids"])


def build_tokenizer(name: str, factory: Callable[[str], Tokenizer] | None = None) -> Tokenizer:
    if factory is not None:
        return factory(name)
    if name == "whitespace":
        return WhitespaceTokenizer()
    return HFTokenizer(name)
