"""Построение индекса: корпус → чанки → векторы → pgvector.

В ``index_meta`` записывается, чем именно построен индекс: модель, подпись
чанкинга, отпечаток конфига, редакция корпуса. Поиск поверх векторов от другой
модели или другой нарезки даёт правдоподобные, но бессмысленные числа, поэтому
несоответствие обнаруживается явно, а не проявляется как «почему-то упал
recall».
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from kz_labor_rag.config import Config
from kz_labor_rag.corpus.chunker import (
    EncodingBudget,
    build_chunks,
    build_tokenizer,
    chunking_signature,
)
from kz_labor_rag.corpus.parser import PARSER_VERSION, parse_file
from kz_labor_rag.embeddings.encoder import Encoder
from kz_labor_rag.retrieval.factory import build_chunking_params, build_encoder, build_store
from kz_labor_rag.retrieval.store import PgVectorStore

log = logging.getLogger(__name__)


@dataclass
class IndexReport:
    chunks: int
    articles: int
    edition_date: str | None
    signature: str
    seconds: float
    overflowing_chunks: int = 0
    max_encoded_tokens: int = 0

    def as_lines(self) -> list[str]:
        lines = [
            f"чанков в индексе : {self.chunks}",
            f"статей покрыто   : {self.articles}",
            f"редакция корпуса : {self.edition_date}",
            f"подпись чанкинга : {self.signature}",
            f"заняло, сек      : {self.seconds:.1f}",
        ]
        lines.append(f"длина кодирования: максимум {self.max_encoded_tokens} токенов")
        if self.overflowing_chunks:
            lines.append(f"ПЕРЕПОЛНЕНО     : {self.overflowing_chunks} чанков")
        return lines


class ChunkOverflowError(RuntimeError):
    """Чанк не влезает в окно модели.

    Фатально, а не предупреждение. Модель усекает такой чанк молча: ни
    исключения, ни строки в логе, просто часть текста не участвует в
    эмбеддинге. Индекс, построенный на усечённых чанках, даёт правдоподобные
    метрики, а следующие итерации начинают чинить усечение вместо того, чтобы
    улучшать поиск, — разделить эти эффекты в таблице уже невозможно.
    """


def make_budget(config: Config, tokenizer) -> EncodingBudget:
    """Бюджет кодирования: предел модели и способ измерить реальную длину.

    Меряется текст целиком, вместе с префиксом: sentencepiece режет границу
    между префиксом и началом текста иначе, чем каждую часть по отдельности,
    поэтому «длина содержимого плюс длина префикса» — не та цифра.
    """
    prefix = config.get("embeddings.passage_prefix")
    return EncodingBudget(
        measure=lambda text: tokenizer.encoded_length(prefix + text),
        limit=int(config.get("embeddings.max_sequence_length")),
    )


def assert_chunks_fit(chunks, budget: EncodingBudget) -> int:
    """Убедиться, что ни один чанк не переполняет окно. Возвращает максимум."""
    lengths = [(c.chunk_id, budget.measure(c.text)) for c in chunks]
    offenders = [(cid, n) for cid, n in lengths if n > budget.limit]
    if offenders:
        raise ChunkOverflowError(
            f"{len(offenders)} из {len(chunks)} чанков длиннее окна модели "
            f"({budget.limit} токенов). Самый длинный: {max(n for _, n in offenders)}. "
            f"Первые: {offenders[:3]}. "
            "Индексация остановлена: молча усечённые чанки испортили бы все "
            "последующие измерения."
        )
    return max((n for _, n in lengths), default=0)


def corpus_sha256(config: Config) -> str | None:
    """Хеш сохранённого HTML корпуса. ``None``, если файла нет.

    Сверять редакцию по хешу файла, а не по разобранной ``edition_date``:
    разбор стоит секунды, а ``index_mismatch`` дёргается и из ``/health``.
    Хеш при этом строже — он ловит и правку текста, при которой метка редакции
    в подвале не изменилась.
    """
    path = Path(config.path_of("corpus.raw_html"))
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_mismatch(config: Config, store: PgVectorStore) -> str | None:
    """Проверить, что построенный индекс соответствует текущему конфигу.

    Возвращает описание расхождения или None. Отдельная функция, потому что
    проверку делают и CLI, и API, и eval-прогон.

    Сверяются не только нарезка и модель. Идентификатор чанка позиционный, а
    обычная сборка делает upsert без удаления, поэтому любое изменение, от
    которого меняются тексты чанков, обязано приводить к пересозданию с нуля.
    Иначе новая редакция кодекса, давшая меньше чанков, оставляла бы хвост
    строк от прошлой — и поиск возвращал бы текст редакции, которой уже нет.
    """
    meta = store.read_meta()
    if meta is None:
        return "индекс не построен (нет записи в index_meta)"

    expected_signature = chunking_signature(build_chunking_params(config))
    expected_model = config.get("embeddings.model")

    if meta.get("chunking_signature") != expected_signature:
        return (
            "индекс построен другой нарезкой: "
            f"в базе {meta.get('chunking_signature')}, в конфиге {expected_signature}. "
            "Нужна переиндексация: kzrag-index build --rebuild"
        )
    if meta.get("embeddings_model") != expected_model:
        return (
            "индекс построен другой моделью эмбеддингов: "
            f"в базе {meta.get('embeddings_model')}, в конфиге {expected_model}. "
            "Нужна переиндексация: kzrag-index build --rebuild"
        )

    expected_corpus = corpus_sha256(config)
    if expected_corpus is None:
        return (
            "сырой HTML корпуса не найден, сверить редакцию индекса не с чем. "
            "Скачайте корпус: make corpus"
        )
    if meta.get("corpus_sha256") != expected_corpus:
        return (
            "индекс построен на другой редакции корпуса: "
            f"в базе {meta.get('corpus_sha256')}, на диске {expected_corpus}. "
            "Нужна переиндексация: kzrag-index build --rebuild"
        )
    if meta.get("parser_version") != PARSER_VERSION:
        return (
            "индекс построен другой версией парсера: "
            f"в базе {meta.get('parser_version')}, в коде {PARSER_VERSION}. "
            "Нужна переиндексация: kzrag-index build --rebuild"
        )

    # Число строк против того, что записала сборка: остаточные строки от
    # прошлого индекса ловятся независимо от причины их появления.
    rows = store.count()
    if meta.get("chunks") is not None and rows != meta["chunks"]:
        return (
            f"в таблице {rows} строк, а последняя сборка записала {meta['chunks']}. "
            "Похоже на остаток от прошлого индекса. "
            "Нужна переиндексация: kzrag-index build --rebuild"
        )
    return None


def build_index(
    config: Config,
    *,
    rebuild: bool = False,
    encoder: Encoder | None = None,
    store: PgVectorStore | None = None,
) -> IndexReport:
    import time

    started = time.perf_counter()

    raw = config.path_of("corpus.raw_html")
    if not Path(raw).exists():
        raise FileNotFoundError(
            f"сырой HTML не найден: {raw}. "
            "Он не коммитится; команда для скачивания — в data/raw/SOURCE.md"
        )

    params = build_chunking_params(config)
    store = store or build_store(config)
    encoder = encoder or build_encoder(config)

    # Обычная сборка делает upsert поверх существующих строк. Этого достаточно,
    # только если нарезка не менялась: при другой нарезке идентификаторы чанков
    # перестают совпадать, и от прошлого индекса остались бы строки, которым
    # в корпусе уже ничего не соответствует. Поиск начал бы возвращать текст,
    # собранный по устаревшим границам, — и заметить это по логам невозможно.
    if not rebuild and store.count() > 0:
        if problem := index_mismatch(config, store):
            log.warning(
                "Индекс не соответствует конфигу (%s). Пересоздаю с нуля: "
                "дописывание поверх оставило бы чанки от прошлой нарезки.",
                problem,
            )
            rebuild = True

    if rebuild:
        log.info("Пересоздаётся схема индекса")
        store.drop()
    store.create_schema()

    log.info("Разбирается корпус: %s", raw)
    code = parse_file(raw, strip_amendment_notes=config.get("corpus.strip_amendment_notes"))

    tokenizer = build_tokenizer(config.get("chunking.tokenizer"))
    budget = make_budget(config, tokenizer)
    chunks = build_chunks(code, tokenizer, params, budget)
    log.info("Получено чанков: %d (стратегия %s)", len(chunks), params.strategy)

    # Ужимание окна уже учло оверхед, но проверка остаётся: она защищает от
    # случая, когда чанки пришли не из нашего чанкера или подгонка не сошлась.
    longest = assert_chunks_fit(chunks, budget)
    log.info("Максимальная длина кодирования: %d из %d токенов", longest, budget.limit)

    vectors = encoder.encode_passages([c.text for c in chunks])
    written = store.upsert(chunks, vectors)

    store.write_meta(
        {
            "version": config.version,
            "description": config.get_or("description", ""),
            "chunking_signature": chunking_signature(params),
            "chunking": config.section("chunking"),
            "embeddings_model": config.get("embeddings.model"),
            "encoder": encoder.descriptor,
            "config_fingerprint": config.fingerprint,
            "corpus_edition_date": code.edition_date,
            "corpus_sha256": corpus_sha256(config),
            "parser_version": code.parser_version,
            "chunks": written,
            "chunks_truncated_by_model": 0,
            "max_encoded_tokens": longest,
            "model_window": budget.limit,
        }
    )

    return IndexReport(
        chunks=written,
        articles=len({a for c in chunks for a in c.articles}),
        edition_date=code.edition_date,
        signature=chunking_signature(params),
        seconds=time.perf_counter() - started,
        overflowing_chunks=0,
        max_encoded_tokens=longest,
    )
