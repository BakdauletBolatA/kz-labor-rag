"""Построение индекса: корпус → чанки → векторы → pgvector.

В ``index_meta`` записывается, чем именно построен индекс: модель, подпись
чанкинга, отпечаток конфига, редакция корпуса. Поиск поверх векторов от другой
модели или другой нарезки даёт правдоподобные, но бессмысленные числа, поэтому
несоответствие обнаруживается явно, а не проявляется как «почему-то упал
recall».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from kz_labor_rag.config import Config
from kz_labor_rag.corpus.chunker import build_chunks, build_tokenizer, chunking_signature
from kz_labor_rag.corpus.parser import parse_file
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
        if self.overflowing_chunks:
            lines.append(
                f"обрезано моделью : {self.overflowing_chunks} чанков "
                f"(максимум {self.max_encoded_tokens} токенов)"
            )
        return lines


def count_overflow(
    texts: list[str], tokenizer, prefix: str, max_sequence_length: int
) -> tuple[int, int]:
    """Сколько чанков не влезает в окно модели вместе с префиксом.

    Ловушка, которую легко не заметить: ``chunk_size_tokens`` считает только
    содержимое, а модель кодирует содержимое ПЛЮС префикс ``passage: `` и два
    служебных токена. При ``chunk_size_tokens = 512`` и пределе модели 512
    хвост каждого полного чанка молча отбрасывается при кодировании — без
    исключения и без строки в логе.

    Само по себе это часть наивности baseline, но знать об этом надо: «уложить
    чанк в окно модели целиком» — готовая гипотеза для отдельной итерации.
    """
    # Два служебных токена: <s> и </s>.
    special = 2
    prefix_tokens = len(tokenizer.offsets(prefix))
    longest = 0
    overflowing = 0
    for text in texts:
        total = len(tokenizer.offsets(text)) + prefix_tokens + special
        longest = max(longest, total)
        if total > max_sequence_length:
            overflowing += 1
    return overflowing, longest


def index_mismatch(config: Config, store: PgVectorStore) -> str | None:
    """Проверить, что построенный индекс соответствует текущему конфигу.

    Возвращает описание расхождения или None. Отдельная функция, потому что
    проверку делают и CLI, и API, и eval-прогон.
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

    raw = config.get("corpus.raw_html")
    if not Path(raw).exists():
        raise FileNotFoundError(
            f"сырой HTML не найден: {raw}. "
            "Он не коммитится; команда для скачивания — в data/raw/SOURCE.md"
        )

    params = build_chunking_params(config)
    store = store or build_store(config)
    encoder = encoder or build_encoder(config)

    if rebuild:
        log.info("Пересоздаётся схема индекса")
        store.drop()
    store.create_schema()

    log.info("Разбирается корпус: %s", raw)
    code = parse_file(raw, strip_amendment_notes=config.get("corpus.strip_amendment_notes"))

    tokenizer = build_tokenizer(config.get("chunking.tokenizer"))
    chunks = build_chunks(code, tokenizer, params)
    log.info("Получено чанков: %d (стратегия %s)", len(chunks), params.strategy)

    overflowing, longest = count_overflow(
        [c.text for c in chunks],
        tokenizer,
        config.get("embeddings.passage_prefix"),
        int(config.get("embeddings.max_sequence_length")),
    )
    if overflowing:
        log.warning(
            "%d из %d чанков не влезают в окно модели (%d токенов): длиннейший — %d "
            "с учётом префикса '%s' и служебных токенов. Хвост таких чанков модель "
            "молча отбросит. Это следствие того, что chunk_size_tokens считает только "
            "содержимое. Уменьшение chunk_size_tokens — отдельная итерация.",
            overflowing,
            len(chunks),
            int(config.get("embeddings.max_sequence_length")),
            longest,
            config.get("embeddings.passage_prefix"),
        )

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
            "parser_version": code.parser_version,
            "chunks": written,
            "chunks_truncated_by_model": overflowing,
            "max_encoded_tokens": longest,
        }
    )

    return IndexReport(
        chunks=written,
        articles=len({a for c in chunks for a in c.articles}),
        edition_date=code.edition_date,
        signature=chunking_signature(params),
        seconds=time.perf_counter() - started,
        overflowing_chunks=overflowing,
        max_encoded_tokens=longest,
    )
