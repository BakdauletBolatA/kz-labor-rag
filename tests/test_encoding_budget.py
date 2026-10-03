"""Ни один чанк не должен превышать окно модели.

Это не стилистическое требование, а защита от порчи всех будущих измерений.
Чанк длиннее окна модель молча усекает: при индексации нет ни исключения, ни
записи в логе, а часть текста просто не участвует в эмбеддинге. Если baseline
посчитан на усечённых чанках, любая следующая итерация будет частично чинить
усечение вместо того, чтобы улучшать поиск, и разделить эти два эффекта в
таблице EVALUATION.md уже невозможно.

Проверка меряет РЕАЛЬНУЮ длину кодирования, а не оценку «содержимое плюс
длина префикса плюс два». Оценка врёт: sentencepiece склеивает префикс с
началом текста иначе, чем токенизирует их по отдельности, и ошибается на
токен-другой.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kz_labor_rag.config import load_config
from kz_labor_rag.corpus.chunker import ChunkingParams, build_chunks, build_tokenizer
from kz_labor_rag.corpus.parser import parse_file
from kz_labor_rag.indexer import make_budget

RAW = Path("data/raw/adilet_K1500000414_rus.html")


def encoded_length(hf_tokenizer, prefix: str, text: str) -> int:
    """Сколько токенов модель получит на вход для этого чанка."""
    return len(hf_tokenizer(prefix + text, add_special_tokens=True)["input_ids"])


@pytest.mark.skipif(not RAW.exists(), reason="сырой HTML не скачан: make corpus")
class TestChunksFitTheModel:
    @pytest.fixture(scope="class")
    def built(self):
        transformers = pytest.importorskip("transformers")

        config = load_config()
        code = parse_file(config.get("corpus.raw_html"))
        params = ChunkingParams.from_config(config.section("chunking"))
        # Чанки строятся ровно так же, как их строит индексация: иначе тест
        # проверял бы не то, что попадает в индекс.
        chunker_tokenizer = build_tokenizer(config.get("chunking.tokenizer"))
        chunks = build_chunks(
            code, chunker_tokenizer, params, make_budget(config, chunker_tokenizer)
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.get("embeddings.model"), use_fast=True
        )
        return {
            "chunks": chunks,
            "tokenizer": tokenizer,
            "prefix": config.get("embeddings.passage_prefix"),
            "limit": int(config.get("embeddings.max_sequence_length")),
        }

    def test_no_chunk_exceeds_the_model_window(self, built):
        limit, prefix, tokenizer = built["limit"], built["prefix"], built["tokenizer"]
        offenders = [
            (c.chunk_id, encoded_length(tokenizer, prefix, c.text))
            for c in built["chunks"]
            if encoded_length(tokenizer, prefix, c.text) > limit
        ]
        assert not offenders, (
            f"{len(offenders)} из {len(built['chunks'])} чанков длиннее окна модели "
            f"({limit} токенов). Модель усечёт их молча, и baseline окажется посчитан "
            f"на неполном тексте. Самый длинный: {max(n for _, n in offenders)} токенов. "
            f"Первые: {offenders[:3]}"
        )

    def test_query_side_also_fits(self, built):
        """Запросы короткие, но префикс запроса тоже занимает место."""
        config = load_config()
        query_prefix = config.get("embeddings.query_prefix")
        longest = "Меня хотят уволить во время отпуска по уходу за ребёнком " * 3
        assert encoded_length(built["tokenizer"], query_prefix, longest) <= built["limit"]

    def test_the_check_can_actually_fail(self, built):
        """Страховка от теста, который проходит по недосмотру.

        Если бы проверка была нечувствительной, она молчала бы и на заведомо
        слишком длинном тексте.
        """
        too_long = "слово " * 5000
        assert encoded_length(built["tokenizer"], built["prefix"], too_long) > built["limit"]
