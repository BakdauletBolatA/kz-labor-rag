"""Общие фикстуры и фейки.

Харнесс собирается до RAG, поэтому все тесты работают на фейковом поиске с
заранее заданной выдачей. Это не костыль, а условие задачи: метрики обязаны
быть проверены на известных ответах раньше, чем появится система, качество
которой они будут измерять.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from kz_labor_rag.config import Config
from kz_labor_rag.eval.generator import Generation
from kz_labor_rag.eval.judge import Judgement
from kz_labor_rag.types import Chunk, RetrievedChunk


def make_chunk(
    article: str,
    clauses: tuple[str, ...] = (),
    text: str = "",
    cid: str = "",
    extra_articles: tuple[str, ...] = (),
) -> Chunk:
    """Чанк одной статьи. ``extra_articles`` — для чанков, перешедших границу."""
    return Chunk(
        chunk_id=cid or f"a{article}-{'_'.join(clauses) or '0'}",
        text=text or f"Текст статьи {article}.",
        articles=(article, *extra_articles),
        spans=tuple((article, c) for c in clauses),
        article_title=f"Статья {article}",
    )


def ranked(*articles: str) -> list[RetrievedChunk]:
    """Выдача из чанков по одной статье на чанк, скор убывает вместе с рангом."""
    return [
        RetrievedChunk(chunk=make_chunk(a), score=1.0 - i * 0.1, rank=i + 1)
        for i, a in enumerate(articles)
    ]


class FakeRetriever:
    """Поиск с заранее прописанной выдачей: вопрос -> список статей."""

    version = "fake-v0"

    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> Sequence[RetrievedChunk]:
        self.calls.append((query, k))
        return self.responses.get(query, [])[:k]


class FakeGenerator:
    """Генератор, отдающий заранее заданный текст ответа."""

    descriptor = {"backend": "fake", "model": "fake", "prompt": "fake@v0"}

    def __init__(self, answer: str = "Согласно ст. 1 — да.") -> None:
        self.answer = answer

    def generate(self, question: str, context) -> Generation:
        from kz_labor_rag.eval.generator import extract_cited_articles

        return Generation(
            answer=self.answer,
            cited_articles=tuple(extract_cited_articles(self.answer)),
            backend="fake",
        )


class FakeJudge:
    """Судья с фиксированным вердиктом."""

    descriptor = {"backend": "fake", "model": "fake", "prompt": "fake@v0"}

    def __init__(self, verdict: str = "supported", score: float = 1.0) -> None:
        self.verdict = verdict
        self.score = score

    def judge(self, question: str, context, answer: str) -> Judgement:
        return Judgement(score=self.score, verdict=self.verdict, backend="fake")


@pytest.fixture
def config() -> Config:
    """Минимальный конфиг, достаточный для харнесса."""
    return Config(
        data={
            "version": "test-v0",
            "description": "конфиг для тестов",
            "chunking": {"strategy": "fixed_tokens", "chunk_size_tokens": 512},
            "eval": {
                "k": 5,
                "languages": ["ru", "kk"],
                "primary_language": "ru",
                "completeness": {
                    "enforce": True,
                    "min_ru": 60,
                    "min_kk": 15,
                    "min_real": 15,
                    "require_human_review": True,
                },
            },
        }
    )
