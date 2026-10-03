"""Общие фикстуры и фейки.

Харнесс собирается до RAG, поэтому все тесты работают на фейковом поиске с
заранее заданной выдачей. Это не костыль, а условие задачи: метрики обязаны
быть проверены на известных ответах раньше, чем появится система, качество
которой они будут измерять.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

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


def ranked(*refs: str) -> list[RetrievedChunk]:
    """Выдача из чанков по одной статье на чанк, скор убывает вместе с рангом.

    ``"54"`` — чанк статьи без пунктов, ``"54/1"`` — чанк с пунктом 1 статьи 54.
    """
    out = []
    for i, ref in enumerate(refs):
        article, _, clause = ref.partition("/")
        chunk = make_chunk(article, (clause,) if clause else (), cid=f"c{i}-{ref}")
        out.append(RetrievedChunk(chunk=chunk, score=1.0 - i * 0.1, rank=i + 1))
    return out


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
                "split": "all",
            },
        }
    )


# --- защита боевых ресурсов от фикстур --------------------------------------
#
# Тесты хранилища работают с той же базой, что и боевой индекс, — просто с
# другими таблицами. Один раз это уже стоило метаданных: drop() сносил таблицу
# index_meta, имя которой было общим на всю базу, и запись о том, чем построен
# боевой индекс, исчезала. Чанки при этом оставались, поиск отвечал, а проверка
# соответствия молчала не потому, что всё сошлось, а потому что сравнивать
# стало не с чем.
#
# Аудит показал, что сейчас общих ресурсов больше нет. Но ровно это же было
# верно и до того бага: ничто не мешало появиться следующему. Поэтому набор
# фиксируется снимком до прогона и сверяется после.

PRODUCTION_FILES = (
    "evals/datasets/kz_labor_v1.jsonl",
    "evals/questions.jsonl",
    "config/default.yaml",
    "src/kz_labor_rag/eval/prompts/REGISTRY.json",
    "src/kz_labor_rag/eval/prompts/answer_ru.v1.txt",
    "src/kz_labor_rag/eval/prompts/faithfulness_ru.v1.txt",
)

# Каталоги, в которые тест не имеет права ничего дописать.
PRODUCTION_DIRS = (".cache/embeddings", "evals/results", "data/processed")

# Таблица боевого индекса baseline. Тесты обязаны работать с любой другой.
PRODUCTION_TABLE = "chunks"


def _file_state() -> dict[str, str | None]:
    state: dict[str, str | None] = {}
    for name in PRODUCTION_FILES:
        path = Path(name)
        state[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
    for name in PRODUCTION_DIRS:
        directory = Path(name)
        if not directory.is_dir():
            state[f"{name}/*"] = ""
            continue
        # Размер и mtime, а не только имена: запись внутрь существующего
        # sqlite-кэша эмбеддингов список файлов не меняет.
        entries = [
            f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}"
            for p in sorted(directory.iterdir())
            if p.is_file()
        ]
        state[f"{name}/*"] = ",".join(entries)
    return state


def _db_state() -> dict[str, str | None]:
    """Состояние боевых таблиц. Пустой словарь, если базы нет."""
    dsn = os.environ.get("KZRAG_TEST_DSN", "postgresql://kzrag:kzrag@localhost:5432/kzrag")
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2) as conn:
            rows = conn.execute(
                "SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename=%s",
                (PRODUCTION_TABLE,),
            ).fetchone()[0]
            if not rows:
                return {"chunks": "нет таблицы"}
            count = conn.execute(f"SELECT count(*) FROM {PRODUCTION_TABLE}").fetchone()[0]
            meta = conn.execute(
                "SELECT count(*) FROM pg_tables WHERE schemaname='public' AND tablename=%s",
                (f"{PRODUCTION_TABLE}_meta",),
            ).fetchone()[0]
            return {"chunks": str(count), "chunks_meta": "есть" if meta else "НЕТ"}
    except Exception:  # noqa: BLE001 — базы может не быть, это не повод падать
        return {}


@pytest.fixture(scope="session", autouse=True)
def production_artifacts_untouched():
    """Прогон тестов не должен менять ничего боевого."""
    before = {**_file_state(), **_db_state()}
    yield
    after = {**_file_state(), **_db_state()}

    changed = {
        key: (before.get(key), after.get(key))
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    if changed:
        details = "\n".join(f"  {k}: {was!r} -> {now!r}" for k, (was, now) in changed.items())
        pytest.fail(
            "Тесты изменили боевые артефакты — фикстура работает с общим ресурсом "
            f"вместо изолированного:\n{details}",
            pytrace=False,
        )
