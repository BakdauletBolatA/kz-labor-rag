"""FastAPI — тонкая обёртка над пакетом ``kz_labor_rag``.

Логики здесь нет намеренно: eval-харнесс импортирует библиотеку напрямую, а не
ходит через HTTP, поэтому API и измерения не могут разъехаться. Всё, что
делает этот модуль, — превращает вызовы функций в JSON.

``/search`` отдаёт сырые скоры и id чанков: без них невозможно глазами
разобрать, почему поиск промахнулся.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kz_labor_rag.config import load_config
from kz_labor_rag.eval.factory import build_generator
from kz_labor_rag.eval.judge import format_context
from kz_labor_rag.indexer import index_mismatch
from kz_labor_rag.retrieval.factory import build_retriever, build_store
from kz_labor_rag.retrieval.store import StoreError

log = logging.getLogger(__name__)

app = FastAPI(
    title="kz-labor-rag",
    description="Вопрос-ответ по Трудовому кодексу РК",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _components():
    config = load_config()
    return config, build_retriever(config), build_generator(config)


class ChunkHit(BaseModel):
    rank: int
    chunk_id: str
    score: float = Field(description="Сырой скор поиска, без нормализации")
    articles: list[str]
    clauses: list[str]
    article_title: str = ""
    chapter: str = ""
    text: str


class SearchResponse(BaseModel):
    query: str
    version: str
    hits: list[ChunkHit]


class AskResponse(BaseModel):
    question: str
    answer: str
    cited_articles: list[str]
    citations: list[str] = Field(default_factory=list, description="Проверенные «ст. N п. M»")
    refused: bool = False
    withheld: bool = Field(
        default=False, description="Модель ответила без верной ссылки, ответ заменён отказом"
    )
    version: str
    hits: list[ChunkHit]
    error: str | None = None


def _to_hits(chunks) -> list[ChunkHit]:
    return [
        ChunkHit(
            rank=c.rank,
            chunk_id=c.chunk_id,
            score=c.score,
            articles=list(c.articles),
            clauses=[f"{a}/{cl}" for a, cl in c.chunk.spans],
            article_title=c.chunk.article_title,
            chapter=c.chunk.chapter,
            text=c.text,
        )
        for c in chunks
    ]


@app.get("/health")
def health() -> dict[str, Any]:
    """Состояние сервиса и, главное, состояние индекса.

    Пустой или несоответствующий конфигу индекс — самая частая причина
    бессмысленной выдачи, поэтому он виден сразу, а не после первого запроса.
    """
    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": f"конфиг не читается: {exc}"}

    payload: dict[str, Any] = {"status": "ok", "version": config.version}
    try:
        store = build_store(config)
        payload["chunks_indexed"] = store.count()
        meta = store.read_meta()
        payload["index"] = {
            "built": meta is not None,
            "corpus_edition_date": (meta or {}).get("corpus_edition_date"),
            "embeddings_model": (meta or {}).get("embeddings_model"),
            "built_at": (meta or {}).get("built_at"),
        }
        if problem := index_mismatch(config, store):
            payload["status"] = "degraded"
            payload["index"]["problem"] = problem
    except StoreError as exc:
        payload["status"] = "degraded"
        payload["index"] = {"built": False, "problem": str(exc)}
    return payload


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, description="Поисковый запрос"),
    k: int | None = Query(default=None, ge=1, le=50),
) -> SearchResponse:
    config, retriever, _ = _components()
    try:
        hits = retriever.search(q, k or int(config.get("retrieval.top_k")))
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponse(query=q, version=retriever.version, hits=_to_hits(hits))


@app.get("/ask", response_model=AskResponse)
def ask(
    q: str = Query(min_length=1, description="Вопрос по Трудовому кодексу"),
    k: int | None = Query(default=None, ge=1, le=50),
) -> AskResponse:
    config, retriever, generator = _components()
    try:
        hits = list(retriever.search(q, k or int(config.get("retrieval.top_k"))))
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    generation = generator.generate(q, hits)
    return AskResponse(
        question=q,
        answer=generation.answer,
        cited_articles=list(generation.cited_articles),
        citations=list(generation.citations),
        refused=generation.refused,
        withheld=generation.withheld,
        version=retriever.version,
        hits=_to_hits(hits),
        error=generation.error,
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/ask/stream")
def ask_stream(
    q: str = Query(min_length=1, description="Вопрос по Трудовому кодексу"),
    k: int | None = Query(default=None, ge=1, le=50),
) -> StreamingResponse:
    """Ответ по мере генерации, Server-Sent Events.

    События: ``hits`` (найденные фрагменты), ``token`` (кусок текста), ``done``
    (итог). Ссылки проверяются только когда ответ дописан, поэтому решающий
    текст — ``done.answer``: если у ответа не нашлось ни одной верной ссылки,
    там будет отказ и ``withheld: true``, и клиент заменяет показанный текст.
    """
    config, retriever, generator = _components()
    try:
        hits = list(retriever.search(q, k or int(config.get("retrieval.top_k"))))
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def events():
        yield _sse("hits", [h.model_dump() for h in _to_hits(hits)])
        if hasattr(generator, "stream"):
            pieces = []
            try:
                for piece in generator.stream(q, hits):
                    pieces.append(piece)
                    yield _sse("token", {"text": piece})
            except Exception as exc:  # noqa: BLE001 — обрыв генерации сообщается клиенту
                yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
                return
            generation = generator.finish("".join(pieces).strip(), hits)
        else:
            generation = generator.generate(q, hits)
            if generation.answer:
                yield _sse("token", {"text": generation.answer})
        yield _sse(
            "done",
            {
                "answer": generation.answer,
                "citations": list(generation.citations),
                "refused": generation.refused,
                "withheld": generation.withheld,
                "error": generation.error,
            },
        )

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/context")
def context(q: str = Query(min_length=1), k: int | None = None) -> dict[str, str]:
    """Тот же контекст, что уходит генератору и судье.

    Отладочная ручка: если ответ выглядит странно, первый вопрос — что модель
    вообще видела.
    """
    config, retriever, _ = _components()
    hits = retriever.search(q, k or int(config.get("retrieval.top_k")))
    return {"query": q, "context": format_context(hits)}
