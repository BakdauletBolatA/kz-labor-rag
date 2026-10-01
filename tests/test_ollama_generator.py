"""Генератор на Ollama: HTTP подменяется, проверяются разбор и проверка ссылок."""

from __future__ import annotations

import io
import json

import pytest
from conftest import make_chunk

from kz_labor_rag.eval import generator as G
from kz_labor_rag.eval.citations import REFUSAL
from kz_labor_rag.types import RetrievedChunk

CONTEXT = [RetrievedChunk(chunk=make_chunk("54", ("1",), cid="c1"), score=0.9, rank=1)]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def ollama(monkeypatch):
    sent = []

    def install(body=None, lines=None, error=None):
        def urlopen(request, timeout):
            sent.append(json.loads(request.data))
            if error:
                raise error
            if lines is not None:
                return FakeResponse(b"\n".join(json.dumps(x).encode() for x in lines))
            return FakeResponse(json.dumps(body).encode())

        monkeypatch.setattr(G.urllib.request, "urlopen", urlopen)
        return G.OllamaGenerator(model="qwen2.5:7b-instruct", base_url="http://x:1"), sent

    return install


def test_answer_is_grounded_and_usage_recorded(ollama):
    gen, sent = ollama(
        body={
            "message": {"content": "Нельзя.\nИсточники: ст. 54 п. 1; ст. 99 п. 2"},
            "prompt_eval_count": 120,
            "eval_count": 15,
        }
    )
    out = gen.generate("Можно ли уволить на больничном?", CONTEXT)
    assert out.citations == ("ст. 54 п. 1",)
    assert out.invalid_citations == ("ст. 99 п. 2",)
    # citation_validity считается по ссылкам модели до проверки.
    assert out.cited_articles == ("54", "99")
    assert (out.input_tokens, out.output_tokens) == (120, 15)
    assert sent[0]["options"]["temperature"] == 0.0
    assert "[Фрагмент 1 — ст. 54 п. 1]" in sent[0]["messages"][0]["content"]


def test_answer_without_valid_citation_is_withheld(ollama):
    gen, _ = ollama(body={"message": {"content": "Работодателю грозит штраф."}})
    out = gen.generate("вопрос", CONTEXT)
    assert out.answer == REFUSAL
    assert out.withheld and out.refused


def test_stream_yields_pieces_until_done(ollama):
    gen, sent = ollama(
        lines=[
            {"message": {"content": "Нель"}, "done": False},
            {"message": {"content": "зя."}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
    )
    assert "".join(gen.stream("вопрос", CONTEXT)) == "Нельзя."
    assert sent[0]["stream"] is True


def test_http_failure_is_an_error_not_an_exception(ollama):
    gen, _ = ollama(error=OSError("connection refused"))
    out = gen.generate("вопрос", CONTEXT)
    assert not out.ok
    assert "connection refused" in out.error
