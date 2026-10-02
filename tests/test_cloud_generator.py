"""Генератор на OpenAI-совместимом API (DeepSeek, xAI): клиент подменяется."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import make_chunk

from kz_labor_rag.eval.citations import REFUSAL
from kz_labor_rag.eval.generator import CloudGenerator
from kz_labor_rag.types import RetrievedChunk

CONTEXT = [RetrievedChunk(chunk=make_chunk("54", ("1",), cid="c1"), score=0.9, rank=1)]


class FakeCompletions:
    def __init__(self, content=None, chunks=None, error=None):
        self.content, self.chunks, self.error = content, chunks, error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
                for c in self.chunks
            )
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def generator(**kw):
    completions = FakeCompletions(**kw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    gen = CloudGenerator(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key_env="X",
        backend="deepseek",
        client=client,
    )
    return gen, completions


def test_answer_is_grounded_like_the_local_one():
    gen, calls = generator(content="Нельзя.\nИсточники: ст. 54 п. 1; ст. 99 п. 2")
    out = gen.generate("вопрос", CONTEXT)
    assert out.citations == ("ст. 54 п. 1",)
    assert out.invalid_citations == ("ст. 99 п. 2",)
    assert out.backend == "deepseek"
    assert (out.input_tokens, out.output_tokens) == (100, 20)
    assert calls.calls[0]["temperature"] == 0.0
    assert "[Фрагмент 1 — ст. 54 п. 1]" in calls.calls[0]["messages"][0]["content"]


def test_answer_without_valid_citation_is_withheld():
    gen, _ = generator(content="Заплатят вдвое.")
    assert gen.generate("вопрос", CONTEXT).answer == REFUSAL


def test_stream_yields_pieces():
    gen, _ = generator(chunks=["Нель", "зя.", None])
    assert "".join(gen.stream("вопрос", CONTEXT)) == "Нельзя."


def test_api_failure_is_an_error_not_an_exception():
    gen, _ = generator(error=RuntimeError("401 unauthorized"))
    out = gen.generate("вопрос", CONTEXT)
    assert not out.ok and "401" in out.error


def test_descriptor_names_the_vendor():
    gen, _ = generator(content="")
    assert gen.descriptor["backend"] == "deepseek"
    assert gen.descriptor["model"] == "deepseek-chat"
