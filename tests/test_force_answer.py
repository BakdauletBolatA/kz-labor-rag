"""Повторный запрос с запретом отказа, когда лучший фрагмент набрал высокий скор."""

from __future__ import annotations

from dataclasses import replace

from conftest import make_chunk

from kz_labor_rag.eval.citations import REFUSAL
from kz_labor_rag.eval.generator import ForceAnswerGenerator, Generation
from kz_labor_rag.types import RetrievedChunk


def hits(score):
    return [RetrievedChunk(chunk=make_chunk("54", ("1",), cid="c1"), score=score, rank=1)]


class Scripted:
    def __init__(self, answer, refused=False, prompt="p@v"):
        self.out = Generation(answer=answer, refused=refused, backend="b", prompt_label=prompt)
        self.calls = 0
        self.descriptor = {"backend": "b", "prompt": prompt}

    def generate(self, question, context):
        self.calls += 1
        return self.out


def wrapper(inner, forced, tau=0.3):
    return ForceAnswerGenerator(inner=inner, forced=forced, threshold=tau)


def test_refusal_with_a_confident_best_fragment_is_asked_again():
    inner, forced = Scripted(REFUSAL, refused=True), Scripted("Нельзя.", prompt="force@v1")
    out = wrapper(inner, forced).generate("q", hits(0.9))
    assert out.answer == "Нельзя." and out.forced
    assert forced.calls == 1


def test_refusal_with_a_weak_best_fragment_stays_a_refusal():
    inner, forced = Scripted(REFUSAL, refused=True), Scripted("Нельзя.")
    out = wrapper(inner, forced).generate("q", hits(0.1))
    assert out.refused and not out.forced
    assert forced.calls == 0


def test_a_normal_answer_is_never_touched():
    inner, forced = Scripted("Можно.", refused=False), Scripted("Нельзя.")
    out = wrapper(inner, forced).generate("q", hits(0.99))
    assert out.answer == "Можно." and forced.calls == 0


def test_if_the_forced_answer_also_refuses_the_original_is_kept():
    inner, forced = Scripted(REFUSAL, refused=True), Scripted(REFUSAL, refused=True)
    out = wrapper(inner, forced).generate("q", hits(0.9))
    assert out.refused and not out.forced


def test_descriptor_records_the_rule():
    d = wrapper(Scripted("x"), Scripted("y", prompt="force@v1"), tau=0.28).descriptor
    assert d["force_answer_above"] == "0.28" and d["forced_prompt"] == "force@v1"


def test_failed_generation_is_not_retried():
    inner = Scripted("")
    inner.out = replace(inner.out, error="HTTP 500", refused=True)
    forced = Scripted("Нельзя.")
    out = wrapper(inner, forced).generate("q", hits(0.9))
    assert not out.ok and forced.calls == 0
