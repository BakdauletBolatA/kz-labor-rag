"""Судья ответов и согласие с ручной разметкой. Сеть не нужна: клиент подменяется."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import make_chunk

from kz_labor_rag.eval.answer_judge import (
    ClaudeAnswerJudge,
    cohen_kappa,
    judge_available,
)
from kz_labor_rag.types import RetrievedChunk

CONTEXT = [RetrievedChunk(chunk=make_chunk("54", ("1",), cid="c1"), score=0.9, rank=1)]


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


def judge_with(payload, stop_reason="end_turn"):
    messages = FakeMessages(payload, stop_reason)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeAnswerJudge("claude-opus-5-5", client=client), messages


def test_verdict_is_parsed_from_structured_output():
    judge, messages = judge_with(
        {"reasoning": "совпадает", "correctness": "correct", "groundedness": "grounded"}
    )
    verdict = judge.judge("вопрос", "ст. 54 п. 1: норма", CONTEXT, "Нельзя.")
    assert (verdict.correctness, verdict.groundedness) == ("correct", "grounded")
    call = messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "ст. 54 п. 1: норма" in call["messages"][0]["content"]


def test_refusal_is_an_error_not_a_score():
    judge, _ = judge_with({}, stop_reason="refusal")
    verdict = judge.judge("вопрос", "эталон", CONTEXT, "ответ")
    assert not verdict.ok
    assert verdict.correctness is None


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert "ANTHROPIC_API_KEY" in judge_available()


class TestKappa:
    def test_perfect_agreement(self):
        assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0

    def test_chance_level_agreement_is_zero(self):
        # Совпадения ровно столько, сколько ожидается случайно.
        assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)

    def test_known_value(self):
        a = ["yes"] * 20 + ["no"] * 30
        b = ["yes"] * 15 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 20
        # p_o = (15 + 20) / 50 = 0.7; p_e = (20·25 + 30·25) / 50² = 0.5; κ = 0.2 / 0.5 = 0.4
        assert cohen_kappa(a, b) == pytest.approx(0.4)

    def test_constant_labels_have_no_kappa(self):
        assert cohen_kappa(["a", "a"], ["a", "a"]) is None

    def test_empty(self):
        assert cohen_kappa([], []) is None
