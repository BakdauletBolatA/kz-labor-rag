"""LLM-судья ответов: правильность по эталону и обоснованность по контексту.

Судья другого вендора и заметно сильнее генератора (локальная 7B-модель):
модель, оценивающая собственный класс ответов, к ним снисходительна. Насколько
судье можно верить, проверяется отдельно — согласием с ручной разметкой
(``cohen_kappa``), а не верой.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from kz_labor_rag.corpus.parser import LaborCode
from kz_labor_rag.eval.citations import cite
from kz_labor_rag.eval.dataset import EvalQuestion
from kz_labor_rag.eval.judge import format_context
from kz_labor_rag.eval.prompts import load_prompt
from kz_labor_rag.types import RetrievedChunk

CORRECTNESS = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
GROUNDEDNESS = {"grounded": 1.0, "partially_grounded": 0.5, "ungrounded": 0.0}

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correctness": {"type": "string", "enum": list(CORRECTNESS)},
        "groundedness": {"type": "string", "enum": list(GROUNDEDNESS)},
    },
    "required": ["reasoning", "correctness", "groundedness"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AnswerVerdict:
    correctness: str | None
    groundedness: str | None
    reasoning: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def reference_for(question: EvalQuestion, code: LaborCode) -> str:
    """Эталон для судьи: полный текст обязательных пунктов либо «ответа нет»."""
    if question.is_unanswerable:
        return f"В Трудовом кодексе ответа на этот вопрос нет. {question.notes}"
    by_number = code.by_number
    lines = []
    for ref in question.required_clauses:
        clauses = {c.number: c.text for c in by_number[ref.article].clauses}
        lines.append(f"{cite(ref.article, ref.clause)}: {clauses[ref.clause]}")
    return "\n".join(lines)


class _PromptedJudge:
    def __init__(self, model: str, prompt_id: str, prompt_version: str, client) -> None:
        self.model = model
        self._prompt = load_prompt(prompt_id, prompt_version)
        self._client = client

    def _render(self, question, reference, context, answer) -> str:
        return self._prompt.text.format(
            question=question,
            reference=reference,
            context=context if isinstance(context, str) else format_context(context),
            answer=answer,
        )

    @staticmethod
    def _parse(text: str) -> AnswerVerdict:
        data = json.loads(text)
        return AnswerVerdict(data["correctness"], data["groundedness"], data["reasoning"])


class ClaudeAnswerJudge(_PromptedJudge):
    def __init__(
        self,
        model: str,
        *,
        prompt_id: str = "answer_judge_ru",
        prompt_version: str = "v1",
        effort: str = "medium",
        client=None,
    ) -> None:
        super().__init__(model, prompt_id, prompt_version, client)
        self.effort = effort

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "prompt": self._prompt.label,
            "prompt_sha256": self._prompt.sha256,
            "effort": self.effort,
        }

    def judge(
        self,
        question: str,
        reference: str,
        context: Sequence[RetrievedChunk] | str,
        answer: str,
    ) -> AnswerVerdict:
        """``context`` — фрагменты или уже собранный ``format_context`` текст."""
        import anthropic

        client = self._client or anthropic.Anthropic()
        try:
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                messages=[
                    {"role": "user", "content": self._render(question, reference, context, answer)}
                ],
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            return AnswerVerdict(None, None, error=f"{type(exc).__name__}: {exc}")
        if response.stop_reason == "refusal":
            return AnswerVerdict(None, None, error="судья отказался оценивать")
        return self._parse(next((b.text for b in response.content if b.type == "text"), ""))


class OpenAIAnswerJudge(_PromptedJudge):
    """Тот же промпт и та же схема ответа, другой вендор."""

    def __init__(
        self,
        model: str,
        *,
        prompt_id: str = "answer_judge_ru",
        prompt_version: str = "v1",
        client=None,
    ) -> None:
        super().__init__(model, prompt_id, prompt_version, client)

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "backend": "openai",
            "model": self.model,
            "prompt": self._prompt.label,
            "prompt_sha256": self._prompt.sha256,
        }

    def judge(
        self,
        question: str,
        reference: str,
        context: Sequence[RetrievedChunk] | str,
        answer: str,
    ) -> AnswerVerdict:
        import openai

        client = self._client or openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self._render(question, reference, context, answer)}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "answer_verdict", "schema": SCHEMA, "strict": True},
                },
            )
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
            return AnswerVerdict(None, None, error=f"{type(exc).__name__}: {exc}")
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            return AnswerVerdict(None, None, error=f"судья отказался оценивать: {message.refusal}")
        return self._parse(message.content or "")


KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
}


def judge_available(provider: str = "anthropic") -> str | None:
    """Почему судью нельзя запустить, или None."""
    if provider not in KEY_ENV:
        return f"неизвестный провайдер судьи '{provider}'"
    if any(os.environ.get(name) for name in KEY_ENV[provider]):
        return None
    return f"{KEY_ENV[provider][0]} не задан — судья не запускался, его метрики не измерены"


def build_answer_judge(config) -> tuple[object | None, str | None]:
    """Судья по секции ``answer_judge`` конфига, либо причина, почему его нет."""
    provider = config.get("answer_judge.provider")
    if reason := judge_available(provider):
        return None, reason
    common = {
        "prompt_id": config.get("answer_judge.prompt_id"),
        "prompt_version": config.get("answer_judge.prompt_version"),
    }
    if provider == "openai":
        return OpenAIAnswerJudge(config.get("answer_judge.openai_model"), **common), None
    return (
        ClaudeAnswerJudge(
            config.get("answer_judge.anthropic_model"),
            effort=config.get("answer_judge.effort"),
            **common,
        ),
        None,
    )


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Согласие двух разметчиков с поправкой на случайное совпадение.

    1 — полное согласие, 0 — не лучше случайного. ``None``, если считать не по
    чему или оба разметчика всегда ставили одну и ту же метку (знаменатель 0).
    """
    if len(a) != len(b):
        raise ValueError("разметки разной длины")
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(a) | set(b)) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)
