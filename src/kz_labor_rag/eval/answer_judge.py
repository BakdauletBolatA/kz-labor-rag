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


class ClaudeAnswerJudge:
    def __init__(
        self,
        model: str,
        *,
        prompt_id: str = "answer_judge_ru",
        prompt_version: str = "v1",
        effort: str = "medium",
        client=None,
    ) -> None:
        self.model = model
        self.effort = effort
        self._prompt = load_prompt(prompt_id, prompt_version)
        self._client = client

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "prompt": self._prompt.label,
            "prompt_sha256": self._prompt.sha256,
            "effort": self.effort,
        }

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def judge(
        self,
        question: str,
        reference: str,
        context: Sequence[RetrievedChunk] | str,
        answer: str,
    ) -> AnswerVerdict:
        """``context`` — фрагменты или уже собранный ``format_context`` текст."""
        import anthropic

        prompt = self._prompt.text.format(
            question=question,
            reference=reference,
            context=context if isinstance(context, str) else format_context(context),
            answer=answer,
        )
        try:
            response = self._get_client().beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            return AnswerVerdict(None, None, error=f"{type(exc).__name__}: {exc}")
        if response.stop_reason == "refusal":
            return AnswerVerdict(None, None, error="судья отказался оценивать")
        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        return AnswerVerdict(data["correctness"], data["groundedness"], data["reasoning"])


def judge_available() -> str | None:
    """Почему судью нельзя запустить, или None."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return None
    return "ANTHROPIC_API_KEY не задан — судья не запускался, его метрики не измерены"


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
