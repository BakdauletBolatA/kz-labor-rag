"""Судья faithfulness за интерфейсом.

Вендор и модель судьи намеренно отличаются от вендора и модели генератора:
модель, оценивающая собственный вывод, систематически к нему снисходительна.
Смена бэкенда судьи не должна требовать правок в харнессе, поэтому здесь
только протокол и реализации.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from kz_labor_rag.eval.prompts import Prompt, load_prompt
from kz_labor_rag.types import RetrievedChunk

# Вердикт судьи → численная faithfulness. Отображение зафиксировано здесь и
# входит в metrics_version: поменять его — значит сделать все прошлые цифры
# несравнимыми с новыми.
VERDICT_SCORES: dict[str, float] = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
}


class JudgeError(RuntimeError):
    """Судья не смог вынести оценку."""


@dataclass(frozen=True)
class Judgement:
    """Результат оценки одного ответа."""

    score: float
    verdict: str
    reasoning: str = ""
    unsupported_claims: tuple[str, ...] = ()
    raw_response: str = ""
    backend: str = ""
    model: str = ""
    prompt_label: str = ""
    prompt_sha256: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Judge(Protocol):
    """Всё, что харнесс знает о судье."""

    @property
    def descriptor(self) -> dict[str, str]:
        """Что записать в результат прогона: бэкенд, модель, версия промпта."""

    def judge(self, question: str, context: Sequence[RetrievedChunk], answer: str) -> Judgement:
        ...


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Собрать фрагменты в текст промпта.

    Формат общий для генератора и судьи: судья обязан видеть ровно тот
    контекст, который видел генератор, иначе оценка обоснованности бессмысленна.
    """
    parts = []
    for chunk in chunks:
        header = f"[Статья {chunk.article}"
        if title := chunk.chunk.article_title:
            header += f". {title}"
        header += "]"
        parts.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict:
    """Достать JSON из ответа модели, пережив markdown-обёртку."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.S)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise JudgeError(f"в ответе судьи нет JSON: {text[:400]}") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise JudgeError(f"невалидный JSON от судьи: {exc}; ответ: {text[:400]}") from exc


@dataclass
class AnthropicJudge:
    """Судья на Claude. Дефолт — Haiku 4.5."""

    model: str = "claude-haiku-4-5-20251001"
    prompt_id: str = "faithfulness_ru"
    prompt_version: str = "v1"
    max_tokens: int = 1024
    temperature: float = 0.0
    api_key_env: str = "ANTHROPIC_API_KEY"
    _prompt: Prompt = field(init=False)
    _client: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._prompt = load_prompt(self.prompt_id, self.prompt_version)

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "prompt": self._prompt.label,
            "prompt_sha256": self._prompt.sha256,
            "temperature": str(self.temperature),
        }

    def _get_client(self):
        if self._client is None:
            if not os.environ.get(self.api_key_env):
                raise JudgeError(
                    f"{self.api_key_env} не задан. Положите ключ в .env "
                    "(в чат его присылать не нужно) либо отключите judge в конфиге — "
                    "тогда посчитаются только метрики поиска."
                )
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise JudgeError("пакет anthropic не установлен: pip install anthropic") from exc
            self._client = Anthropic(api_key=os.environ[self.api_key_env])
        return self._client

    def judge(self, question: str, context: Sequence[RetrievedChunk], answer: str) -> Judgement:
        prompt = self._prompt.text.format(
            question=question, context=format_context(context), answer=answer
        )
        base = {
            "backend": "anthropic",
            "model": self.model,
            "prompt_label": self._prompt.label,
            "prompt_sha256": self._prompt.sha256,
        }
        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(block.text for block in response.content if block.type == "text")
            parsed = _extract_json(raw)
        except JudgeError as exc:
            return Judgement(score=0.0, verdict="error", error=str(exc), **base)
        except Exception as exc:  # noqa: BLE001 — падение одного вопроса не должно ронять прогон
            return Judgement(
                score=0.0, verdict="error", error=f"{type(exc).__name__}: {exc}", **base
            )

        verdict = str(parsed.get("verdict", "")).strip()
        if verdict not in VERDICT_SCORES:
            return Judgement(
                score=0.0,
                verdict="error",
                error=f"неизвестный verdict: {verdict!r}",
                raw_response=raw,
                **base,
            )

        unsupported = tuple(
            str(c.get("claim", ""))
            for c in parsed.get("claims", [])
            if isinstance(c, dict) and not c.get("supported", False)
        )
        return Judgement(
            score=VERDICT_SCORES[verdict],
            verdict=verdict,
            reasoning=str(parsed.get("reasoning", "")),
            unsupported_claims=unsupported,
            raw_response=raw,
            **base,
        )


@dataclass
class DisabledJudge:
    """Заглушка на случай, когда judge выключен или ключа нет.

    Возвращает ``None``-подобную оценку через ``error``, а не ноль: ноль — это
    «ответ не обоснован», а отсутствие судьи — «мы не измеряли». Смешивать эти
    два случая в агрегате нельзя, иначе faithfulness провалится по причине,
    не имеющей отношения к качеству системы.
    """

    reason: str = "судья отключён в конфиге"

    @property
    def descriptor(self) -> dict[str, str]:
        return {"backend": "disabled", "reason": self.reason}

    def judge(self, question: str, context: Sequence[RetrievedChunk], answer: str) -> Judgement:
        return Judgement(score=0.0, verdict="skipped", error=self.reason, backend="disabled")
