"""Генератор ответа за интерфейсом.

Симметричен ``judge.py``: модель и промпт задаются конфигом, версия промпта
пишется в результат прогона. Промпт генератора версионируется независимо от
промпта судьи — они меняются по разным причинам и в разное время.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from kz_labor_rag.eval.judge import format_context
from kz_labor_rag.eval.prompts import Prompt, load_prompt
from kz_labor_rag.types import RetrievedChunk, normalize_article

# Ссылки вида «ст. 52», «статья 52», «(ст. 52 п. 1)».
_ARTICLE_CITATION = re.compile(
    # Составные номера («ст. 73-1») обязаны ловиться целиком: иначе
    # citation_validity примет ссылку на 73-1 за ссылку на 73.
    r"\b(?:ст\.?|стать[ияеёю]м?и?)\s*(\d{1,3}(?:-\d{1,2})?)", re.IGNORECASE
)


def extract_cited_articles(answer: str) -> list[str]:
    """Вытащить номера статей, на которые сослался генератор.

    Нужно для ``citation_validity`` — детерминированной проверки без LLM.
    Порядок сохраняется, дубли убираются.
    """
    seen: set[str] = set()
    out: list[str] = []
    for match in _ARTICLE_CITATION.finditer(answer):
        num = normalize_article(match.group(1))
        if num not in seen:
            seen.add(num)
            out.append(num)
    return out


class GenerationError(RuntimeError):
    """Генератор не смог выдать ответ."""


@dataclass(frozen=True)
class Generation:
    """Сгенерированный ответ плюс всё, что нужно для воспроизведения строки."""

    answer: str
    cited_articles: tuple[str, ...] = ()
    backend: str = ""
    model: str = ""
    prompt_label: str = ""
    prompt_sha256: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Generator(Protocol):
    @property
    def descriptor(self) -> dict[str, str]:
        ...

    def generate(self, question: str, context: Sequence[RetrievedChunk]) -> Generation:
        ...


@dataclass
class AnthropicGenerator:
    """Генератор на Claude. Дефолт — Sonnet 5, вендор совпадает с судьёй,
    но модель другая: см. README, раздел про разделение ролей."""

    model: str = "claude-sonnet-5"
    prompt_id: str = "answer_ru"
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
                raise GenerationError(
                    f"{self.api_key_env} не задан. Положите ключ в .env "
                    "либо выключите generation в конфиге."
                )
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise GenerationError(
                    "пакет anthropic не установлен: pip install anthropic"
                ) from exc
            self._client = Anthropic(api_key=os.environ[self.api_key_env])
        return self._client

    def generate(self, question: str, context: Sequence[RetrievedChunk]) -> Generation:
        prompt = self._prompt.text.format(question=question, context=format_context(context))
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
        except GenerationError as exc:
            return Generation(answer="", error=str(exc), **base)
        except Exception as exc:  # noqa: BLE001 — один упавший вопрос не роняет прогон
            return Generation(answer="", error=f"{type(exc).__name__}: {exc}", **base)

        answer = "".join(block.text for block in response.content if block.type == "text").strip()
        return Generation(
            answer=answer,
            cited_articles=tuple(extract_cited_articles(answer)),
            input_tokens=getattr(response.usage, "input_tokens", None),
            output_tokens=getattr(response.usage, "output_tokens", None),
            **base,
        )


@dataclass
class DisabledGenerator:
    """Заглушка, когда генерация выключена: считаем только метрики поиска."""

    reason: str = "генерация отключена в конфиге"

    @property
    def descriptor(self) -> dict[str, str]:
        return {"backend": "disabled", "reason": self.reason}

    def generate(self, question: str, context: Sequence[RetrievedChunk]) -> Generation:
        return Generation(answer="", backend="disabled", error=self.reason)
