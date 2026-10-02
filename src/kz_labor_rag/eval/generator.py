"""Генератор ответа за интерфейсом.

Симметричен ``judge.py``: модель и промпт задаются конфигом, версия промпта
пишется в результат прогона. Промпт генератора версионируется независимо от
промпта судьи — они меняются по разным причинам и в разное время.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kz_labor_rag.eval.citations import ground
from kz_labor_rag.eval.judge import format_context
from kz_labor_rag.eval.prompts import Prompt, load_prompt
from kz_labor_rag.types import RetrievedChunk, normalize_article

# Номер статьи: составные («73-1») обязаны ловиться целиком, иначе
# citation_validity примет ссылку на 73-1 за ссылку на 73.
_NUMBER = r"\d{1,3}(?:-\d{1,2})?"

# Начало ссылки: «ст. 52», «статья 52», «(ст. 52 п. 1)».
_CITATION_HEAD = re.compile(rf"\b(?:ст\.?|стать[ияеёю]м?и?)\s*({_NUMBER})", re.IGNORECASE)

# Продолжение перечисления: «, 53», « и 54». Ключевое слово в перечислении не
# повторяется, и без этого «согласно статьям 52 и 53» давало одну статью из
# двух. Связок норм в датасете семь, то есть теряется самый частый способ
# сослаться сразу на несколько статей, а выдуманная вторая ссылка становится
# для citation_validity невидимой.
_CITATION_MORE = re.compile(rf"\s*(?:,|и)\s*({_NUMBER})")


def extract_cited_articles(answer: str) -> list[str]:
    """Вытащить номера статей, на которые сослался генератор.

    Нужно для ``citation_validity`` — детерминированной проверки без LLM.
    Порядок сохраняется, дубли убираются.

    Перечисление обрывается на первом же не-номере, поэтому «ст. 52 п. 1»
    не превращает пункт в статью. Обратная сторона: «ст. 62 и 5 дней» отдаст
    несуществующую ссылку на ст. 5. Это осознанный размен — пропущенная ссылка
    прячет галлюцинацию, лишняя её в худшем случае преувеличивает.
    """
    seen: set[str] = set()
    out: list[str] = []

    def take(raw: str) -> None:
        num = normalize_article(raw)
        if num not in seen:
            seen.add(num)
            out.append(num)

    for match in _CITATION_HEAD.finditer(answer):
        take(match.group(1))
        pos = match.end()
        while tail := _CITATION_MORE.match(answer, pos):
            take(tail.group(1))
            pos = tail.end()
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
    # Пункты, на которые ответ ссылается и которые есть в показанном контексте.
    citations: tuple[str, ...] = ()
    # Ссылки на пункты, которых модель не видела.
    invalid_citations: tuple[str, ...] = ()
    refused: bool = False
    # Модель ответила без единой верной ссылки, и ответ заменён отказом.
    withheld: bool = False
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Generator(Protocol):
    @property
    def descriptor(self) -> dict[str, str]: ...

    def generate(self, question: str, context: Sequence[RetrievedChunk]) -> Generation: ...


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
class OllamaGenerator:
    """Генератор на локальной модели через HTTP API Ollama.

    Ответ проходит ``ground``: ссылки сверяются с показанными фрагментами, и
    ответ без единой верной ссылки заменяется отказом. ``cited_articles``
    берутся из ссылок модели до этой проверки, поэтому ``citation_validity``
    показывает, как часто модель ссылается на то, чего не видела.
    """

    model: str
    base_url: str
    prompt_id: str = "answer_ru"
    prompt_version: str = "v2"
    max_tokens: int = 512
    temperature: float = 0.0
    num_ctx: int = 4096
    seed: int = 0
    timeout: float = 600.0
    _prompt: Prompt = field(init=False)

    def __post_init__(self) -> None:
        self._prompt = load_prompt(self.prompt_id, self.prompt_version)

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "backend": "ollama",
            "model": self.model,
            "prompt": self._prompt.label,
            "prompt_sha256": self._prompt.sha256,
            "temperature": str(self.temperature),
        }

    def _request(self, question: str, context: Sequence[RetrievedChunk], stream: bool):
        prompt = self._prompt.text.format(question=question, context=format_context(context))
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
                "seed": self.seed,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=self.timeout)

    def stream(self, question: str, context: Sequence[RetrievedChunk]) -> Iterator[str]:
        """Куски ответа по мере генерации. Проверку ссылок делает вызывающий."""
        with self._request(question, context, stream=True) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line)
                if piece := event.get("message", {}).get("content"):
                    yield piece
                if event.get("done"):
                    break

    def finish(self, raw: str, context: Sequence[RetrievedChunk], **usage) -> Generation:
        grounded = ground(raw, context)
        return Generation(
            answer=grounded.text,
            cited_articles=tuple(
                dict.fromkeys(c.article for c in (*grounded.citations, *grounded.invalid_citations))
            ),
            citations=tuple(str(c) for c in grounded.citations),
            invalid_citations=tuple(str(c) for c in grounded.invalid_citations),
            refused=grounded.refused,
            withheld=grounded.withheld,
            raw=raw,
            backend="ollama",
            model=self.model,
            prompt_label=self._prompt.label,
            prompt_sha256=self._prompt.sha256,
            **usage,
        )

    def generate(self, question: str, context: Sequence[RetrievedChunk]) -> Generation:
        try:
            with self._request(question, context, stream=False) as response:
                body = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 — один упавший вопрос не роняет прогон
            return Generation(
                answer="",
                backend="ollama",
                model=self.model,
                prompt_label=self._prompt.label,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self.finish(
            body.get("message", {}).get("content", "").strip(),
            context,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
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
