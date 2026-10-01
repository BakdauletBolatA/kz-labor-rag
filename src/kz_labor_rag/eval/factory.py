"""Сборка генератора и судьи по конфигу.

Ключи API читаются только отсюда и только из переменных окружения. Главное
правило: отсутствие ключа — это не ошибка. Пайплайн обязан пройти целиком,
посчитать recall@5, MRR и article_recall@5 и честно записать faithfulness как
``null``. Падать здесь значит терять все метрики поиска из-за компонента,
который к поиску отношения не имеет.
"""

from __future__ import annotations

import logging
import os

from kz_labor_rag.config import Config
from kz_labor_rag.eval.generator import AnthropicGenerator, DisabledGenerator, Generator
from kz_labor_rag.eval.judge import AnthropicJudge, DisabledJudge, Judge

log = logging.getLogger(__name__)

# Какой ключ нужен какому провайдеру.
PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def missing_key_env(provider: str) -> str | None:
    """Вернуть имя незаданной переменной окружения или None, если ключ есть."""
    env_name = PROVIDER_KEY_ENV.get(provider)
    if env_name is None:
        return None
    return None if os.environ.get(env_name) else env_name


def build_generator(config: Config) -> Generator:
    if not config.get("generation.enabled"):
        return DisabledGenerator("генерация отключена в конфиге")

    provider = config.get("generation.provider")
    if provider != "anthropic":
        return DisabledGenerator(f"провайдер генерации '{provider}' пока не поддержан")

    if env_name := missing_key_env(provider):
        log.warning(
            "%s не задан — генерация ответов отключена. Метрики поиска "
            "(recall@k, MRR, clause-метрики) считаются как обычно, "
            "faithfulness и citation_validity будут null. "
            "Чтобы включить: скопируйте .env.example в .env и заполните %s.",
            env_name,
            env_name,
        )
        return DisabledGenerator(f"{env_name} не задан")

    return AnthropicGenerator(
        model=config.get("generation.model"),
        prompt_id=config.get("generation.prompt_id"),
        prompt_version=config.get("generation.prompt_version"),
        max_tokens=int(config.get("generation.max_tokens")),
        temperature=float(config.get("generation.temperature")),
    )


def build_judge(config: Config, *, generator: Generator | None = None) -> Judge:
    """Собрать судью.

    Если генератор отключён, судья отключается вместе с ним: судить нечего, а
    шестьдесят вызовов API впустую — плохой способ это обнаружить.
    """
    if generator is not None and isinstance(generator, DisabledGenerator):
        return DisabledJudge("генератор отключён, судить нечего")

    if not config.get("judge.enabled"):
        return DisabledJudge("судья отключён в конфиге")

    provider = config.get("judge.provider")
    if provider != "anthropic":
        return DisabledJudge(f"провайдер судьи '{provider}' пока не поддержан")

    if env_name := missing_key_env(provider):
        log.warning(
            "%s не задан — судья faithfulness отключён. Метрики поиска "
            "считаются как обычно, faithfulness будет null.",
            env_name,
        )
        return DisabledJudge(f"{env_name} не задан")

    return AnthropicJudge(
        model=config.get("judge.model"),
        prompt_id=config.get("judge.prompt_id"),
        prompt_version=config.get("judge.prompt_version"),
        max_tokens=int(config.get("judge.max_tokens")),
        temperature=float(config.get("judge.temperature")),
    )
