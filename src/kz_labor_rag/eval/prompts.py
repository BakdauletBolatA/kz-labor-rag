"""Версионированные промпты генератора и судьи.

Требование проекта: промпт судьи и его версия лежат в репозитории и меняются
только вместе с бампом версии, иначе цифры между итерациями несравнимы.
Одного соглашения тут мало — его легко нарушить случайно, поправив формулировку
«просто чтобы было понятнее». Поэтому связка версия→содержимое зафиксирована
хешами в ``REGISTRY.json``: правка текста промпта без бампа версии обрушивает
прогон с явным сообщением, а не портит молча следующую строку EVALUATION.md.

Промпты генератора и судьи версионируются независимо друг от друга.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
REGISTRY_PATH = PROMPTS_DIR / "REGISTRY.json"


class PromptRegistryError(RuntimeError):
    """Промпт разошёлся с зафиксированной версией."""


@dataclass(frozen=True)
class Prompt:
    """Загруженный промпт вместе со всем, что нужно записать в результат."""

    prompt_id: str
    version: str
    text: str
    sha256: str

    @property
    def label(self) -> str:
        return f"{self.prompt_id}@{self.version}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_registry() -> dict[str, dict[str, str]]:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def load_prompt(prompt_id: str, version: str) -> Prompt:
    """Прочитать промпт и убедиться, что он не разошёлся с реестром."""
    path = PROMPTS_DIR / f"{prompt_id}.{version}.txt"
    if not path.exists():
        raise PromptRegistryError(
            f"промпт {prompt_id}@{version} не найден: ожидался файл {path}. "
            "Новая версия промпта — это новый файл, а не правка старого."
        )

    text = path.read_text(encoding="utf-8")
    digest = _sha256(text)
    registry = _load_registry()
    recorded = registry.get(prompt_id, {}).get(version)

    if recorded is None:
        raise PromptRegistryError(
            f"промпт {prompt_id}@{version} не зарегистрирован в {REGISTRY_PATH.name}. "
            f"Добавьте запись с хешем {digest}, иначе прогон невоспроизводим."
        )
    if recorded != digest:
        raise PromptRegistryError(
            f"промпт {prompt_id}@{version} изменён без бампа версии.\n"
            f"  в реестре: {recorded}\n"
            f"  на диске:  {digest}\n"
            "Метрики, посчитанные разными промптами, несравнимы. "
            f"Создайте {prompt_id}.<новая версия>.txt и укажите её в конфиге."
        )

    return Prompt(prompt_id=prompt_id, version=version, text=text, sha256=digest)


def register_prompt(prompt_id: str, version: str) -> str:
    """Внести существующий файл промпта в реестр. Перезапись версии запрещена."""
    path = PROMPTS_DIR / f"{prompt_id}.{version}.txt"
    if not path.exists():
        raise PromptRegistryError(f"нечего регистрировать: файл {path} не найден")

    digest = _sha256(path.read_text(encoding="utf-8"))
    registry = _load_registry()
    existing = registry.get(prompt_id, {}).get(version)
    if existing is not None and existing != digest:
        raise PromptRegistryError(
            f"{prompt_id}@{version} уже зарегистрирован с другим хешем. "
            "Версии неизменяемы — заведите новую."
        )

    registry.setdefault(prompt_id, {})[version] = digest
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest
