"""Загрузка конфига.

Требование проекта: все параметры чанкинга и поиска живут в YAML, а не в коде.
Поэтому здесь нет значений по умолчанию — отсутствие ключа в конфиге это
ошибка, а не повод молча подставить «разумное» значение. Тихий дефолт в коде
означает, что строка в EVALUATION.md не восстанавливается из конфига, а это
ровно то, ради чего весь проект и затевался.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

# Значения, которые допустимо переопределить переменной окружения.
# Всё остальное меняется только правкой YAML — иначе прогон невоспроизводим.
ENV_OVERRIDES: dict[str, str] = {
    "KZRAG_VERSION": "version",
    "KZRAG_DATABASE_URL": "vector_store.dsn",
    "KZRAG_EMBEDDINGS_PROVIDER": "embeddings.provider",
    "KZRAG_EMBEDDINGS_MODEL": "embeddings.model",
    "KZRAG_EVAL_DATASET": "eval.dataset",
    "KZRAG_DEVICE": "embeddings.device",
}


class ConfigError(KeyError):
    """В конфиге нет запрошенного параметра."""


@dataclass(frozen=True)
class Config:
    """Обёртка над словарём конфига с доступом по точечному пути."""

    data: dict[str, Any]
    path: Path | None = None

    def get(self, dotted: str) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(
                    f"параметр '{dotted}' отсутствует в конфиге"
                    + (f" {self.path}" if self.path else "")
                    + ". Значений по умолчанию в коде нет намеренно: добавьте параметр в YAML."
                )
            node = node[part]
        return node

    def get_or(self, dotted: str, default: Any) -> Any:
        """Для необязательных параметров — только там, где отсутствие ключа
        действительно означает «выключено», а не «забыли»."""
        try:
            return self.get(dotted)
        except ConfigError:
            return default

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"'{name}' в конфиге не секция, а {type(value).__name__}")
        return value

    @property
    def version(self) -> str:
        return str(self.get("version"))

    # Подписи чанкинга здесь намеренно нет. Она была — и считалась по секции
    # chunking конфига, то есть НЕ менялась при смене версии чанкера. Именно
    # такой случай и произошёл: исправление окна модели перенарезало корпус
    # целиком, а эта подпись осталась прежней. Единственный источник истины —
    # chunker.chunking_signature, и в результат прогона она попадает из
    # index_meta, то есть из того индекса, на котором поиск реально работал.

    @property
    def fingerprint(self) -> str:
        """Отпечаток всего конфига. Пишется в результат прогона: по нему видно,
        что две строки EVALUATION.md действительно получены разными настройками,
        а не одинаковыми с разными подписями."""
        payload = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, *, apply_env: bool = True) -> Config:
    """Прочитать YAML и наложить разрешённые переменные окружения."""
    path = Path(path or os.environ.get("KZRAG_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise ConfigError(f"конфиг не найден: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"конфиг {path} должен быть словарём верхнего уровня")

    if apply_env:
        for env_name, dotted in ENV_OVERRIDES.items():
            if (raw := os.environ.get(env_name)) is not None:
                _set_dotted(data, dotted, yaml.safe_load(raw))

    return Config(data=data, path=path)
