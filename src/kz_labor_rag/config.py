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

# По этому файлу опознаётся корень репозитория.
ROOT_MARKER = "pyproject.toml"


def _walk_up(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ROOT_MARKER).exists():
            return candidate
    return None


def find_repo_root(start: Path | None = None) -> Path | None:
    """Найти корень репозитория.

    Сначала вверх от текущего каталога — это покрывает запуск из любого места
    внутри репозитория. Если не нашлось, пробуем от каталога установленного
    пакета: при установке через ``pip install -e`` он лежит в ``src/`` внутри
    того же репозитория, и это единственная зацепка, когда команду запускают
    вообще из другого места — например, из домашнего каталога, откуда вверх
    подниматься некуда.

    При обычной установке в site-packages ``pyproject.toml`` выше пакета нет,
    поиск вернёт ``None``, и поведение останется прежним — путями от текущего
    каталога. В Docker это и нужно: рабочий каталог там ``/app``.
    """
    if found := _walk_up((start or Path.cwd()).resolve()):
        return found
    if start is not None:
        return None
    return _walk_up(Path(__file__).resolve().parent)

# Значения, которые допустимо переопределить переменной окружения.
# Всё остальное меняется только правкой YAML — иначе прогон невоспроизводим.
ENV_OVERRIDES: dict[str, str] = {
    "KZRAG_VERSION": "version",
    "KZRAG_DATABASE_URL": "vector_store.dsn",
    "KZRAG_EMBEDDINGS_PROVIDER": "embeddings.provider",
    "KZRAG_EMBEDDINGS_MODEL": "embeddings.model",
    "KZRAG_EVAL_DATASET": "eval.dataset",
    "KZRAG_DEVICE": "embeddings.device",
    # В Docker Ollama живёт в соседнем контейнере, локально — на хосте.
    "KZRAG_OLLAMA_URL": "generation.base_url",
    # Та же переменная говорит compose, какую модель скачать.
    "KZRAG_GENERATION_MODEL": "generation.model",
}


class ConfigError(KeyError):
    """В конфиге нет запрошенного параметра."""


@dataclass(frozen=True)
class Config:
    """Обёртка над словарём конфига с доступом по точечному пути."""

    data: dict[str, Any]
    path: Path | None = None
    # Каталог, относительно которого разрешаются относительные пути из конфига.
    root: Path | None = None

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

    def path_of(self, dotted: str) -> Path:
        """Значение конфига как путь, разрешённый от корня репозитория.

        В конфиге пути записаны относительными — так их удобно читать и они
        одинаковы в Docker и локально. Но разрешать их относительно текущего
        каталога значит требовать запускать всё только из корня.
        """
        value = Path(str(self.get(dotted)))
        if value.is_absolute() or self.root is None:
            return value
        return self.root / value

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
    requested = Path(path or os.environ.get("KZRAG_CONFIG", DEFAULT_CONFIG_PATH))
    root = find_repo_root()

    path = requested
    if not path.exists() and not path.is_absolute() and root is not None:
        # Запуск не из корня репозитория — ищем конфиг от корня.
        path = root / requested
    if not path.exists():
        raise ConfigError(
            f"конфиг не найден: {requested}"
            + (f" (искал также в {root})" if root is not None else "")
        )

    # Пути внутри конфига разрешаются от корня репозитория, а если корень не
    # опознан — от каталога конфига, чтобы поведение оставалось предсказуемым.
    root = root or path.resolve().parent.parent

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"конфиг {path} должен быть словарём верхнего уровня")

    if apply_env:
        # Значение берётся строкой как есть, без разбора YAML. Все
        # переопределений строковые, а разбор превращал безобидные метки в
        # другие типы: KZRAG_VERSION=1.0 становился числом, а
        # KZRAG_VERSION=2026-08-13 — датой, на которой падал уже отпечаток
        # конфига (date не сериализуется в JSON). То есть попытка пометить
        # прогон датой роняла прогон, и по сообщению об ошибке связь с
        # переменной окружения не читалась.
        for env_name, dotted in ENV_OVERRIDES.items():
            if (raw := os.environ.get(env_name)) is not None:
                _set_dotted(data, dotted, raw)

    return Config(data=data, path=path, root=root)
