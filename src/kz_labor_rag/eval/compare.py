"""Сравнение двух прогонов eval.

Смысл проекта — таблица «до/после». Поэтому самое опасное, что может сделать
харнесс, — нарисовать такую таблицу для двух прогонов, которые сравнивать
нельзя. Числа будут выглядеть осмысленно, разница будет выглядеть как эффект
итерации, а на деле окажется следствием того, что под прогонами разный корпус,
разная нарезка или разный промпт судьи.

Случай не гипотетический. Исправление окна модели перенарезало корпус целиком:
из 143 чанков ни один не совпал со старым. Если бы baseline уже был посчитан,
его цифры молча превратились бы в мусор, а сравнение с ними продолжило бы
рисоваться. Спас только гейт готовности датасета.

Поэтому сравнение отказывается работать, пока все ключи сопоставимости не
совпали, и называет каждое расхождение поимённо.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Что обязано совпадать, чтобы два прогона можно было ставить в одну таблицу.
# Ключ — путь в блоке comparability, значение — как объяснить расхождение.
COMPARABILITY_KEYS: dict[str, str] = {
    "metrics_version": "изменились сами формулы метрик",
    "k": "метрики посчитаны на окне разного размера",
    "context_format": "модели показывали контекст в разном формате",
    "chunking_signature": "корпус нарезан по-другому — тексты чанков не те же",
    "embeddings_model": "векторы посчитаны другой моделью",
    "dataset_sha256": "прогоны сделаны на разных датасетах",
    "generator_prompt": "ответы генерировались разными промптами",
    "judge_prompt": "faithfulness измерена разными промптами судьи",
}

# Метрики для таблицы «до/после», в порядке показа. Имена вида recall@k
# зависят от k, поэтому берутся из самих прогонов: захардкоженное «recall@5»
# при k=10 не находило ничего, и главные метрики молча выпадали из таблицы,
# оставляя её выглядеть правдоподобно.
METRIC_ORDER: tuple[str, ...] = (
    "recall@",
    "strict_hit@",
    "mrr",
    "article_recall@",
    "citation_validity",
    "faithfulness",
)


def table_metrics(*slices: dict) -> list[str]:
    """Имена метрик, реально присутствующих в прогонах, в каноническом порядке."""
    keys = list(dict.fromkeys(key for part in slices for key in part))
    out: list[str] = []
    for name in METRIC_ORDER:
        out += [
            key for key in keys if (key.startswith(name) if name.endswith("@") else key == name)
        ]
    return out


class IncomparableRunsError(RuntimeError):
    """Прогоны нельзя ставить в одну таблицу."""


@dataclass(frozen=True)
class Mismatch:
    key: str
    reason: str
    before: Any
    after: Any

    def __str__(self) -> str:
        return f"{self.key}: {self.before!r} -> {self.after!r} ({self.reason})"


def comparability_of(result: dict) -> dict[str, Any]:
    """Достать блок сопоставимости, пережив результаты старого формата."""
    block = result.get("comparability")
    if isinstance(block, dict):
        return block
    # Прогоны, сделанные до появления блока, сравнивать не с чем: отсутствие
    # ключа честнее подставного значения.
    return {key: None for key in COMPARABILITY_KEYS}


def find_mismatches(before: dict, after: dict) -> list[Mismatch]:
    left, right = comparability_of(before), comparability_of(after)
    return [
        Mismatch(key=key, reason=reason, before=left.get(key), after=right.get(key))
        for key, reason in COMPARABILITY_KEYS.items()
        if left.get(key) != right.get(key)
    ]


def assert_comparable(before: dict, after: dict) -> None:
    """Отказаться сравнивать несопоставимые прогоны."""
    mismatches = find_mismatches(before, after)
    if not mismatches:
        return

    listed = "\n".join(f"  - {m}" for m in mismatches)
    raise IncomparableRunsError(
        f"Прогоны {before.get('version')} и {after.get('version')} несравнимы:\n"
        f"{listed}\n\n"
        "Таблица «до/после» по ним была бы вымыслом: разница в цифрах объяснялась бы "
        "не итерацией, а перечисленным выше.\n"
        "Нужен перепрогон baseline на текущей конфигурации, а не сравнение с архивом."
    )


def _slice(result: dict, language: str | None) -> dict[str, Any]:
    aggregates = result.get("aggregates", {})
    if language is None:
        return aggregates.get("primary", {})
    return aggregates.get("by_language", {}).get(language, {})


def build_table(before: dict, after: dict, *, language: str | None = None) -> str:
    """Собрать строки таблицы «до/после». Только для сопоставимых прогонов."""
    assert_comparable(before, after)

    left, right = _slice(before, language), _slice(after, language)
    header = (
        f"{'метрика':<22}{before.get('version', '?'):>16}{after.get('version', '?'):>16}"
        f"{'дельта':>12}"
    )
    lines = [header, "-" * len(header)]

    for metric in table_metrics(left, right):
        a, b = left.get(metric), right.get(metric)
        if a is None and b is None:
            continue
        cell_a = "—" if a is None else f"{a:.3f}"
        cell_b = "—" if b is None else f"{b:.3f}"
        delta = "—" if a is None or b is None else f"{b - a:+.3f}"
        lines.append(f"{metric:<22}{cell_a:>16}{cell_b:>16}{delta:>12}")

    was = set(left.get("retrieval_failures", []))
    now = set(right.get("retrieval_failures", []))
    fixed, broken = sorted(was - now), sorted(now - was)
    lines.append("")
    lines.append(f"починилось вопросов: {len(fixed)}  {', '.join(fixed[:10])}")
    lines.append(f"сломалось вопросов : {len(broken)}  {', '.join(broken[:10])}")
    return "\n".join(lines)
