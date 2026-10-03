"""Доверительные интервалы для метрик на небольшой выборке вопросов.

Bootstrap по вопросам: вопросы выбираются с возвращением, метрика
пересчитывается, и 2.5-й и 97.5-й перцентили дают 95% интервал. Формулы для
нормального распределения здесь не годятся: метрики ограничены отрезком
[0, 1] и на 72 вопросах далеки от нормальности.

Для сравнения двух конфигураций — парный bootstrap: в каждой выборке берутся
одни и те же вопросы для обеих, и считается разница. Это мощнее, чем сравнивать
два отдельных интервала: трудность вопроса одинакова для обеих конфигураций и
из разницы вычитается.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 10_000


def _means(values: np.ndarray, seed: int, resamples: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(
    values: Sequence[float], *, seed: int, resamples: int = RESAMPLES, level: float = 0.95
) -> tuple[float, float] | None:
    """95% интервал среднего; None на пустой выборке."""
    if not len(values):
        return None
    means = _means(np.asarray(values, dtype=float), seed, resamples)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(round(low, 12)), float(round(high, 12))


@dataclass(frozen=True)
class Difference:
    """Разница «после минус до» с 95% интервалом по парному bootstrap."""

    mean: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """Интервал не содержит нуля."""
        return self.low > 0 or self.high < 0


def paired_difference(
    before: Sequence[float], after: Sequence[float], *, seed: int, resamples: int = RESAMPLES
) -> Difference:
    if len(before) != len(after):
        raise ValueError("парное сравнение требует одних и тех же вопросов в обоих прогонах")
    deltas = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    low, high = bootstrap_ci(deltas, seed=seed, resamples=resamples)
    return Difference(mean=float(round(deltas.mean(), 12)), low=low, high=high)
