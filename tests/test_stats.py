"""Bootstrap-интервалы на выборках с известным ответом."""

from __future__ import annotations

import pytest

from kz_labor_rag.eval.stats import bootstrap_ci, paired_difference


class TestBootstrapCI:
    def test_constant_sample_has_a_point_interval(self):
        assert bootstrap_ci([1.0] * 20, seed=1) == (1.0, 1.0)

    def test_interval_contains_the_mean(self):
        values = [0.0] * 30 + [1.0] * 70
        low, high = bootstrap_ci(values, seed=1)
        assert low < 0.7 < high

    def test_more_data_narrows_the_interval(self):
        small = bootstrap_ci([0.0, 1.0] * 10, seed=1)
        large = bootstrap_ci([0.0, 1.0] * 200, seed=1)
        assert large[1] - large[0] < small[1] - small[0]

    def test_same_seed_same_interval(self):
        values = [0.0, 0.5, 1.0, 1.0, 0.0, 1.0]
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_empty_sample_has_no_interval(self):
        assert bootstrap_ci([], seed=1) is None


class TestPairedDifference:
    def test_identical_systems_differ_by_zero(self):
        a = [0.0, 1.0, 0.5, 1.0]
        diff = paired_difference(a, a, seed=1)
        assert diff.mean == 0.0 and diff.low == 0.0 and diff.high == 0.0
        assert not diff.significant

    def test_clear_improvement_excludes_zero(self):
        before = [0.0] * 40 + [1.0] * 40
        after = [1.0] * 70 + [0.0] * 10
        diff = paired_difference(before, after, seed=1)
        assert diff.mean == pytest.approx(0.375)
        assert diff.low > 0
        assert diff.significant

    def test_pairing_matters(self):
        # Одна и та же разница средних: при парах, которые меняются согласованно,
        # она надёжна; если бы выборки были независимы, интервал был бы шире.
        before = [0.0, 0.2, 0.4, 0.6, 0.8] * 8
        after = [x + 0.1 for x in before]
        diff = paired_difference(before, after, seed=1)
        assert diff.low == pytest.approx(0.1) and diff.high == pytest.approx(0.1)

    def test_lengths_must_match(self):
        with pytest.raises(ValueError):
            paired_difference([1.0], [1.0, 0.0], seed=1)
