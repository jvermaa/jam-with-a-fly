"""Phase S, Gate 1a-A: the pure helper mean_ci (mean and 95 % Student-t interval
over four seeds). Made-up numbers only; no brain is built and nothing is run."""

import numpy as np
import pytest

from src import kc_gate

T = 3.182446  # two-sided 95 % Student t quantile, 3 degrees of freedom


def test_t_quantile_constant():
    assert kc_gate.T_975_DF3 == T


def test_known_example():
    r = kc_gate.mean_ci([10, 20, 30, 40])
    # sample std (ddof=1) = 12.9099; half-width = 3.182446 * 12.9099 / 2 = 20.54
    assert r["mean"] == 25.0
    assert r["ci95_low"] == 4.46
    assert r["ci95_high"] == 45.54
    assert r["per_seed"] == [10, 20, 30, 40]


def test_interval_is_mean_plus_minus_t_times_sample_std_over_two():
    values = [31.7, -4.25, 12.0, 58.9]
    r = kc_gate.mean_ci(values)
    mean = sum(values) / 4
    std = (sum((x - mean) ** 2 for x in values) / 3) ** 0.5  # ddof = 1
    half = T * std / 2
    assert std == pytest.approx(np.std(values, ddof=1))
    assert r["mean"] == round(mean, 2)
    assert r["ci95_low"] == pytest.approx(mean - half, abs=0.005 + 1e-9)
    assert r["ci95_high"] == pytest.approx(mean + half, abs=0.005 + 1e-9)
    assert r["ci95_low"] == round(mean - half, 2)
    assert r["ci95_high"] == round(mean + half, 2)
    assert r["ci95_low"] < r["mean"] < r["ci95_high"]
    assert r["per_seed"] == values


def test_values_are_rounded_to_two_decimals():
    r = kc_gate.mean_ci([1.23456, 2.34567, 3.45678, 4.56789])
    for key in ("mean", "ci95_low", "ci95_high"):
        assert isinstance(r[key], float)
        assert r[key] == round(r[key], 2)


def test_identical_values_give_zero_width_interval():
    r = kc_gate.mean_ci([7.5, 7.5, 7.5, 7.5])
    assert r["mean"] == 7.5
    assert r["ci95_low"] == 7.5
    assert r["ci95_high"] == 7.5


@pytest.mark.parametrize("values", [[None, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, None], [None, None, None, None]])
def test_any_none_gives_no_mean_or_interval(values):
    r = kc_gate.mean_ci(values)
    assert r["mean"] is None
    assert r["ci95_low"] is None
    assert r["ci95_high"] is None
    assert r["per_seed"] == values
