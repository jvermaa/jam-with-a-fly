"""Step 1.5 pass check for the echo score: a synthetic perfect echo gives
F1 = 1.0 and an empty answer gives F1 = 0. Grids are made up here, so no
connectome data, build/ or results/ files are needed."""

import numpy as np
import pytest

from src import constants as c
from src import score

SEED = 0


def grid(hits, n_steps=c.CALL_STEPS):
    """hits: list of (voice, step) -> bool grid (6, n_steps)."""
    g = np.zeros((6, n_steps), dtype=bool)
    for v, s in hits:
        g[v, s] = True
    return g


# Kick on the beats, snare on 2 and 4, hats on the off-beats, one crash on the last step.
CALL_HITS = ([(0, s) for s in range(0, 32, 8)] + [(1, s) for s in range(4, 32, 8)]
             + [(2, s) for s in range(2, 32, 4)] + [(5, 31)])
CALL = grid(CALL_HITS)


def delayed(hits, delay, extra=4):
    return grid([(v, s + delay) for v, s in hits], c.CALL_STEPS + extra)


# --- required pass check ---------------------------------------------------

def test_perfect_echo_gives_f1_one():
    out = score.echo_f1(CALL, CALL.copy())
    assert out["overall"]["f1"] == 1.0
    assert out["overall"]["precision"] == 1.0
    assert out["overall"]["recall"] == 1.0
    assert out["overall"]["matched"] == out["overall"]["call_hits"] == out["overall"]["answer_hits"] == len(CALL_HITS)
    assert len(out["per_voice"]) == 6
    for v, p in enumerate(out["per_voice"]):
        if CALL[v].any():
            assert p["f1"] == 1.0
        else:
            assert p["f1"] == 0.0 and p["call_hits"] == 0


def test_empty_answer_gives_f1_zero():
    out = score.echo_f1(CALL, np.zeros_like(CALL))
    assert out["overall"]["f1"] == 0.0
    assert out["overall"]["precision"] == 0.0
    assert out["overall"]["recall"] == 0.0
    assert out["overall"]["matched"] == 0
    assert out["overall"]["answer_hits"] == 0
    assert all(p["f1"] == 0.0 for p in out["per_voice"])


# --- matching rules --------------------------------------------------------

def test_empty_call_and_empty_answer_gives_zero_not_nan():
    empty = np.zeros((6, c.CALL_STEPS), dtype=bool)
    assert score.echo_f1(empty, empty)["overall"]["f1"] == 0.0


@pytest.mark.parametrize("offset, expected", [(-2, 0.0), (-1, 1.0), (0, 1.0), (1, 1.0), (2, 0.0)])
def test_one_step_off_matches_two_steps_off_does_not(offset, expected):
    out = score.echo_f1(grid([(0, 10)]), grid([(0, 10 + offset)]))
    assert out["overall"]["f1"] == expected


def test_match_count():
    assert score.match_count([10], [11], 1) == 1
    assert score.match_count([10], [12], 1) == 0
    assert score.match_count([10], [12], 2) == 1
    assert score.match_count([], [3], 1) == 0
    assert score.match_count([3], [], 1) == 0
    # Chain: each call hit takes the answer hit one step later.
    assert score.match_count([0, 2, 4], [1, 3, 5], 1) == 3


def test_one_answer_hit_cannot_match_two_call_hits():
    assert score.match_count([3, 5], [4], 1) == 1
    out = score.echo_f1(grid([(0, 3), (0, 5)]), grid([(0, 4)]))
    assert out["overall"]["matched"] == 1
    assert out["overall"]["precision"] == 1.0
    assert out["overall"]["recall"] == 0.5
    # And the other way round: two answer hits cannot both match one call hit.
    assert score.match_count([4], [3, 5], 1) == 1


def test_hit_on_wrong_voice_does_not_count():
    out = score.echo_f1(grid([(0, 4)]), grid([(1, 4)]))
    assert out["overall"]["f1"] == 0.0
    assert out["overall"]["matched"] == 0
    assert out["overall"]["call_hits"] == 1
    assert out["overall"]["answer_hits"] == 1
    assert out["per_voice"][0]["answer_hits"] == 0
    assert out["per_voice"][1]["call_hits"] == 0


def test_precision_recall_hand_worked():
    # Voice 0: call 0,4,8,12; answer 0 (exact), 5 (one off 4), 20 (no partner) -> 2 matched.
    # Voice 1: call 2; answer 2 -> 1 matched. Voice 2: answer 7 only -> 0 matched.
    call = grid([(0, 0), (0, 4), (0, 8), (0, 12), (1, 2)])
    answer = grid([(0, 0), (0, 5), (0, 20), (1, 2), (2, 7)])
    out = score.echo_f1(call, answer)

    v0 = out["per_voice"][0]
    assert (v0["matched"], v0["call_hits"], v0["answer_hits"]) == (2, 4, 3)
    assert v0["precision"] == pytest.approx(2 / 3)
    assert v0["recall"] == pytest.approx(2 / 4)
    assert v0["f1"] == pytest.approx(4 / 7)
    assert out["per_voice"][1]["f1"] == 1.0
    assert out["per_voice"][2]["f1"] == 0.0

    # Overall pools the hits: 3 matched of 5 call hits and 5 answer hits.
    o = out["overall"]
    assert (o["matched"], o["call_hits"], o["answer_hits"]) == (3, 5, 5)
    assert o["precision"] == pytest.approx(3 / 5)
    assert o["recall"] == pytest.approx(3 / 5)
    assert o["f1"] == pytest.approx(3 / 5)


# --- lag search ------------------------------------------------------------

def test_best_lag_recovers_known_lag():
    # CALL has a hit on the last step (31), so a too-small lag cuts that hit off.
    raw = delayed(CALL_HITS, 2)
    assert raw.shape == (6, c.CALL_STEPS + 4)
    lag, shifted, out = score.best_lag(CALL, raw)
    assert lag == 2
    assert shifted.shape == CALL.shape
    assert np.array_equal(shifted, CALL)
    assert out["overall"]["f1"] == 1.0


def test_best_lag_zero_for_undelayed_answer():
    lag, shifted, out = score.best_lag(CALL, delayed(CALL_HITS, 0))
    assert lag == 0
    assert np.array_equal(shifted, CALL)
    assert out["overall"]["f1"] == 1.0


def test_best_lag_prefers_smallest_lag_on_ties():
    # No hit near the end: an answer one step late scores F1 1.0 at lag 0 (inside
    # the +-1 tolerance), at lag 1 (exact) and at lag 2 (one early). Lag 0 wins.
    hits = [(0, 4), (1, 12), (2, 20)]
    lag, _, out = score.best_lag(grid(hits), delayed(hits, 1))
    assert lag == 0
    assert out["overall"]["f1"] == 1.0
    # All-empty answer: every lag scores 0, lag 0 is kept.
    lag, _, out = score.best_lag(CALL, np.zeros((6, c.CALL_STEPS + 4), dtype=bool))
    assert lag == 0
    assert out["overall"]["f1"] == 0.0


def test_best_lag_respects_max_lag():
    lag, _, out = score.best_lag(CALL, delayed(CALL_HITS, 2), max_lag=0)
    assert lag == 0
    assert out["overall"]["f1"] == 0.0  # two steps off is outside the tolerance


# --- random baseline -------------------------------------------------------

def test_random_baseline_is_deterministic_for_a_fixed_seed():
    raw = delayed(CALL_HITS, 1)
    a = score.random_baseline(CALL, raw, n_runs=20, seed=SEED)
    b = score.random_baseline(CALL, raw, n_runs=20, seed=SEED)
    other = score.random_baseline(CALL, raw, n_runs=20, seed=SEED + 1)
    assert a == b
    assert a != other
    assert a["n_runs"] == 20
    assert a["seed"] == SEED


def test_random_baseline_lies_between_zero_and_one():
    out = score.random_baseline(CALL, delayed(CALL_HITS, 1), n_runs=20, seed=SEED)
    assert 0.0 <= out["overall_f1_mean"] <= 1.0
    assert out["overall_f1_std"] >= 0.0
    assert len(out["per_voice_f1_mean"]) == 6
    assert all(0.0 <= x <= 1.0 for x in out["per_voice_f1_mean"])
    # Random answers must score below the perfect echo they were sized from.
    assert out["overall_f1_mean"] < 1.0


def test_random_baseline_is_zero_when_answer_has_no_hits():
    out = score.random_baseline(CALL, np.zeros((6, c.CALL_STEPS + 4), dtype=bool), n_runs=10, seed=SEED)
    assert out["overall_f1_mean"] == 0.0
    assert out["overall_f1_std"] == 0.0
    assert out["per_voice_f1_mean"] == [0.0] * 6
