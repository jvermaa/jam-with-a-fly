"""Echo score: how well does the fly's answer repeat the call? (PLAN.md Step 1.5)

Grids are boolean arrays of shape (6 voices, n_steps): True = a hit on that
16th-note step. F1 is computed per voice and overall, and a hit counts as
matched if the other grid has an unmatched hit of the same voice within
+-HIT_TOLERANCE_STEPS. Each hit can be matched at most once.

Real vs. chosen: the score is CHOSEN. "Echo the call" is the goal we set; the
fly has no such goal.
"""

import numpy as np

from src import constants as c


def match_count(call_steps, answer_steps, tolerance):
    """Largest number of one-to-one pairs with |call - answer| <= tolerance (both sorted)."""
    i = j = matched = 0
    while i < len(call_steps) and j < len(answer_steps):
        d = answer_steps[j] - call_steps[i]
        if abs(d) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return matched


def _f1(tp, n_call, n_answer):
    precision = tp / n_answer if n_answer else 0.0
    recall = tp / n_call if n_call else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "matched": tp, "call_hits": n_call, "answer_hits": n_answer}


def echo_f1(call_grid, answer_grid, tolerance=c.HIT_TOLERANCE_STEPS):
    """Per-voice and overall (hits pooled over voices) F1 of answer_grid against call_grid."""
    call_grid, answer_grid = np.asarray(call_grid, dtype=bool), np.asarray(answer_grid, dtype=bool)
    per_voice, tp, n_call, n_answer = [], 0, 0, 0
    for v in range(call_grid.shape[0]):
        cs, as_ = np.flatnonzero(call_grid[v]), np.flatnonzero(answer_grid[v])
        m = match_count(cs, as_, tolerance)
        per_voice.append(_f1(m, len(cs), len(as_)))
        tp, n_call, n_answer = tp + m, n_call + len(cs), n_answer + len(as_)
    return {"overall": _f1(tp, n_call, n_answer), "per_voice": per_voice}


def best_lag(call_grid, raw_grid, max_lag=c.MAX_LAG_STEPS, tolerance=c.HIT_TOLERANCE_STEPS):
    """Shift the raw answer earlier by 0..max_lag steps; return (lag, shifted grid, score) with the best overall F1.

    raw_grid has n_call_steps + extra columns (the simulated tail), so a shifted
    answer still covers every call step. Ties go to the smallest lag.
    """
    n = np.asarray(call_grid).shape[1]
    best = None
    for lag in range(max_lag + 1):
        shifted = np.asarray(raw_grid, dtype=bool)[:, lag:lag + n]
        score = echo_f1(call_grid, shifted, tolerance)
        if best is None or score["overall"]["f1"] > best[2]["overall"]["f1"]:
            best = (lag, shifted, score)
    return best


def random_baseline(call_grid, raw_grid, n_runs=100, seed=0, max_lag=c.MAX_LAG_STEPS, tolerance=c.HIT_TOLERANCE_STEPS):
    """Mean F1 of random answers with the same per-voice hit count as raw_grid.

    Each random answer goes through the same lag search as the fly's answer, so
    the baseline gets the same benefit from picking the best of the lags.
    """
    raw_grid = np.asarray(raw_grid, dtype=bool)
    rng = np.random.default_rng(seed)
    n_voices, width = raw_grid.shape
    n_hits = raw_grid.sum(axis=1)
    overall, per_voice = [], []
    for _ in range(n_runs):
        fake = np.zeros_like(raw_grid)
        for v in range(n_voices):
            fake[v, rng.choice(width, size=int(n_hits[v]), replace=False)] = True
        _, _, score = best_lag(call_grid, fake, max_lag, tolerance)
        overall.append(score["overall"]["f1"])
        per_voice.append([p["f1"] for p in score["per_voice"]])
    return {
        "overall_f1_mean": float(np.mean(overall)),
        "overall_f1_std": float(np.std(overall)),
        "per_voice_f1_mean": [float(x) for x in np.mean(per_voice, axis=0)],
        "n_runs": n_runs,
        "seed": seed,
    }
