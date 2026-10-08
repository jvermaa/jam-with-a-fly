"""Small measurements shared by the Phase S C3/C4 checks (input strength, reset-per-hit).

All of these read spike counts the simulation produced; none changes the model.

Real vs. chosen: the counts are output of the REAL wiring. Every window,
threshold and baseline defined here is CHOSEN.
"""

import numpy as np

RUNAWAY_KC_PCT = 50.0      # "runaway" = more than this share of KCs firing within one refractory cycle
PERSIST_AFTER_MS = 500.0   # "self-sustained" = spikes still occurring this long after the last input ends


def first_time_above_ms(kc_step_counts, n_kc, cycle_steps, dt_ms, pct=RUNAWAY_KC_PCT):
    """Time (ms, start of the window) of the first refractory-cycle window in which more than
    pct % of the KCs fire; None if that never happens.

    A neuron fires at most once per refractory cycle (cycle_steps simulation steps), so the
    spike count in a cycle-long window equals the number of distinct KCs that fired in it.
    """
    counts = np.asarray(kc_step_counts, dtype=np.int64)
    if len(counts) < cycle_steps:
        return None
    rolling = np.convolve(counts, np.ones(cycle_steps, dtype=np.int64), mode="valid")
    above = np.flatnonzero(100.0 * rolling / n_kc > pct)
    return None if not len(above) else round(float(above[0] * dt_ms), 3)


def persistence(all_step_counts, last_input_end_s, dt_ms, after_ms=PERSIST_AFTER_MS):
    """What is left once the input has stopped.

    Returns {"last_spike_s", "spikes_after_ms", "spikes_after", "late_window_s", "self_sustained"}:
    self_sustained is True when any neuron still spikes more than after_ms after the last
    input spike could have been delivered. None when the run is too short to tell.
    """
    counts = np.asarray(all_step_counts, dtype=np.int64)
    fired = np.flatnonzero(counts)
    start = int(np.ceil((last_input_end_s * 1000.0 + after_ms) / dt_ms - 1e-9))
    late_sec = max(len(counts) - start, 0) * dt_ms / 1000.0
    late = int(counts[start:].sum()) if start < len(counts) else 0
    return {
        "last_spike_s": None if not len(fired) else round(float(fired[-1] * dt_ms / 1000.0), 4),
        "spikes_after_ms": after_ms,
        "spikes_after": late,
        "late_window_s": round(late_sec, 4),
        "self_sustained": None if late_sec <= 0 else late > 0,
    }


def window_sum(row, start_steps, window_steps):
    """Per window: spikes of one group in [start, start + window_steps), clipped to the run."""
    row = np.asarray(row)
    return [int(row[s:min(s + window_steps, len(row))].sum()) for s in start_steps]


def pct_change(a, b):
    """100 x (a - b) / b, or None when b is 0 (a percentage is then undefined)."""
    return None if b == 0 else round(100.0 * (a - b) / b, 2)


def own_hit_vs_quiet(call_row, silent_row, hit_steps, quiet_steps, window_steps):
    """One motor group, one run pair: spikes in the windows after its own voice's hits, against
    (a) the same windows of the silent run and (b) windows of the same call run that start on
    grid steps where no voice is hit ("quiet" windows; means per window, since the counts differ).
    """
    own = window_sum(call_row, hit_steps, window_steps)
    sil = window_sum(silent_row, hit_steps, window_steps)
    quiet = window_sum(call_row, quiet_steps, window_steps)
    own_mean = float(np.mean(own)) if own else 0.0
    quiet_mean = float(np.mean(quiet)) if quiet else 0.0
    return {
        "hits": len(own),
        "call_spikes": sum(own),
        "silent_spikes": sum(sil),
        "change_pct_vs_silent": pct_change(sum(own), sum(sil)),
        "quiet_windows": len(quiet),
        "call_spikes_per_hit_window": round(own_mean, 2),
        "call_spikes_per_quiet_window": round(quiet_mean, 2),
        "change_pct_vs_quiet_windows": pct_change(own_mean, quiet_mean),
    }
