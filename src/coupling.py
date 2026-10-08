"""Input-coupling metric (Phase S, Gate 1): does a drum hit move its own motor group?

For every hit of the call, count the spikes of the matching motor group in the
window right after the hit (default 0-100 ms) and compare with the same window
of the silent run of the same seed. The silent run uses identical random
numbers for the tonic drive (src/run_fly.py), so the difference is caused by
the call.

Real vs. chosen: the spikes are the simulation's output on the REAL wiring. The
window, and reading "voice i's motor group answers voice i's hit" as coupling,
are CHOSEN.
"""

import numpy as np

WINDOW_MS = 100.0


def hit_window_counts(motor_step_counts, hit_t, dt_ms, window_ms=WINDOW_MS):
    """Per hit: (first sim step, one-past-last sim step) of its window, clipped to the run."""
    n = np.asarray(motor_step_counts).shape[1]
    start = np.floor(np.asarray(hit_t, dtype=float) * 1000.0 / dt_ms + 1e-9).astype(int)
    stop = np.minimum(start + round(window_ms / dt_ms), n)
    return start, stop


def coupling(call_motor, silent_motor, hit_t, hit_voice, voice_names, dt_ms, window_ms=WINDOW_MS):
    """Per voice: spikes in the post-hit windows, call vs silent, and % change.

    Returns {voice name: {"hits", "call_spikes", "silent_spikes", "change_pct", "per_hit_change_pct"}}
    for the voices that have at least one hit. change_pct is None when the
    silent windows hold no spikes at all (a percentage is then undefined).
    """
    call_motor, silent_motor = np.asarray(call_motor), np.asarray(silent_motor)
    start, stop = hit_window_counts(call_motor, hit_t, dt_ms, window_ms)
    out = {}
    for v, name in enumerate(voice_names):
        sel = np.flatnonzero(np.asarray(hit_voice) == v)
        if not len(sel):
            continue
        call = [int(call_motor[v, start[i]:stop[i]].sum()) for i in sel]
        silent = [int(silent_motor[v, start[i]:stop[i]].sum()) for i in sel]
        pct = lambda a, b: None if b == 0 else round(100.0 * (a - b) / b, 2)
        out[name] = {
            "hits": len(sel),
            "call_spikes": sum(call),
            "silent_spikes": sum(silent),
            "change_pct": pct(sum(call), sum(silent)),
            "per_hit_change_pct": [pct(a, b) for a, b in zip(call, silent)],
        }
    return out
