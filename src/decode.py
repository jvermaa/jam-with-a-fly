"""Decoder: motor-neuron spikes -> drum hits -> .mid (PLAN.md Step 1.5).

  1. Bin each motor group's spikes into 16th-note steps of 125 ms.
  2. A step is a hit if its count is above that group's threshold:
     silent-baseline mean + 2 x std of the group's step counts. The threshold
     comes from the silent run only; it is never tuned to the call.
  3. Lag search: shift the whole answer earlier by 0..MAX_LAG_STEPS steps and
     keep the lag with the best echo F1. The lag is logged.
  4. Velocity from how far the count is above threshold, clipped to 40..127.

The fly's hits are never edited: no hit is added, removed or moved on its own.
The only shift is the single global lag from step 3, applied to every hit alike.

Real vs. chosen: the spikes are the simulation's output on the REAL wiring.
Reading a motor group as a drum, the threshold rule, the lag and the velocity
mapping are all CHOSEN.

Usage (from the repo root):
  python -m src.decode results/run_call_01_seed0.npz results/silent_seed0.npz
"""

import json
import pathlib
import sys

import mido
import numpy as np

from src import constants as c
from src import encode, score
from src.provenance import provenance

ROOT = pathlib.Path(__file__).resolve().parent.parent
THRESHOLD_STDS = 2.0     # PLAN.md: mean + 2 x std
VELOCITY_MIN, VELOCITY_MAX = 40, 127
VELOCITY_FULL_STDS = 3.0  # this many silent-run stds above threshold = velocity 127
TICKS_PER_BEAT = 480


def bin_steps(step_counts, dt_ms):
    """(voices, sim_steps) spike counts -> (voices, n_16th_steps) counts. Only whole 16th steps are kept."""
    step_counts = np.asarray(step_counts)
    t = np.arange(step_counts.shape[1]) * dt_ms / 1000.0
    bins = np.floor(t / c.STEP_SEC + 1e-9).astype(int)
    n_bins = int(np.floor(step_counts.shape[1] * dt_ms / 1000.0 / c.STEP_SEC + 1e-9))
    out = np.zeros((step_counts.shape[0], n_bins), dtype=np.int64)
    for v in range(step_counts.shape[0]):
        out[v] = np.bincount(bins, weights=step_counts[v], minlength=n_bins + 1)[:n_bins]
    return out


def thresholds(silent_binned):
    """Per-voice (threshold, mean, std) from the silent baseline's step counts."""
    mean, std = silent_binned.mean(axis=1), silent_binned.std(axis=1)
    return mean + THRESHOLD_STDS * std, mean, std


def call_grid_from_hits(hit_t, hit_voice, n_voices=6):
    grid = np.zeros((n_voices, c.CALL_STEPS), dtype=bool)
    steps = np.round(np.asarray(hit_t) / c.STEP_SEC).astype(int)
    for s, v in zip(steps, hit_voice):
        if 0 <= s < c.CALL_STEPS:
            grid[int(v), s] = True
    return grid


def velocities(binned, thr, std):
    """Velocity for every (voice, step); only meaningful where binned > thr."""
    excess = (binned - thr[:, None]) / np.where(std > 0, std, 1.0)[:, None]
    frac = np.clip(excess / VELOCITY_FULL_STDS, 0.0, 1.0)
    return np.clip(np.round(VELOCITY_MIN + frac * (VELOCITY_MAX - VELOCITY_MIN)), VELOCITY_MIN, VELOCITY_MAX).astype(int)


def decode(call_motor_steps, silent_motor_steps, call_grid, dt_ms):
    """Returns dict: raw_grid, thresholds, lag, answer grid/velocities (call-length), score."""
    binned = bin_steps(call_motor_steps, dt_ms)
    thr, mean, std = thresholds(bin_steps(silent_motor_steps, dt_ms))
    raw_grid = binned > thr[:, None]
    vel = velocities(binned, thr, std)
    lag, answer_grid, sc = score.best_lag(call_grid, raw_grid)
    return {
        "binned": binned, "threshold": thr, "silent_mean": mean, "silent_std": std,
        "raw_grid": raw_grid, "lag_steps": lag,
        "answer_grid": answer_grid, "answer_vel": vel[:, lag:lag + call_grid.shape[1]],
        "score": sc,
    }


def answer_hits(answer_grid, answer_vel):
    """Sorted list of {"step", "t", "voice", "vel"} for the answer."""
    hits = [{"step": int(s), "t": float(s * c.STEP_SEC), "voice": int(v), "vel": int(answer_vel[v, s])}
            for v, s in zip(*np.nonzero(answer_grid))]
    return sorted(hits, key=lambda h: (h["step"], h["voice"]))


def write_midi(hits, path, voices=None):
    """Answer hits -> one-track .mid at 120 BPM, exactly 2 bars, voice-table notes, 16th-note lengths."""
    voices = voices or encode.load_voices()
    note = {v["idx"]: v["midi_note"] for v in voices}
    ticks_per_step = TICKS_PER_BEAT * c.BEATS_PER_BAR // c.STEPS_PER_BAR
    events = []
    for h in hits:
        on = h["step"] * ticks_per_step
        events.append((on, 1, mido.Message("note_on", channel=9, note=note[h["voice"]], velocity=h["vel"])))
        events.append((on + ticks_per_step, 0, mido.Message("note_off", channel=9, note=note[h["voice"]], velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    mid = mido.MidiFile(type=0, ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(c.BPM), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    now = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - now))
        now = tick
    end = c.CALL_BARS * c.BEATS_PER_BAR * TICKS_PER_BEAT
    track.append(mido.MetaMessage("end_of_track", time=max(end - now, 0)))
    mid.save(path)


def decode_run(run_npz, silent_npz, label="untrained"):
    """Decode + score one saved run. Writes the .mid and the score JSON; returns the score dict."""
    run_npz, silent_npz = pathlib.Path(run_npz), pathlib.Path(silent_npz)
    run, silent = np.load(run_npz), np.load(silent_npz)
    seed, dt_ms = int(run["seed"]), float(run["dt_ms"])
    if int(silent["seed"]) != seed:
        raise ValueError("silent baseline has a different seed than the run")
    voices = encode.load_voices()
    run_id = run_npz.stem.removeprefix("run_")  # e.g. call_01_seed0

    call_grid = call_grid_from_hits(run["hit_t"], run["hit_voice"])
    d = decode(run["motor_step_counts"], silent["motor_step_counts"], call_grid, dt_ms)
    hits = answer_hits(d["answer_grid"], d["answer_vel"])
    baseline = score.random_baseline(call_grid, d["raw_grid"], n_runs=100, seed=seed)

    mid_path = ROOT / "answers" / f"{label}_{run_id}.mid"
    write_midi(hits, mid_path, voices)

    n_cells = call_grid.size
    result = {
        "step": "1.5",
        **provenance(seed),
        "run_id": f"{label}_{run_id}",
        "run_npz": str(run_npz.resolve().relative_to(ROOT)),
        "silent_npz": str(silent_npz.resolve().relative_to(ROOT)),
        "answer_mid": str(mid_path.relative_to(ROOT)),
        "lag_steps": d["lag_steps"],
        "f1": d["score"]["overall"]["f1"],
        "baseline_f1": baseline["overall_f1_mean"],
        "score": d["score"],
        "random_baseline": baseline,
        "n_answer_hits": len(hits),
        "hit_density": len(hits) / n_cells,
        "voices_used": int(d["answer_grid"].any(axis=1).sum()),
        "not_silent_not_saturated": 0 < len(hits) < n_cells,
        "per_voice": [{
            "voice": v["name"],
            "threshold": float(d["threshold"][i]),
            "silent_mean": float(d["silent_mean"][i]),
            "silent_std": float(d["silent_std"][i]),
            "call_hits": int(call_grid[i].sum()),
            "answer_hits": int(d["answer_grid"][i].sum()),
            "f1": d["score"]["per_voice"][i]["f1"],
            "baseline_f1": baseline["per_voice_f1_mean"][i],
            "call_grid": "".join("x" if x else "." for x in call_grid[i]),
            "answer_grid": "".join("x" if x else "." for x in d["answer_grid"][i]),
        } for i, v in enumerate(voices)],
        "answer_hits": hits,
    }
    (ROOT / "results" / f"score_{label}_{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    r = decode_run(sys.argv[1], sys.argv[2])
    print(f"{r['run_id']}: {r['n_answer_hits']} hits, {r['voices_used']} voices, lag {r['lag_steps']}, "
          f"F1 {r['f1']:.3f} vs random baseline {r['baseline_f1']:.3f}")
    for p in r["per_voice"]:
        print(f"  {p['voice']:11} call   {p['call_grid']}\n  {'':11} answer {p['answer_grid']}")
    print(f"wrote {r['answer_mid']}")


if __name__ == "__main__":
    main()
