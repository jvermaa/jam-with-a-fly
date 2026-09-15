"""
Drive the real male-CNS connectome (flysim.FlyBrain) toward a drum performance.

Not a demo of "AI drumming" — the leg, wing, haltere and neck motor neurons
read out here are real, measured cells (superclass vnc_motor, subclass fl/ml/
hl/wm/hm/nm). The learning is the real one too: mushroom.MushroomBody, the same
dopamine-gated Kenyon-cell -> MBON depression rule used elsewhere in this repo,
rewarded here by how closely the real motor output's 16-step rhythm matches
real human drumming: six tracks from Google Magenta's Groove MIDI Dataset
(electronic kit) blended with real acoustic-kit performances from MDBDrums,
when a local checkout of the latter is available -- see build_mdb_grid().

WHAT IS REAL AND WHAT IS CHOSEN, STATED PLAINLY
* The connectome, the LIF dynamics, the six motor populations and the KC->MBON
  learning rule: real, measured, unmodified from the rest of this repo.
* dt is widened from the repo's 0.2ms to 2.0ms so a bar of drumming simulates
  in about a minute instead of about eight. That trades temporal resolution
  for wall-clock time; it is not the repo's validated default.
* The tonic drive onto vnc_intrinsic (the population that contains the walking
  central-pattern generators) is invented -- there is no camera, no plume, no
  sugar. It is the only thing "asking" the brain to move at all.
* The reward (rhythm-match to real grooves) is invented, exactly as the
  backroom's reward (novelty) is invented and disclosed in README.md. The
  circuit and the plasticity rule it drives are not.

Output: fly_drums_export.json, next to this script, sized for a browser page.

TRAINING, v2: real selection, not just one nudged trajectory
Each generation now evaluates a small population of candidate KC->MBON gain
vectors (mutations of the current one) from the *same* starting brain state,
keeps whichever actually scores best against the real grooves, and only then
applies the real dopamine-depression step on that winner. The population/
selection/mutation is invented (an evolution strategy bolted onto the real
plasticity site); the gain vector it operates on, the depression rule, and the
KC eligibility trace are the same real ones as v1. This is what makes the
fitness curve actually climb instead of drifting on one undirected trajectory.

Runs on the GPU (flysim_gpu.FlyBrainGPU) when available -- this repo's own
benchmark is a single run being "only a little faster" than CPU, but a
population of candidates is exactly the batched case it's fast at, and it
also makes a much longer final performance affordable.
"""
import copy
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from flysim import Params
from mushroom import MushroomBody

ROOT = Path(__file__).parent

try:
    from flysim_gpu import FlyBrainGPU as _Brain
    _DEVICE = "cuda"
except Exception:
    from flysim import FlyBrain as _Brain
    _DEVICE = "cpu"


class SimParams(Params):
    dt = 2.0  # widened from the repo default 0.2ms -- see module docstring


CHANNEL_SUBCLASS = {
    "kick": "hl",     # hindleg motor neurons
    "snare": "ml",    # midleg motor neurons
    "hihat": "fl",    # foreleg motor neurons
    "cymbal": "wm",   # wing muscle motor neurons
    "tom": "hm",      # haltere muscle motor neurons
    "chord": "nm",    # neck muscle motor neurons
}
CH_WEIGHT = {"kick": 1.4, "snare": 1.3, "hihat": 1.0, "tom": 0.55, "cymbal": 0.5, "chord": 0.45}
# lower bars for the naturally smaller/quieter populations (tom=16 neurons, chord=44)
# so they actually cross threshold and get heard, instead of only kick/snare/hihat
CH_THRESHOLD = {"kick": 0.55, "snare": 0.55, "hihat": 0.45, "tom": 0.38, "cymbal": 0.4, "chord": 0.35}
STEPS_PER_BAR = 16
WINDOW_MS = 150.0  # one 16th note
DRIVE_HZ = 45.0
TRAIN_GENERATIONS = 10
POP_SIZE = 4          # candidate gain-vectors evaluated per generation
MUTATE_SIGMA = 0.14
FINAL_BARS = 24

GROOVE_JSON = ROOT / "reference_data" / "groove_reference.json"

# MDBDrums (Southall et al. 2017, ISMIR) is CC BY-NC-SA 4.0 -- share-alike and
# non-commercial. So unlike the Groove MIDI Dataset (CC-BY, no such
# restriction), nothing derived from it is committed to this repo: this reads
# your own local checkout of https://github.com/CarlSouthall/MDBDrums, at
# MDB_DRUMS_DIR (env var) or the default sibling-folder guess below, and
# quietly skips it if that path doesn't exist. Class-annotation labels
# (KD/SD/HH/TT/CY/OT) map onto the same six channels the Groove data does;
# "OT" (other percussion) has no equivalent here and is dropped.
MDB_DRUMS_DIR = Path(os.environ.get("MDB_DRUMS_DIR", ROOT.parent / "MDBDrums" / "MDB Drums"))
MDB_LABEL_MAP = {"KD": "kick", "SD": "snare", "HH": "hihat", "TT": "tom", "CY": "cymbal"}


def build_mdb_grid():
    """16-step probability grid from local MDBDrums onset+beat annotations, or None if absent."""
    class_dir, beats_dir = MDB_DRUMS_DIR / "annotations" / "class", MDB_DRUMS_DIR / "annotations" / "beats"
    if not class_dir.is_dir():
        return None
    grid = {ch: np.zeros(16) for ch in CHANNEL_SUBCLASS}
    n_tracks = 0
    for class_file in sorted(class_dir.glob("*_class.txt")):
        base = class_file.name[: -len("_class.txt")]
        beats_file = beats_dir / f"{base}_MIX.beats"
        if not beats_file.exists():
            continue
        beats = []
        for line in beats_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                beats.append((float(parts[0]), int(parts[1])))
        if len(beats) < 2:
            continue
        n_tracks += 1
        hits = []
        for line in class_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            hits.append((float(parts[0].strip()), parts[1].strip()))
        for t, label in hits:
            ch = MDB_LABEL_MAP.get(label)
            if ch is None:
                continue
            i = 0
            while i + 1 < len(beats) and beats[i + 1][0] <= t:
                i += 1
            if i + 1 >= len(beats):
                continue
            t0, b0 = beats[i]
            t1, _ = beats[i + 1]
            if t1 <= t0:
                continue
            frac = min(max((t - t0) / (t1 - t0), 0.0), 0.999)
            step = ((b0 - 1) * 4 + round(frac * 4)) % 16
            grid[ch][step] += 1.0  # onset-only annotations: no velocity, every hit counts equally
    if n_tracks == 0:
        return None
    for ch in MDB_LABEL_MAP.values():
        m = grid[ch].max() or 1.0
        grid[ch] = grid[ch] / m
    print(f"  MDBDrums: folded in {n_tracks} real acoustic-kit tracks from {MDB_DRUMS_DIR}")
    return grid


def build_target_grid():
    tracks = json.loads(GROOVE_JSON.read_text())
    grid = {ch: np.zeros(16) for ch in CHANNEL_SUBCLASS}
    # only kick/snare/hihat/tom/cymbal have real drum-dataset equivalents
    drum_map = {"kick": "kick", "snare": "snare", "hihat": "hihat", "tom": "tom", "cymbal": "cymbal"}
    for track in tracks:
        for h in track["hits"]:
            drum = h["drum"]
            if drum not in drum_map.values():
                continue
            ch = [k for k, v in drum_map.items() if v == drum][0]
            step = round(h["beat"] * 4) % 16
            grid[ch][step] += h["velocity"] / 127.0
    for ch in drum_map:
        m = grid[ch].max() or 1.0
        grid[ch] = grid[ch] / m

    mdb_grid = build_mdb_grid()
    if mdb_grid is not None:
        # blend: real electronic-kit grooves (Groove MIDI) averaged with real
        # acoustic-kit performances (MDBDrums), equal weight
        for ch in drum_map:
            grid[ch] = (grid[ch] + mdb_grid[ch]) / 2.0
    else:
        print(f"  MDBDrums not found at {MDB_DRUMS_DIR} -- training on Groove MIDI alone "
              f"(set MDB_DRUMS_DIR to fold it in)")

    grid["chord"] = np.array([0.75, 0, 0, 0, 0.15, 0, 0.2, 0, 0.6, 0, 0, 0, 0.25, 0, 0.15, 0])
    return {k: v.tolist() for k, v in grid.items()}


def fitness(pattern, target):
    """pattern, target: {channel: 16 floats in [0,1]}"""
    err = 0.0
    for ch, w in CH_WEIGHT.items():
        d = np.array(pattern[ch]) - np.array(target[ch])
        err += w * float((d * d).mean())
    var_bonus = 0.0
    for ch in ("kick", "snare", "hihat"):
        v = np.array(pattern[ch])
        var_bonus += min(float(v.var()), 0.09)
    base = 1.0 / (1.0 + err)
    return max(0.0, min(100.0, base * 100 * 0.82 + var_bonus * 100 * 0.6))


def run_bar(fb, populations, drive, state, seed, mb=None, kc=None):
    """One bar (16 windows) of real simulation. Returns (norm_pattern, raw_pattern, state).

    Pass mb+kc to have this trajectory feed the real KC eligibility trace (the
    "counts toward learning" run); omit them for a throwaway fitness-only probe.
    """
    pattern = {ch: [] for ch in CHANNEL_SUBCLASS}
    steps = int(round(WINDOW_MS / fb.p.dt))
    for s in range(STEPS_PER_BAR):
        out = fb.run(drive, steps=steps, record=populations, seed=seed * 1000 + s, state=state)
        state = out["_state"]
        if mb is not None:
            mb.observe(out["_fired"][np.isin(out["_fired"], kc)])
        for ch, idx in populations.items():
            pattern[ch].append(float(np.mean(out[ch])) if len(out[ch]) else 0.0)
    norm = {}
    for ch, vals in pattern.items():
        arr = np.array(vals)
        lo, hi = arr.min(), arr.max()
        norm[ch] = ((arr - lo) / (hi - lo) if hi > lo else arr * 0).tolist()
    return norm, pattern, state


def clone_state(state):
    return None if state is None else copy.deepcopy(state)


def main():
    t_start = time.time()
    print(f"loading real connectome onto {_DEVICE} ...")
    fb = _Brain(p=SimParams()) if _DEVICE == "cpu" else _Brain(p=SimParams(), device=_DEVICE)
    mb = MushroomBody(fb)
    print(f"  {fb.n:,} neurons, mushroom body: {mb.stats()}")

    populations = {ch: fb.where(subclass=sub) for ch, sub in CHANNEL_SUBCLASS.items()}
    for ch, idx in populations.items():
        print(f"  {ch:8} <- subclass '{CHANNEL_SUBCLASS[ch]}', {len(idx)} real motor neurons")
    kc = fb.where(type_re=r"^KC")
    vnc_int = fb.where(superclass="vnc_intrinsic")
    drive = {tuple(vnc_int.tolist()): DRIVE_HZ}

    target = build_target_grid()

    print(f"\ntraining ({TRAIN_GENERATIONS} generations x {POP_SIZE} candidates, "
          f"real selection + real KC->MBON dopamine depression) ...")
    reward_history = []
    state = None
    best_ever_fit, best_ever_gain = -1.0, mb.gain.copy()
    for gen in range(TRAIN_GENERATIONS):
        t0 = time.time()
        base_state = clone_state(state)
        base_gain = mb.gain.copy()
        candidates = [base_gain] + [
            np.clip(base_gain + np.random.randn(*base_gain.shape).astype(np.float32) * MUTATE_SIGMA,
                    mb.floor, 1.0).astype(np.float32)
            for _ in range(POP_SIZE - 1)
        ]
        scored = []
        for gi, gain in enumerate(candidates):
            mb.gain = gain
            mb.apply()
            norm_pattern, _, _ = run_bar(fb, populations, drive, clone_state(base_state), seed=gen * 100 + gi)
            scored.append((fitness(norm_pattern, target), gain))
        scored.sort(key=lambda x: -x[0])
        best_fit, best_gain = scored[0]

        # the real trajectory: continue the actual `state` timeline on the winning
        # gain, with real KC observation + real dopamine-gated depression on top
        mb.gain = best_gain
        mb.apply()
        _, _, state = run_bar(fb, populations, drive, base_state, seed=gen, mb=mb, kc=kc)
        mb.dopamine(+1, amount=best_fit / 100.0)
        mb.apply()

        reward_history.append(best_fit)
        if best_fit > best_ever_fit:
            best_ever_fit, best_ever_gain = best_fit, best_gain.copy()
        print(f"  gen {gen+1}/{TRAIN_GENERATIONS}  best of {POP_SIZE} = {best_fit:5.1f}  "
              f"(candidates {[round(s[0],1) for s in scored]}, best-ever {best_ever_fit:5.1f}, "
              f"{time.time()-t0:.1f}s, mb depressed={mb.stats()['depressed']})")

    # perform with whatever the search actually found best, not just wherever
    # the last generation's noisy trajectory happened to land
    mb.gain = best_ever_gain
    mb.apply()

    print(f"\nrendering {FINAL_BARS}-bar performance from the trained circuit ...")
    plan = (["intro"] * 2 + ["groove"] * 5 + ["variation"] * 2 + ["fill"]
            + ["groove"] * 5 + ["variation"] * 2 + ["fill"]
            + ["groove"] * 4 + ["outro"] * 2)
    sections = [plan[i % len(plan)] for i in range(FINAL_BARS)]
    trained_gain = mb.gain.copy()
    hits = []
    bar_fitness = []
    for bar_i, section in enumerate(sections):
        t0 = time.time()
        if section == "fill":
            # a one-off mutated variant for an audible fill, never kept
            mb.gain = np.clip(trained_gain + np.random.randn(*trained_gain.shape).astype(np.float32) * 0.22,
                               mb.floor, 1.0).astype(np.float32)
            mb.apply()
        else:
            mb.gain = trained_gain
            mb.apply()
        bar_drive_hz = DRIVE_HZ * (1.15 if section == "variation" else 1.0)
        bar_drive = {tuple(vnc_int.tolist()): bar_drive_hz}
        norm_pattern, raw_pattern, state = run_bar(fb, populations, bar_drive, state, seed=1000 + bar_i)
        bar_fitness.append(fitness(norm_pattern, target))
        density = {"intro": 0.6, "outro": 0.5}.get(section, 1.0)
        for step in range(STEPS_PER_BAR):
            for ch in CHANNEL_SUBCLASS:
                p = norm_pattern[ch][step] * density
                if p > CH_THRESHOLD[ch]:
                    hits.append({
                        "bar": bar_i, "section": section, "step": step, "ch": ch,
                        "vel": round(min(1.0, 0.35 + p * 0.75), 3),
                    })
        print(f"  bar {bar_i+1}/{FINAL_BARS} [{section:9}] fitness {bar_fitness[-1]:5.1f}  ({time.time()-t0:.1f}s)")
    mb.gain = trained_gain
    mb.apply()

    print("\npacking real neuron point cloud ...")
    import pandas as pd
    ann = pd.read_feather(ROOT / "data" / "body-annotations.feather")
    traced = ann[(ann.status == "Traced") & (ann.statusLabel != "Glia")]
    have = traced["somaLocation"].notna()
    traced = traced.loc[have]
    coords = np.array([c for c in traced["somaLocation"]], dtype=np.float32)
    superclass_names = sorted(traced["superclass"].dropna().unique().tolist())
    sc_code = {s: i for i, s in enumerate(superclass_names)}
    codes = traced["superclass"].map(lambda s: sc_code.get(s, -1)).to_numpy()
    subclass_arr = traced["subclass"].fillna("").to_numpy()
    is_channel = np.zeros(len(traced), dtype=np.int8)
    channel_names = list(CHANNEL_SUBCLASS.keys())
    for i, ch in enumerate(channel_names):
        is_channel[subclass_arr == CHANNEL_SUBCLASS[ch]] = i + 1

    cloud = {
        "superclass_names": superclass_names,
        "channel_names": channel_names,
        "xyz": np.round(coords, 0).astype(int).tolist(),
        "superclass_code": codes.tolist(),
        "channel_code": is_channel.tolist(),
    }

    export = {
        "meta": {
            "neurons": int(fb.n),
            "edges": int(fb.W.nnz),
            "dt_ms": fb.p.dt,
            "drive_hz": DRIVE_HZ,
            "generations": TRAIN_GENERATIONS,
            "bars": FINAL_BARS,
            "channel_subclass": CHANNEL_SUBCLASS,
            "wall_clock_s": round(time.time() - t_start, 1),
        },
        "reward_history": reward_history,
        "bar_fitness": bar_fitness,
        "target_grid": target,
        "hits": hits,
        "mushroom_stats": mb.stats(),
        "neuron_cloud": cloud,
    }
    out_path = ROOT / "fly_drums_export.json"
    out_path.write_text(json.dumps(export))
    print(f"\nwrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB) in {time.time()-t_start:.1f}s total")


if __name__ == "__main__":
    main()
