"""Profile one simulation and measure what the trainable parameters can do (Phase S, Gate 0).

Three questions, each answered by an actual run:
  1. Cost of one WWRY run on this machine: wall time, CPU time / wall time (1.0 =
     one core busy), peak memory.
  2. What exactly upstream's training changes: the KC->MBON synapses that
     mushroom.MushroomBody addresses, and how they split by dopamine side.
  3. Does changing those synapses move the motor output at all? Same seed, same
     call, same random numbers; only the gains differ:
       * "mutant": one draw of upstream's mutation (Gaussian, MUTATE_SIGMA, clipped
         to [floor, 1]);
       * "floor":  every gain at the floor, the largest change the rule allows.
     If neither moves the motor step counts, no amount of training on these
     parameters can change the fly's answer.

Real vs. chosen: the wiring, neuron model and plasticity site are REAL (upstream
flysim.py / mushroom.py, imported and not edited). The mutation, its size and
the "floor" probe are CHOSEN.

Usage (from the repo root):  python -m src.profile_sim
"""

import json
import os
import pathlib
import resource
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fly_drums_sim import (  # upstream's settings, unmodified
    MUTATE_SIGMA,
    POP_SIZE,
    TRAIN_GENERATIONS,
)
from flysim import FlyBrain  # upstream, unmodified
from mushroom import MushroomBody  # upstream, unmodified
from src import constants as c
from src import decode, run_fly
from src.provenance import provenance
from src.timing import Timing

SEED = 0
CALL = ROOT / "calls" / "wwry.mid"
UNUSED_STORE = ROOT / "checkpoints" / "unused_mb_store.npz"  # never written; keeps build/ out of it


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e6 if sys.platform == "darwin" else 1e3)  # macOS reports bytes, Linux kilobytes


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def timed_run(fb, call, timing):
    w0, c0 = time.time(), cpu_seconds()
    arrays, summary = run_fly.run(fb, call, SEED, timing)
    wall, cpu = time.time() - w0, cpu_seconds() - c0
    return arrays, summary, {"wall_s": round(wall, 2), "cpu_s": round(cpu, 2), "cpu_over_wall": round(cpu / wall, 3)}


def compare(base, other, silent, call_grid, timing):
    """How far `other` motor output is from `base` (same seed and input)."""
    a, b = base["motor_step_counts"], other["motor_step_counts"]
    voices = [v["name"] for v in run_fly.encode.load_voices()]
    d_base = decode.decode(a, silent["motor_step_counts"], call_grid, c.DT_MS, timing)
    d_other = decode.decode(b, silent["motor_step_counts"], call_grid, c.DT_MS, timing)
    return {
        "motor_step_counts_identical": bool(np.array_equal(a, b)),
        "sim_steps_that_differ": int((a != b).any(axis=0).sum()),
        "sim_steps_total": int(a.shape[1]),
        "motor_spikes_base": {v: int(a[i].sum()) for i, v in enumerate(voices)},
        "motor_spikes_other": {v: int(b[i].sum()) for i, v in enumerate(voices)},
        "motor_spikes_change_pct": {v: round(100.0 * (int(b[i].sum()) - int(a[i].sum())) / max(int(a[i].sum()), 1), 2)
                                    for i, v in enumerate(voices)},
        "decoded_cells_that_flip": int((d_base["raw_grid"] != d_other["raw_grid"]).sum()),
        "decoded_cells_total": int(d_base["raw_grid"].size),
        "f1_base": d_base["score"]["overall"]["f1"],
        "f1_other": d_other["score"]["overall"]["f1"],
        "threshold_note": "both decoded against the untrained silent run of the same seed",
    }


def main():
    timing = Timing.for_call(CALL)
    t0 = time.time()
    fb = FlyBrain(p=run_fly.SimParams())
    load_s, rss_loaded = time.time() - t0, peak_rss_mb()

    call, call_summary, call_cost = timed_run(fb, CALL, timing)
    rss_one_run = peak_rss_mb()
    silent, _, silent_cost = timed_run(fb, None, timing)

    mb = MushroomBody(fb, store=UNUSED_STORE)
    kc_member = np.zeros(fb.n, dtype=bool)
    kc_member[mb.kc] = True
    kc_spikes = int(kc_member[call["spike_neuron"]].sum())
    kc_fired = len(np.unique(call["spike_neuron"][kc_member[call["spike_neuron"]]]))
    sim_sec = call_summary["sim_seconds"]

    call_grid = decode.call_grid_from_hits(call["hit_t"], call["hit_voice"], timing=timing)

    rng = np.random.default_rng(SEED)
    base_gain = mb.gain.copy()
    mutant = np.clip(base_gain + rng.standard_normal(base_gain.shape).astype(np.float32) * MUTATE_SIGMA,
                     mb.floor, 1.0).astype(np.float32)
    probes = {}
    for name, gain in (("mutant", mutant), ("floor", np.full_like(base_gain, mb.floor))):
        mb.gain = gain
        mb.apply()
        other, _, _ = timed_run(fb, CALL, timing)
        probes[name] = {
            "mean_gain": round(float(gain.mean()), 4),
            "synapses_below_1": int((gain < 1.0).sum()),
            **compare(call, other, silent, call_grid, timing),
        }
    mb.gain = base_gain
    mb.apply()

    result = {
        "phase": "S",
        "gate": 0,
        **provenance(SEED),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "sim_seconds": sim_sec,
        "sim_steps": call_summary["steps"],
        "cpu_count_logical": os.cpu_count(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),  # None = library default, nothing pinned
        "load_brain_s": round(load_s, 2),
        "rss_after_load_mb": round(rss_loaded, 1),
        "one_call_run": call_cost,
        "one_silent_run": silent_cost,
        "peak_rss_after_one_run_mb": round(rss_one_run, 1),
        "peak_rss_whole_script_mb": round(peak_rss_mb(), 1),
        "total_spikes_call_run": call_summary["total_spikes_all_neurons"],
        "upstream_training_settings": {
            "source": "fly_drums_sim.py",
            "generations": TRAIN_GENERATIONS,
            "population": POP_SIZE,
            "mutate_sigma": MUTATE_SIGMA,
            "mutation_rng": "np.random (unseeded) upstream",
        },
        "trainable": {
            "what": "one gain per KC->MBON synapse (mushroom.MushroomBody.gain), multiplying that synapse's weight",
            **{k: v for k, v in mb.stats().items() if k in ("synapses", "reward_side", "punish_side")},
            "kenyon_cells": len(mb.kc),
            "mbons": len(mb.mbon),
            "mbon_types_without_a_side": mb.unassigned,
            "gain_floor": mb.floor,
            "gain_ceiling": 1.0,
            "dopamine_lr": mb.lr,
            "trace_decay_per_observe": mb.trace_decay,
            "loaded_stored_gains": bool(mb.loaded),
        },
        "kenyon_cell_activity_call_run": {
            "kcs_that_fired": kc_fired,
            "kc_spikes": kc_spikes,
            "mean_rate_hz": round(kc_spikes / len(mb.kc) / sim_sec, 1),
            "model_max_rate_hz": round(1000.0 / (fb.refr_steps * c.DT_MS), 1),
        },
        "gain_sensitivity": probes,
    }
    out = ROOT / "results" / "wwry_profile.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("one_call_run", "peak_rss_after_one_run_mb", "trainable",
                                             "kenyon_cell_activity_call_run", "gain_sensitivity")}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
