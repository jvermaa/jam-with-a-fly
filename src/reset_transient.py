"""Reset-per-hit transient (Phase S, C4): what does one hit do to a network at rest?

For each of the 32 sixteenth steps of calls/wwry.mid: every neuron is reset to
rest, only that step's input is presented (the step's hits, or nothing), and
100 ms are simulated with no tonic drive. Because the state is reset before
every step, nothing carries over from one beat to the next.

Reported: motor spikes per voice in the 0-20, 0-50 and 0-100 ms windows for hit
steps vs silent steps, the time from a single hit to runaway, and the coupling
metric per window on seeds 200-203 (mean + 95 % CI, kick and snare).

The input spikes of a step are exactly the ones the encoder produces for that
hit in the full call (same seed), shifted so the step starts at time 0.

Real vs. chosen: wiring and neuron model REAL (unmodified). CHOSEN: state reset
between beats (to isolate the onset response), tonic drive 0 Hz, the encoder,
the windows, and the runaway threshold.

Usage (from the repo root):  python -m src.reset_transient
"""

import json
import multiprocessing
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constants as c
from src import encode, kc_gate, transient
from src.provenance import provenance
from src.timing import Timing

CALL = ROOT / "calls" / "wwry.mid"
SEEDS = [200, 201, 202, 203]
SIM_MS = 100.0
WINDOWS_MS = [20.0, 50.0, 100.0]
TONIC_DRIVE_HZ = 0.0
GATE_VOICES = ("Kick", "Snare")
KC_SPARSE_BELOW_PCT = 20.0
COUPLING_PASS_PCT = 20.0


def step_inputs(spike_t, step_index, step_sec, burst_sec):
    """Mask of the encoder spikes that belong to grid step step_index, and their time within it.

    Hits sit on the grid and a burst is shorter than a step, so the spikes of a step's hits are
    exactly those in [step start, step start + burst length).
    """
    t0 = step_index * step_sec
    rel = np.asarray(spike_t, dtype=float) - t0
    sel = (rel > -1e-9) & (rel < burst_sec - 1e-12)
    return sel, np.maximum(rel[sel], 0.0)


def run_seed(seed):
    """All 32 reset-and-present runs for one seed. Returns compact per-step arrays."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from flysim import FlyBrain  # upstream, unmodified
    from src import run_fly

    fb = FlyBrain(p=run_fly.SimParams())
    dt_ms = float(fb.p.dt)
    n_steps = round(SIM_MS / dt_ms)
    timing = Timing.for_call(CALL)
    jo_bodies = encode.load_jo_groups()
    voices, jo, motor = run_fly.load_groups(fb, jo_bodies)
    jo_all = np.concatenate(jo)
    kc = fb.where(type_re=r"^KC")
    is_kc = np.zeros(fb.n, dtype=bool)
    is_kc[kc] = True

    cfg = encode.load_encoder_config()
    hits = encode.read_hits(CALL, voices, timing)
    enc = encode.encode(hits, jo_bodies, config=cfg, seed=seed, voices=voices)
    hit_step = [round(h["t"] / timing.step_sec) for h in hits]

    steps = []
    for s in range(c.CALL_STEPS):
        sel, rel = step_inputs(enc["spike_t"], s, timing.step_sec, cfg["burst_duration_ms"] / 1000.0)
        input_step = np.floor(rel * 1000.0 / dt_ms + 1e-9).astype(np.int64)
        input_neuron = np.array([fb.body_to_i[int(b)] for b in enc["spike_body"][sel]], dtype=np.int64)
        keep = input_step < n_steps
        spike_step, spike_neuron = run_fly.simulate(fb, jo_all, input_step[keep], input_neuron[keep], seed,
                                                    n_steps, TONIC_DRIVE_HZ)
        kc_sel = is_kc[spike_neuron]
        steps.append({
            "step": s,
            "voices_hit": sorted({voices[h["voice"]]["name"] for h, hs in zip(hits, hit_step) if hs == s}),
            "input_spikes_delivered": len(set(zip(input_step[keep].tolist(), input_neuron.tolist()))),
            "total_spikes": len(spike_step),
            "neurons_fired": len(np.unique(spike_neuron)),
            "kc_fired": len(np.unique(spike_neuron[kc_sel])),
            "motor_step_counts": run_fly.group_step_counts(spike_step, spike_neuron, motor, fb.n, n_steps),
            "kc_step_counts": np.bincount(spike_step[kc_sel], minlength=n_steps),
            "all_step_counts": np.bincount(spike_step, minlength=n_steps),
        })
    return {"seed": seed, "dt_ms": dt_ms, "n_steps": n_steps, "n_neurons": fb.n, "n_kc": len(kc),
            "cycle_steps": int(fb.refr_steps), "voice_names": [v["name"] for v in voices],
            "motor_group_sizes": [len(m) for m in motor], "steps": steps}


def kind_of(step):
    """"silent" or the single voice hit on that step (WWRY never hits two voices at once)."""
    return "silent" if not step["voices_hit"] else "+".join(step["voices_hit"])


def seed_table(run):
    """Per window and motor voice: mean spikes per step, by the kind of step presented."""
    names, dt_ms = run["voice_names"], run["dt_ms"]
    kinds = sorted({kind_of(s) for s in run["steps"]})
    out = {}
    for w in WINDOWS_MS:
        k = round(w / dt_ms)
        table = {}
        for kind in kinds:
            rows = [s for s in run["steps"] if kind_of(s) == kind]
            table[kind] = {"steps": len(rows),
                           "mean_motor_spikes": {n: round(float(np.mean([r["motor_step_counts"][i, :k].sum() for r in rows])), 3)
                                                 for i, n in enumerate(names)}}
        out[f"0-{int(w)}ms"] = table
    return out


def coupling_for(table, voice, other):
    """Own-voice hit vs silent step (the Gate 1a metric) and vs the other gate voice's hit."""
    own = table[voice]["mean_motor_spikes"][voice]
    return {
        "own_hit_mean_spikes": own,
        "silent_step_mean_spikes": table["silent"]["mean_motor_spikes"][voice],
        "other_voice_hit_mean_spikes": table[other]["mean_motor_spikes"][voice],
        "change_pct_vs_silent_step": transient.pct_change(own, table["silent"]["mean_motor_spikes"][voice]),
        "change_pct_vs_other_voice_hit": transient.pct_change(own, table[other]["mean_motor_spikes"][voice]),
    }


def main():
    timing = Timing.for_call(CALL)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(len(SEEDS)) as pool:
        runs = pool.map(run_seed, SEEDS, chunksize=1)
    first = runs[0]
    dt_ms, n_kc, cycle = first["dt_ms"], first["n_kc"], first["cycle_steps"]

    per_seed, runaway, kc_pct, hit_rows = {}, [], [], []
    for run in runs:
        table = seed_table(run)
        per_seed[str(run["seed"])] = {
            "by_window": table,
            "coupling": {w: {v: coupling_for(t, v, o) for v, o in (GATE_VOICES, GATE_VOICES[::-1])} for w, t in table.items()},
        }
        for s in run["steps"]:
            if not s["voices_hit"]:
                continue
            t = transient.first_time_above_ms(s["kc_step_counts"], n_kc, cycle, dt_ms)
            runaway.append(t)
            kc_pct.append(100.0 * s["kc_fired"] / n_kc)
            hit_rows.append({"seed": run["seed"], "step": s["step"], "voice": kind_of(s),
                             "input_spikes_delivered": s["input_spikes_delivered"], "total_spikes": s["total_spikes"],
                             "pct_neurons_fired": round(100.0 * s["neurons_fired"] / run["n_neurons"], 2),
                             "pct_kc_fired": round(100.0 * s["kc_fired"] / n_kc, 2),
                             "time_to_runaway_ms": t})
    silent_spikes = sum(s["total_spikes"] for run in runs for s in run["steps"] if not s["voices_hit"])

    ci = {}
    for w in per_seed[str(SEEDS[0])]["by_window"]:
        ci[w] = {}
        for v in GATE_VOICES:
            vals = [per_seed[str(s)]["coupling"][w][v] for s in SEEDS]
            ci[w][v] = {
                "own_hit_mean_spikes": kc_gate.mean_ci([x["own_hit_mean_spikes"] for x in vals]),
                "silent_step_mean_spikes": kc_gate.mean_ci([x["silent_step_mean_spikes"] for x in vals]),
                "change_pct_vs_silent_step": kc_gate.mean_ci([x["change_pct_vs_silent_step"] for x in vals]),
                "change_pct_vs_other_voice_hit": kc_gate.mean_ci([x["change_pct_vs_other_voice_hit"] for x in vals]),
            }

    reached = [t for t in runaway if t is not None]
    kc_mean = float(np.mean(kc_pct))
    w100 = ci["0-100ms"]
    silent_zero = all(w100[v]["silent_step_mean_spikes"]["mean"] == 0 for v in GATE_VOICES)
    checks = {
        "kc_sparse": kc_mean < KC_SPARSE_BELOW_PCT,
        "motor_active": all(w100[v]["own_hit_mean_spikes"]["mean"] > 0 for v in GATE_VOICES),
        "kick_ci_low_ge_20": None if silent_zero else (w100["Kick"]["change_pct_vs_silent_step"]["ci95_low"] or -1e9) >= COUPLING_PASS_PCT,
        "snare_ci_low_ge_20": None if silent_zero else (w100["Snare"]["change_pct_vs_silent_step"]["ci95_low"] or -1e9) >= COUPLING_PASS_PCT,
    }
    if not checks["kc_sparse"] or not checks["motor_active"]:
        gate = False
    elif silent_zero:
        gate = None
    else:
        gate = bool(checks["kick_ci_low_ge_20"] and checks["snare_ci_low_ge_20"])

    result = {
        "phase": "S",
        "step": "C4",
        **provenance(SEEDS),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": dt_ms,
        "tonic_drive_hz": TONIC_DRIVE_HZ,
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "sim_ms_per_step": SIM_MS,
        "decision": "state reset between beats; chosen to isolate the onset response",
        "method": "per sixteenth step of the call: all neurons at rest, only that step's encoder spikes, 100 ms simulated",
        "runaway_definition": f"first {cycle * dt_ms:g} ms window (one refractory cycle) in which more than "
                              f"{transient.RUNAWAY_KC_PCT:g} % of KCs fire",
        "kc_sparse_definition": f"mean share of KCs that fire at least once in the {SIM_MS:g} ms after a hit < {KC_SPARSE_BELOW_PCT:g} % "
                                "(a sixteenth step is 183 ms, so this window is shorter and the test no stricter than Gate 1a's)",
        "ci_method": "mean +- t(0.975, df=3) x sample std / sqrt(4) over seeds 200-203; per seed, mean spikes per step of each kind",
        "motor_group_sizes": dict(zip(first["voice_names"], first["motor_group_sizes"])),
        "silent_steps_total_spikes_all_seeds": silent_spikes,
        "time_to_runaway_ms": {
            "hit_steps": len(runaway),
            "reached_runaway": len(reached),
            "min": min(reached) if reached else None,
            "median": float(np.median(reached)) if reached else None,
            "max": max(reached) if reached else None,
            "by_voice": {v: {"hit_steps": len(ts), "reached_runaway": len([t for t in ts if t is not None]),
                             "median": float(np.median([t for t in ts if t is not None])) if any(t is not None for t in ts) else None}
                         for v in GATE_VOICES
                         for ts in [[r["time_to_runaway_ms"] for r in hit_rows if r["voice"] == v]]},
        },
        "kc_pct_fired_within_100ms_of_a_hit_mean": round(kc_mean, 2),
        "coupling_ci_by_window": ci,
        "per_seed": per_seed,
        "hit_steps": hit_rows,
        "checks": checks,
        "gate_1a_pass": gate,
        "gate_1a_note": ("silent steps hold 0 motor spikes at 0 Hz tonic drive, so a % change against silent is undefined; "
                         "left for the human to rule on" if gate is None else None),
    }
    out = ROOT / "results" / "wwry_reset_transient.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    print("time to runaway:", {k: v for k, v in result["time_to_runaway_ms"].items()})
    print("KC % fired within 100 ms of a hit:", result["kc_pct_fired_within_100ms_of_a_hit_mean"],
          "| silent-step spikes:", silent_spikes)
    for w, d in ci.items():
        for v in GATE_VOICES:
            print(f"  {w:8} {v:5}: own {d[v]['own_hit_mean_spikes']['mean']} silent {d[v]['silent_step_mean_spikes']['mean']} | "
                  f"vs silent {d[v]['change_pct_vs_silent_step']} | vs other hit {d[v]['change_pct_vs_other_voice_hit']}")
    print("per-seed table (seed 200, 0-100ms):", json.dumps(per_seed["200"]["by_window"]["0-100ms"]))
    for r in hit_rows[:12]:
        print("  ", r)
    print(checks, "| gate 1a:", gate, f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
