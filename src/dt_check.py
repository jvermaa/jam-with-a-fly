"""Timestep check (Phase S, C2): is the runaway an artefact of the 2 ms timestep?

Upstream fly_drums_sim.py widens the timestep from the model's 0.2 ms to 2.0 ms
for speed. At 2 ms a spike's whole synaptic effect lands in one step and the
refractory period rounds to 2 steps. This script reruns the wake check at
dt = 0.2 ms: tonic drive {0, 1, 45} Hz x {silent, call}, one seed.

dt is set through upstream's Params object (flysim.Params.dt), the same way
upstream itself changes it; flysim.py is not edited.

Screen pass: at some drive, the silent run has < 20 % of KCs active per
sixteenth step AND the call run does not drive the whole network to the
ceiling. Only if the screen passes is the coupling metric run at that drive on
seeds 200-203 (mean + 95 % CI per voice; Gate 1a pass = CI lower bound >= +20 %
for kick and snare with KCs sparse).

Real vs. chosen: wiring and neuron model REAL (unmodified); timestep, tonic
drive, encoder and all thresholds here CHOSEN.

Usage (from the repo root):  python -m src.dt_check
"""

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fly_drums_sim import DRIVE_HZ  # upstream's tonic drive rate, unmodified
from src import coupling, encode, kc_gate, sim_pool
from src.provenance import provenance
from src.timing import Timing

DT_MS = 0.2
CALL = ROOT / "calls" / "wwry.mid"
SCREEN_SEED = 200
SEEDS = [200, 201, 202, 203]
DRIVES_HZ = [0.0, 1.0, DRIVE_HZ]
KC_SPARSE_BELOW_PCT = 20.0
# "Does not drive the whole network to ceiling", made checkable: in the call run, fewer than this
# share of ALL neurons at the ceiling, and the Kenyon cells themselves still sparse.
NETWORK_AT_CEILING_BELOW_PCT = 1.0
COUPLING_PASS_PCT = 20.0


def row(r):
    return {
        "total_spikes": r["summary"]["total_spikes_all_neurons"],
        "kc_pct_active_per_16th_mean": r["kc"]["kc_pct_active_per_16th_mean"],
        "kc_pct_active_per_16th_max": r["kc"]["kc_pct_active_per_16th_max"],
        "kc_pct_fired_at_least_once": r["kc"]["kc_pct_fired_at_least_once"],
        "kc_mean_rate_hz": r["kc"]["kc_mean_rate_hz"],
        "pct_all_neurons_above_200hz": r["regions"]["all"]["pct_above_200hz"],
        "pct_all_neurons_at_ceiling": r["regions"]["all"]["pct_at_ceiling"],
        "pct_all_neurons_fired": r["regions"]["all"]["pct_fired"],
        "motor_rate_hz": {"Kick": r["motor_rate_hz"]["Kick"], "Snare": r["motor_rate_hz"]["Snare"]},
        "motor_rate_hz_all_voices": r["motor_rate_hz"],
        "regions": r["regions"],
        "wall_time_s": r["summary"]["runtime_s"],
    }


def main():
    timing = Timing.for_call(CALL)
    names = [v["name"] for v in encode.load_voices()]
    t0 = time.time()
    jobs = [{"call": call, "seed": SCREEN_SEED, "bpm": timing.bpm, "drive_hz": hz, "tag": [hz, call is not None]}
            for hz in DRIVES_HZ for call in (str(CALL), None)]
    by = {tuple(r["tag"]): r for r in sim_pool.run_jobs(jobs, dt_ms=DT_MS)}
    any_run = next(iter(by.values()))

    levels, passing = [], []
    for hz in DRIVES_HZ:
        call, silent = by[(hz, True)], by[(hz, False)]
        checks = {
            "silent_kc_sparse": silent["kc"]["kc_pct_active_per_16th_mean"] < KC_SPARSE_BELOW_PCT,
            "call_not_at_ceiling": (call["regions"]["all"]["pct_at_ceiling"] < NETWORK_AT_CEILING_BELOW_PCT
                                    and call["kc"]["kc_pct_active_per_16th_mean"] < KC_SPARSE_BELOW_PCT),
        }
        cp = coupling.coupling(call["motor_step_counts"], silent["motor_step_counts"],
                               call["hit_t"], call["hit_voice"], names, DT_MS)
        levels.append({"tonic_drive_hz": hz, "call_run": row(call), "silent_run": row(silent),
                       "coupling_0_100ms_screen_seed": cp, "checks": checks, "screen_pass": all(checks.values())})
        if all(checks.values()):
            passing.append(hz)

    result = {
        "phase": "S",
        "step": "C2",
        **provenance(SEEDS if passing else SCREEN_SEED),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": DT_MS,
        "dt_set_by": "flysim.Params.dt on the Params object passed to FlyBrain (not hardcoded in a protected file)",
        "sim_steps": any_run["summary"]["steps"],
        "rate_ceiling_hz": any_run["rate_ceiling_hz"],
        "rate_ceiling_note": "one spike per refractory period; refractory 2.2 ms rounds up to whole steps "
                             "(11 steps = 2.2 ms at dt 0.2; 2 steps = 4 ms, i.e. 250 Hz, at dt 2.0)",
        "at_ceiling_definition": f"mean rate >= {sim_pool.AT_CEILING_FRACTION} x ceiling",
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "kc_kc_scale": 1.0,
        "screen_seed": SCREEN_SEED,
        "screen_rule": f"silent run: KCs active per sixteenth step < {KC_SPARSE_BELOW_PCT} %; call run: "
                       f"< {NETWORK_AT_CEILING_BELOW_PCT} % of all neurons at ceiling and KCs still < {KC_SPARSE_BELOW_PCT} % per step",
        "levels": levels,
        "screen_pass": bool(passing),
        "screen_wall_time_s": round(time.time() - t0, 1),
    }

    if passing:
        # Among passing drive levels, the one with the best screen-seed coupling.
        def worst(hz):
            cp = next(lv for lv in levels if lv["tonic_drive_hz"] == hz)["coupling_0_100ms_screen_seed"]
            vals = [cp["Kick"]["change_pct"], cp["Snare"]["change_pct"]]
            return -1e9 if None in vals else min(vals)
        hz = max(passing, key=worst)
        jobs = [{"call": call, "seed": seed, "bpm": timing.bpm, "drive_hz": hz, "tag": [seed, call is not None]}
                for seed in SEEDS if seed != SCREEN_SEED for call in (str(CALL), None)]
        more = {tuple(r["tag"]): r for r in sim_pool.run_jobs(jobs, dt_ms=DT_MS)}
        more[(SCREEN_SEED, True)], more[(SCREEN_SEED, False)] = by[(hz, True)], by[(hz, False)]
        per_seed = [coupling.coupling(more[(s, True)]["motor_step_counts"], more[(s, False)]["motor_step_counts"],
                                      more[(s, True)]["hit_t"], more[(s, True)]["hit_voice"], names, DT_MS) for s in SEEDS]
        ci = {v: kc_gate.mean_ci([p[v]["change_pct"] for p in per_seed]) for v in ("Kick", "Snare")}
        kc_step = float(np.mean([more[(s, True)]["kc"]["kc_pct_active_per_16th_mean"] for s in SEEDS]))
        checks = {
            "kc_sparse_per_16th": kc_step < KC_SPARSE_BELOW_PCT,
            "kick_ci_low_ge_20": ci["Kick"]["ci95_low"] is not None and ci["Kick"]["ci95_low"] >= COUPLING_PASS_PCT,
            "snare_ci_low_ge_20": ci["Snare"]["ci95_low"] is not None and ci["Snare"]["ci95_low"] >= COUPLING_PASS_PCT,
        }
        result["coupling"] = {
            "tonic_drive_hz": hz, "seeds": SEEDS, "ci": ci, "per_seed": per_seed,
            "kc_pct_active_per_16th_mean_over_seeds": round(kc_step, 2),
            "rows_per_seed": {str(s): {"call_run": row(more[(s, True)]), "silent_run": row(more[(s, False)])} for s in SEEDS},
            "checks": checks, "gate_1a_pass": all(checks.values()),
        }
    result["gate_1a_pass"] = bool(passing) and result["coupling"]["gate_1a_pass"]
    result["total_wall_time_s"] = round(time.time() - t0, 1)

    out = ROOT / "results" / "wwry_dt02_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"dt {DT_MS} ms, {result['sim_steps']} steps, ceiling {result['rate_ceiling_hz']} Hz")
    for lv in levels:
        for name in ("silent_run", "call_run"):
            r = lv[name]
            print(f"drive {lv['tonic_drive_hz']:4.1f} {name:10}: spikes {r['total_spikes']:>9} | KC/16th {r['kc_pct_active_per_16th_mean']}% "
                  f"{r['kc_mean_rate_hz']} Hz | >200Hz {r['pct_all_neurons_above_200hz']}% ceiling {r['pct_all_neurons_at_ceiling']}% "
                  f"fired {r['pct_all_neurons_fired']}% | motor {r['motor_rate_hz']} | {r['wall_time_s']} s")
        print("   regions >200Hz (call):", {k: v["pct_above_200hz"] for k, v in lv["call_run"]["regions"].items()})
        print("   coupling seed", SCREEN_SEED, {k: v["change_pct"] for k, v in lv["coupling_0_100ms_screen_seed"].items()}, lv["checks"])
    if passing:
        print("coupling:", json.dumps({k: result["coupling"][k] for k in ("tonic_drive_hz", "ci", "checks")}, indent=1))
    print("screen", "PASS" if passing else "FAIL", "| gate 1a", "PASS" if result["gate_1a_pass"] else "FAIL",
          f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
