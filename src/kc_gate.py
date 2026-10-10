"""Gate 1a, step A (Phase S): does turning down KC->KC excitation un-saturate the network?

KC->KC weights are scaled by 0, 0.25 and 0.5 through the wrapper in
src/kc_recurrence.py (a CHOSEN modelling change, see that file; the protected
simulator is not edited).

  1. Screen, seed 0: every scale x tonic drive {0, 1, 4.5, 11.25, 22.5, 45} Hz,
     call and silent. For each scale the drive level is picked on seed 0:
     among levels where the KCs are sparse and the kick and snare motor groups
     fire, the one with the highest min(kick, snare) coupling; if no level
     qualifies, the highest min coupling regardless.
  2. Confirm, seeds 0-3 at the picked drive: per-voice coupling mean and 95 %
     confidence interval over the four seeds (Student t, 3 degrees of freedom).

PASS: for some scale, the interval's lower bound is >= +20 % for kick AND
snare, with fewer than 20 % of KCs active per sixteenth step.

Real vs. chosen: wiring REAL; KC->KC scale, tonic drive, encoder, window and
the pass rule CHOSEN.

Usage (from the repo root):  python -m src.kc_gate
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fly_drums_sim import DRIVE_HZ  # upstream's tonic drive rate, unmodified
from src import constants as c
from src import coupling, encode, sim_pool
from src.provenance import provenance
from src.timing import Timing

CALL = ROOT / "calls" / "wwry.mid"
SCALES = [0.0, 0.25, 0.5]
DRIVES_HZ = [0.0, 1.0, DRIVE_HZ * 0.1, DRIVE_HZ * 0.25, DRIVE_HZ * 0.5, DRIVE_HZ]
SCREEN_SEED = 0
SEEDS = [0, 1, 2, 3]
KC_SPARSE_BELOW_PCT = 20.0
COUPLING_PASS_PCT = 20.0
T_975_DF3 = 3.182446  # two-sided 95 % Student t quantile for n = 4


def pair_metrics(call, silent, names):
    cp = coupling.coupling(call["motor_step_counts"], silent["motor_step_counts"],
                           call["hit_t"], call["hit_voice"], names, c.DT_MS)
    return {
        "kc_call_run": call["kc"],
        "kc_silent_run": silent["kc"],
        "pct_all_neurons_above_200hz_call_run": call["regions"]["all"]["pct_above_200hz"],
        "regions_call_run": call["regions"],
        "motor_rate_hz_call_run": call["motor_rate_hz"],
        "motor_rate_hz_silent_run": silent["motor_rate_hz"],
        "total_spikes": {"call": call["summary"]["total_spikes_all_neurons"],
                         "silent": silent["summary"]["total_spikes_all_neurons"]},
        "coupling_0_100ms": cp,
    }


def min_coupling(m):
    k, s = m["coupling_0_100ms"]["Kick"]["change_pct"], m["coupling_0_100ms"]["Snare"]["change_pct"]
    return None if k is None or s is None else min(k, s)


def qualifies(m):
    return (m["kc_call_run"]["kc_pct_active_per_16th_mean"] < KC_SPARSE_BELOW_PCT
            and m["motor_rate_hz_call_run"]["Kick"] > 0 and m["motor_rate_hz_call_run"]["Snare"] > 0
            and min_coupling(m) is not None)


def mean_ci(values):
    if any(v is None for v in values):
        return {"per_seed": values, "mean": None, "ci95_low": None, "ci95_high": None}
    v = np.asarray(values, dtype=float)
    half = T_975_DF3 * v.std(ddof=1) / np.sqrt(len(v))
    return {"per_seed": values, "mean": round(float(v.mean()), 2),
            "ci95_low": round(float(v.mean() - half), 2), "ci95_high": round(float(v.mean() + half), 2)}


def run_pairs(keys, timing):
    """keys: (scale, drive_hz, seed) -> {key: metrics}."""
    jobs = [{"call": call, "seed": seed, "bpm": timing.bpm, "drive_hz": hz, "kc_kc_scale": scale,
             "tag": [scale, hz, seed, call is not None]}
            for scale, hz, seed in keys for call in (str(CALL), None)]
    by = {tuple(r["tag"]): r for r in sim_pool.run_jobs(jobs)}
    names = [v["name"] for v in encode.load_voices()]
    return {k: pair_metrics(by[(*k, True)], by[(*k, False)], names) for k in keys}


def main():
    timing = Timing.for_call(CALL)
    screen = run_pairs([(s, hz, SCREEN_SEED) for s in SCALES for hz in DRIVES_HZ], timing)

    picked = {}
    for s in SCALES:
        rows = [(hz, screen[(s, hz, SCREEN_SEED)]) for hz in DRIVES_HZ]
        ok = [(hz, m) for hz, m in rows if qualifies(m)]
        pool = ok or [(hz, m) for hz, m in rows if min_coupling(m) is not None]
        picked[s] = {"drive_hz": max(pool, key=lambda r: min_coupling(r[1]))[0], "from_qualifying_levels": bool(ok)}

    confirm = run_pairs([(s, picked[s]["drive_hz"], seed) for s in SCALES for seed in SEEDS if seed != SCREEN_SEED], timing)
    confirm.update({(s, picked[s]["drive_hz"], SCREEN_SEED): screen[(s, picked[s]["drive_hz"], SCREEN_SEED)] for s in SCALES})

    scales = []
    for s in SCALES:
        hz = picked[s]["drive_hz"]
        per_seed = [confirm[(s, hz, seed)] for seed in SEEDS]
        ci = {v: mean_ci([m["coupling_0_100ms"][v]["change_pct"] for m in per_seed]) for v in ("Kick", "Snare")}
        kc_step = float(np.mean([m["kc_call_run"]["kc_pct_active_per_16th_mean"] for m in per_seed]))
        checks = {
            "kc_sparse_per_16th": kc_step < KC_SPARSE_BELOW_PCT,
            "kick_ci_low_ge_20": ci["Kick"]["ci95_low"] is not None and ci["Kick"]["ci95_low"] >= COUPLING_PASS_PCT,
            "snare_ci_low_ge_20": ci["Snare"]["ci95_low"] is not None and ci["Snare"]["ci95_low"] >= COUPLING_PASS_PCT,
        }
        scales.append({
            "kc_kc_scale": s,
            "picked_drive_hz": hz,
            "picked_from_qualifying_levels": picked[s]["from_qualifying_levels"],
            "screen_seed0_by_drive": [{"tonic_drive_hz": d, **screen[(s, d, SCREEN_SEED)]} for d in DRIVES_HZ],
            "confirm_seeds": SEEDS,
            "coupling_ci": ci,
            "kc_pct_active_per_16th_mean_over_seeds": round(kc_step, 2),
            "kc_mean_rate_hz_per_seed": [m["kc_call_run"]["kc_mean_rate_hz"] for m in per_seed],
            "pct_all_neurons_above_200hz_per_seed": [m["pct_all_neurons_above_200hz_call_run"] for m in per_seed],
            "motor_rate_hz_call_run_per_seed": [m["motor_rate_hz_call_run"] for m in per_seed],
            "regions_call_run_seed0": per_seed[0]["regions_call_run"],
            "checks": checks,
            "pass": all(checks.values()),
        })

    result = {
        "phase": "S",
        "gate": "1a-A",
        **provenance(SEEDS),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "modelling_choice": "KC-KC treated as non-excitatory (weights scaled by kc_kc_scale in a wrapper); "
                            "real contacts mostly axo-axonic; modelling choice, not stated by the connectome",
        "ci_method": "mean +- t(0.975, df=3) x sample std / sqrt(4) over seeds 0-3; per seed, change_pct pools all hits of the voice",
        "pass_rule": f"CI lower bound >= +{COUPLING_PASS_PCT} % for kick and snare, and mean share of KCs active "
                     f"per sixteenth step < {KC_SPARSE_BELOW_PCT} %",
        "scales": scales,
        "pass": any(s["pass"] for s in scales),
    }
    out = ROOT / "results" / "wwry_kc_gate.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    for s in scales:
        print(f"=== KC-KC scale {s['kc_kc_scale']}")
        for row in s["screen_seed0_by_drive"]:
            k = row["kc_call_run"]
            print(f"  drive {row['tonic_drive_hz']:5.2f}: KC/16th {k['kc_pct_active_per_16th_mean']:6.2f}% {k['kc_mean_rate_hz']:6.2f} Hz | "
                  f">200Hz all {row['pct_all_neurons_above_200hz_call_run']}% | motor K {row['motor_rate_hz_call_run']['Kick']} "
                  f"S {row['motor_rate_hz_call_run']['Snare']} | coupling K {row['coupling_0_100ms']['Kick']['change_pct']} "
                  f"S {row['coupling_0_100ms']['Snare']['change_pct']} | spikes {row['total_spikes']}")
        print(f"  picked drive {s['picked_drive_hz']} (qualifying: {s['picked_from_qualifying_levels']}); "
              f"kick {s['coupling_ci']['Kick']} snare {s['coupling_ci']['Snare']}")
        print(f"  KC/16th over seeds {s['kc_pct_active_per_16th_mean_over_seeds']}% | {s['checks']}")
        print("  regions >200Hz seed0:", {k: v["pct_above_200hz"] for k, v in s["regions_call_run_seed0"].items()})
    print("PASS" if result["pass"] else "FAIL", f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
