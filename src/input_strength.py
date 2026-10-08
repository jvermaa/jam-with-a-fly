"""Input strength at 0 Hz tonic drive (Phase S, C3): is there a sound level that does not ignite the network?

With no tonic drive the model is silent until the call arrives
(results/wwry_wake_check.json), and the default encoder then ignites a
self-sustained state. This grid makes the input weaker and stronger:

  JO set {all 554, auditory-only} x burst rate {50, 150, 300} Hz x burst length {15, 30} ms

Screen (seed 200, call + silent): a config passes when the call does not cause a
runaway (no activity more than 500 ms after the last hit's burst, and KCs sparse)
AND the kick and snare motor groups fire more after their own hits than in the
silent run. Configs are screened on sparsity/runaway only, never on F1.
Passing configs get the coupling metric on seeds 200-203 (mean + 95 % CI).

"Auditory-only" = the same 6 JO groups, each cut down to its members annotated
subclass "auditory" (the voice <-> group table is unchanged).

Real vs. chosen: wiring and neuron model REAL (unmodified). CHOSEN: tonic drive
0 Hz, the JO set, burst rate and length, every window and threshold here.

Usage (from the repo root):  python -m src.input_strength
"""

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constants as c
from src import encode, kc_gate, sim_pool, transient
from src.provenance import provenance
from src.timing import Timing

CALL = ROOT / "calls" / "wwry.mid"
SCREEN_SEED = 200
SEEDS = [200, 201, 202, 203]
TONIC_DRIVE_HZ = 0.0
JO_SETS = ["all", "auditory"]
RATES_HZ = [50.0, 150.0, 300.0]
DURATIONS_MS = [15.0, 30.0]
WINDOW_MS = 100.0
GATE_VOICES = ("Kick", "Snare")
KC_SPARSE_BELOW_PCT = 20.0
COUPLING_PASS_PCT = 20.0
REFRACTORY_CYCLE_STEPS = 2  # ceil(2.2 ms / 2.0 ms), as flysim computes it at dt 2 ms


def jo_sets():
    """{"all": 6 bodyId lists, "auditory": the same groups cut to members annotated auditory}."""
    z = np.load(ROOT / "build" / "graph.npz", allow_pickle=False)
    auditory = set(z["bodies"][z["subclass"].astype(str) == "auditory"].tolist())
    groups = encode.load_jo_groups()
    return {"all": [g.tolist() for g in groups],
            "auditory": [[int(b) for b in g if int(b) in auditory] for g in groups]}


def encoder(rate_hz, duration_ms):
    return {**encode.load_encoder_config(), "max_rate_hz": rate_hz, "burst_duration_ms": duration_ms}


def grid_steps(hit_t, hit_voice, timing, dt_ms):
    """Simulation step at which each kick/snare hit starts, and at which each no-hit grid step starts."""
    hit_step = np.rint(np.asarray(hit_t) / timing.step_sec).astype(int)
    start = lambda s: int(np.floor(s * timing.step_sec * 1000.0 / dt_ms + 1e-9))
    quiet = [start(s) for s in range(c.CALL_STEPS) if s not in set(hit_step.tolist())]
    return {v: [start(s) for s, hv in zip(hit_step, hit_voice) if hv == v] for v in set(np.asarray(hit_voice).tolist())}, quiet


def measure(call, silent, names, timing, duration_ms):
    """Everything reported for one (config, seed): the call run against its own-seed silent run."""
    dt_ms = call["dt_ms"]
    by_voice, quiet = grid_steps(call["hit_t"], call["hit_voice"], timing, dt_ms)
    k = round(WINDOW_MS / dt_ms)
    motor = {names[v]: transient.own_hit_vs_quiet(call["motor_step_counts"][v], silent["motor_step_counts"][v], steps, quiet, k)
             for v, steps in sorted(by_voice.items())}
    last_input_end = float(np.max(call["hit_t"])) + duration_ms / 1000.0
    persist = transient.persistence(call["all_step_counts"], last_input_end, dt_ms)
    n_kc = call["kc"]["n_kc"]
    return {
        "total_spikes": {"call": call["summary"]["total_spikes_all_neurons"], "silent": silent["summary"]["total_spikes_all_neurons"]},
        "input_spikes_delivered": call["summary"]["input_spikes_delivered"],
        "kc_pct_active_per_16th_mean": call["kc"]["kc_pct_active_per_16th_mean"],
        "kc_pct_active_per_16th_max": call["kc"]["kc_pct_active_per_16th_max"],
        "kc_mean_rate_hz": call["kc"]["kc_mean_rate_hz"],
        "kc_pct_active_per_16th_mean_silent": silent["kc"]["kc_pct_active_per_16th_mean"],
        "pct_all_neurons_above_200hz": call["regions"]["all"]["pct_above_200hz"],
        "pct_all_neurons_fired": call["regions"]["all"]["pct_fired"],
        "motor_rate_hz": {v: call["motor_rate_hz"][v] for v in GATE_VOICES},
        "motor_rate_hz_silent": {v: silent["motor_rate_hz"][v] for v in GATE_VOICES},
        "time_to_runaway_ms": transient.first_time_above_ms(call["kc_step_counts"], n_kc, REFRACTORY_CYCLE_STEPS, dt_ms),
        "after_last_hit": persist,
        "motor_after_own_hits_0_100ms": motor,
    }


def screen_checks(m):
    no_runaway = m["after_last_hit"]["self_sustained"] is False and m["kc_pct_active_per_16th_mean"] < KC_SPARSE_BELOW_PCT
    rises = all(m["motor_after_own_hits_0_100ms"][v]["call_spikes"] > m["motor_after_own_hits_0_100ms"][v]["silent_spikes"]
                for v in GATE_VOICES)
    return {"not_self_sustained": m["after_last_hit"]["self_sustained"] is False,
            "kc_sparse_per_16th": m["kc_pct_active_per_16th_mean"] < KC_SPARSE_BELOW_PCT,
            "no_runaway": no_runaway,
            "motor_rises_after_kick_and_snare_hits": rises}


def run_pairs(configs, seeds, sets, timing):
    """configs: (jo_set, rate, duration). One call run per (config, seed), one silent run per (jo_set, seed)."""
    jobs = [{"call": str(CALL), "seed": seed, "bpm": timing.bpm, "drive_hz": TONIC_DRIVE_HZ, "jo_bodies": sets[js],
             "encoder_config": encoder(rate, dur), "want": ["step_counts"], "tag": ["call", js, rate, dur, seed]}
            for js, rate, dur in configs for seed in seeds]
    jobs += [{"call": None, "seed": seed, "bpm": timing.bpm, "drive_hz": TONIC_DRIVE_HZ, "jo_bodies": sets[js],
              "want": ["step_counts"], "tag": ["silent", js, None, None, seed]}
             for js in sorted({cfg[0] for cfg in configs}) for seed in seeds]
    by = {tuple(r["tag"]): r for r in sim_pool.run_jobs(jobs)}
    names = [v["name"] for v in encode.load_voices()]
    return {(js, rate, dur, seed): measure(by[("call", js, rate, dur, seed)], by[("silent", js, None, None, seed)], names, timing, dur)
            for js, rate, dur in configs for seed in seeds}


def main():
    timing = Timing.for_call(CALL)
    sets = jo_sets()
    t0 = time.time()
    configs = [(js, rate, dur) for js in JO_SETS for rate in RATES_HZ for dur in DURATIONS_MS]
    screen = run_pairs(configs, [SCREEN_SEED], sets, timing)

    rows, passing = [], []
    for cfg in configs:
        m = screen[(*cfg, SCREEN_SEED)]
        checks = screen_checks(m)
        ok = checks["no_runaway"] and checks["motor_rises_after_kick_and_snare_hits"]
        rows.append({"jo_set": cfg[0], "burst_rate_hz": cfg[1], "burst_duration_ms": cfg[2], **m,
                     "checks": checks, "screen_pass": ok})
        if ok:
            passing.append(cfg)

    confirmed = []
    if passing:
        more = run_pairs(passing, [s for s in SEEDS if s != SCREEN_SEED], sets, timing)
        more.update({(*cfg, SCREEN_SEED): screen[(*cfg, SCREEN_SEED)] for cfg in passing})
        for cfg in passing:
            per_seed = [more[(*cfg, s)] for s in SEEDS]
            ci = {v: {key: kc_gate.mean_ci([m["motor_after_own_hits_0_100ms"][v][key] for m in per_seed])
                      for key in ("change_pct_vs_silent", "change_pct_vs_quiet_windows", "call_spikes", "silent_spikes")}
                  for v in GATE_VOICES}
            kc_step = float(np.mean([m["kc_pct_active_per_16th_mean"] for m in per_seed]))
            silent_zero = all(m["motor_after_own_hits_0_100ms"][v]["silent_spikes"] == 0 for m in per_seed for v in GATE_VOICES)
            low = lambda v, ci=ci: ci[v]["change_pct_vs_silent"]["ci95_low"]
            checks = {
                "kc_sparse_per_16th": kc_step < KC_SPARSE_BELOW_PCT,
                "motor_active": all(m["motor_rate_hz"][v] > 0 for m in per_seed for v in GATE_VOICES),
                "kick_ci_low_ge_20": None if low("Kick") is None else low("Kick") >= COUPLING_PASS_PCT,
                "snare_ci_low_ge_20": None if low("Snare") is None else low("Snare") >= COUPLING_PASS_PCT,
            }
            if not checks["kc_sparse_per_16th"] or not checks["motor_active"]:
                gate = False
            elif checks["kick_ci_low_ge_20"] is None or checks["snare_ci_low_ge_20"] is None:
                gate = None
            else:
                gate = bool(checks["kick_ci_low_ge_20"] and checks["snare_ci_low_ge_20"])
            confirmed.append({
                "jo_set": cfg[0], "burst_rate_hz": cfg[1], "burst_duration_ms": cfg[2], "seeds": SEEDS,
                "coupling_ci": ci,
                "kc_pct_active_per_16th_mean_over_seeds": round(kc_step, 2),
                "self_sustained_per_seed": [m["after_last_hit"]["self_sustained"] for m in per_seed],
                "per_seed": {str(s): m for s, m in zip(SEEDS, per_seed)},
                "checks": checks,
                "gate_1a_pass": gate,
                "gate_1a_note": ("the silent run holds 0 motor spikes at 0 Hz tonic drive, so a % change against silent is "
                                 "undefined; left for the human to rule on" if gate is None and silent_zero else None),
            })

    result = {
        "phase": "S",
        "step": "C3",
        **provenance(SEEDS if passing else SCREEN_SEED),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "tonic_drive_hz": TONIC_DRIVE_HZ,
        "decision": "input strength chosen; tonic drive 0 Hz",
        "jo_set_sizes": {name: {"total": sum(len(g) for g in groups), "per_group": [len(g) for g in groups]}
                         for name, groups in sets.items()},
        "screen_seed": SCREEN_SEED,
        "screen_rule": f"no runaway (no spikes later than {transient.PERSIST_AFTER_MS:g} ms after the last hit's burst ends, and "
                       f"KCs active per sixteenth step < {KC_SPARSE_BELOW_PCT:g} %) AND kick and snare motor groups fire more in "
                       f"the {WINDOW_MS:g} ms after their own hits than in the silent run",
        "runaway_definition": f"time_to_runaway_ms = first {REFRACTORY_CYCLE_STEPS * c.DT_MS:g} ms window in which more than "
                              f"{transient.RUNAWAY_KC_PCT:g} % of KCs fire",
        "quiet_windows": "windows of the call run that start on sixteenth steps of the two bars where no voice is hit",
        "ci_method": "mean +- t(0.975, df=3) x sample std / sqrt(4) over seeds 200-203",
        "configs": rows,
        "screen_pass": bool(passing),
        "confirmed": confirmed,
        "gate_1a_pass": (any(x["gate_1a_pass"] for x in confirmed) or
                         (None if any(x["gate_1a_pass"] is None for x in confirmed) else False)),
        "total_wall_time_s": round(time.time() - t0, 1),
    }
    out = ROOT / "results" / "wwry_input_strength.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    print("JO set sizes:", result["jo_set_sizes"])
    for r in rows:
        a, mo = r["after_last_hit"], r["motor_after_own_hits_0_100ms"]
        print(f"{r['jo_set']:8} {r['burst_rate_hz']:5.0f} Hz {r['burst_duration_ms']:4.0f} ms: in {r['input_spikes_delivered']:5} | "
              f"spikes {r['total_spikes']['call']:>9} | KC/16th {r['kc_pct_active_per_16th_mean']:6.2f}% | >200Hz {r['pct_all_neurons_above_200hz']}% | "
              f"motor K {r['motor_rate_hz']['Kick']} S {r['motor_rate_hz']['Snare']} | runaway at {r['time_to_runaway_ms']} ms | "
              f"last spike {a['last_spike_s']} s, late spikes {a['spikes_after']}, sustained {a['self_sustained']} | "
              f"own-hit K {mo['Kick']['call_spikes']} S {mo['Snare']['call_spikes']} vs quiet K {mo['Kick']['change_pct_vs_quiet_windows']} "
              f"S {mo['Snare']['change_pct_vs_quiet_windows']} | {'PASS' if r['screen_pass'] else 'fail'}")
    for x in confirmed:
        print("confirmed", x["jo_set"], x["burst_rate_hz"], x["burst_duration_ms"], json.dumps(x["coupling_ci"]), x["checks"], x["gate_1a_pass"])
    print("screen", "PASS" if passing else "FAIL", "| gate 1a:", result["gate_1a_pass"], f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
