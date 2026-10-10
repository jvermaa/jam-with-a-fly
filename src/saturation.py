"""Gate 1a (Phase S): are the Kenyon cells saturated, and does less tonic drive fix it?

At upstream's tonic drive every Kenyon cell (KC) fires at the model's ceiling,
with or without sound (results/wwry_profile.json). A saturated mushroom body
carries no information about the call, and the learning rule's "which KC was
just active" has nothing to select. This script:

  1. reads from the built graph how the APL neuron (the mushroom body's
     feedback inhibitor) is signed, and what excites and inhibits the KCs;
  2. runs calls/wwry.mid and a silent baseline at 100 %, 50 %, 25 % and 10 % of
     upstream's tonic drive and reports KC activity, motor rates and the
     per-hit coupling metric (src/coupling.py).

PASS: some drive level where fewer than 20 % of KCs fire at all, the kick and
snare motor groups still fire, and coupling is >= +20 % for both.

Real vs. chosen: the wiring and signs are REAL as built by upstream
build_graph.py (not edited). The tonic drive, and scaling it, are CHOSEN.

Usage (from the repo root):  python -m src.saturation
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fly_drums_sim import DRIVE_HZ  # upstream's tonic drive rate, unmodified
from flysim import FlyBrain  # upstream, unmodified
from src import constants as c
from src import coupling, encode, sim_pool
from src.provenance import provenance
from src.timing import Timing

SEED = 0
CALL = ROOT / "calls" / "wwry.mid"
DRIVE_FRACTIONS = [1.0, 0.5, 0.25, 0.1]
KC_SPARSE_BELOW_PCT = 20.0
COUPLING_PASS_PCT = 20.0


def graph_facts(fb):
    """Sign and transmitter of APL, and what drives the Kenyon cells, as built in build/graph.npz."""
    is_kc = np.zeros(fb.n, dtype=bool)
    kc = fb.where(type_re=r"^KC")
    is_kc[kc] = True
    W = fb.W  # CSC: column = presynaptic neuron
    apl = []
    for i in fb.where(type_re=r"^APL$"):
        a, b = fb.indptr[i], fb.indptr[i + 1]
        tgt, w = fb.indices[a:b], fb.wdata[a:b]
        apl.append({
            "bodyId": int(fb.bodies[i]),
            "type": str(fb.types[i]),
            "transmitter_in_graph": str(fb.nt[i]),
            "outgoing_edges": len(w),
            "excitatory_edges": int((w > 0).sum()),
            "inhibitory_edges": int((w < 0).sum()),
            "edges_onto_kc": int(is_kc[tgt].sum()),
            "mv_onto_kc_per_spike_total": round(float(w[is_kc[tgt]].sum()), 1),
        })
    onto_kc = W.tocsr()[kc, :].tocsc()
    data, pre = onto_kc.data, np.repeat(np.arange(fb.n), np.diff(onto_kc.indptr))
    from_kc = is_kc[pre]
    return {
        "apl": apl,
        "kc_transmitter_in_graph": sorted(set(fb.nt[kc].tolist())),
        "kc_input_mv_per_presynaptic_spike": {
            "excitatory_total": round(float(data[data > 0].sum()), 0),
            "excitatory_from_other_kcs": round(float(data[(data > 0) & from_kc].sum()), 0),
            "inhibitory_total": round(float(data[data < 0].sum()), 0),
            "inhibitory_from_apl": round(float(sum(a["mv_onto_kc_per_spike_total"] for a in apl)), 0),
        },
        "kc_to_kc_edges": int(from_kc.sum()),
        "threshold_above_rest_mv": float(fb.p.v_thresh - fb.p.v_rest),
    }


def main():
    timing = Timing.for_call(CALL)
    voices = encode.load_voices()
    names = [v["name"] for v in voices]

    jobs = []
    for frac in DRIVE_FRACTIONS:
        for call in (str(CALL), None):
            jobs.append({"call": call, "seed": SEED, "bpm": timing.bpm, "drive_hz": DRIVE_HZ * frac,
                         "tag": [frac, call is not None]})
    results = sim_pool.run_jobs(jobs)
    by = {tuple(r["tag"]): r for r in results}

    levels = []
    for frac in DRIVE_FRACTIONS:
        call, silent = by[(frac, True)], by[(frac, False)]
        cp = coupling.coupling(call["motor_step_counts"], silent["motor_step_counts"],
                               call["hit_t"], call["hit_voice"], names, c.DT_MS)
        kick, snare = cp["Kick"]["change_pct"], cp["Snare"]["change_pct"]
        checks = {
            "kc_sparse": call["kc"]["kc_pct_fired_at_least_once"] < KC_SPARSE_BELOW_PCT,
            "motor_active": call["motor_rate_hz"]["Kick"] > 0 and call["motor_rate_hz"]["Snare"] > 0,
            "coupling_kick_and_snare": kick is not None and snare is not None
                                       and kick >= COUPLING_PASS_PCT and snare >= COUPLING_PASS_PCT,
        }
        levels.append({
            "drive_fraction": frac,
            "tonic_drive_hz": DRIVE_HZ * frac,
            "kc_call_run": call["kc"],
            "kc_silent_run": silent["kc"],
            "motor_rate_hz_call_run": call["motor_rate_hz"],
            "motor_rate_hz_silent_run": silent["motor_rate_hz"],
            "total_spikes_call_run": call["summary"]["total_spikes_all_neurons"],
            "jo_spikes_during_call": call["summary"]["jo_spikes_during_call"],
            "coupling_0_100ms": cp,
            "checks": checks,
            "pass": all(checks.values()),
            "runtime_s": {"call": call["summary"]["runtime_s"], "silent": silent["summary"]["runtime_s"]},
        })

    fb = FlyBrain(p=sim_pool_params())
    result = {
        "phase": "S",
        "gate": "1a",
        **provenance(SEED),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "upstream_drive_hz": DRIVE_HZ,
        "kc_pct_definition": "kc_pct_fired_at_least_once = share of the KCs that spike at least once in the "
                             "whole run (used for the pass check); kc_pct_active_per_16th_* = share that spike "
                             "within one sixteenth step",
        "pass_rule": f"KC fired-at-least-once < {KC_SPARSE_BELOW_PCT} %, kick and snare motor groups firing, "
                     f"coupling >= +{COUPLING_PASS_PCT} % for kick and snare",
        "graph_facts": graph_facts(fb),
        "levels": levels,
        "pass": any(lv["pass"] for lv in levels),
    }
    out = ROOT / "results" / "wwry_saturation.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result["graph_facts"], indent=1))
    for lv in levels:
        k = lv["kc_call_run"]
        print(f"drive {lv['tonic_drive_hz']:5.2f} Hz: KC fired {k['kc_pct_fired_at_least_once']}% "
              f"(per 16th {k['kc_pct_active_per_16th_mean']}%), {k['kc_mean_rate_hz']} Hz | "
              f"motor Hz {lv['motor_rate_hz_call_run']} | coupling kick {lv['coupling_0_100ms']['Kick']['change_pct']} "
              f"snare {lv['coupling_0_100ms']['Snare']['change_pct']} | {lv['checks']}")
    print("PASS" if result["pass"] else "FAIL", f"-> wrote {out.relative_to(ROOT)}")


def sim_pool_params():
    from src import run_fly

    return run_fly.SimParams()


if __name__ == "__main__":
    main()
