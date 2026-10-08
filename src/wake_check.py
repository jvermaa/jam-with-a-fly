"""Gate 1a, step B (Phase S): does the network need the tonic drive, and where is it saturated?

Runs calls/wwry.mid and a silent baseline, seed 0, at tonic drive 0 Hz, 1 Hz and
upstream's 45 Hz, on the graph as built (no KC->KC scaling). Reports total
spikes, Kenyon cell activity, motor rates, whether the call alone wakes
anything, and for every top-level region the share of neurons firing above
200 Hz.

Real vs. chosen: the wiring is REAL and unmodified here. The tonic drive and
its level are CHOSEN.

Usage (from the repo root):  python -m src.wake_check
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fly_drums_sim import DRIVE_HZ  # upstream's tonic drive rate, unmodified
from src import constants as c
from src import encode, sim_pool
from src.provenance import provenance
from src.timing import Timing

SEED = 0
CALL = ROOT / "calls" / "wwry.mid"
DRIVES_HZ = [0.0, 1.0, DRIVE_HZ]


def main():
    timing = Timing.for_call(CALL)
    jobs = [{"call": call, "seed": SEED, "bpm": timing.bpm, "drive_hz": hz, "tag": [hz, call is not None]}
            for hz in DRIVES_HZ for call in (str(CALL), None)]
    by = {tuple(r["tag"]): r for r in sim_pool.run_jobs(jobs)}

    levels = []
    for hz in DRIVES_HZ:
        call, silent = by[(hz, True)], by[(hz, False)]
        jo_in = sum(call["summary"]["jo_spikes_during_call"].values())
        row = {"tonic_drive_hz": hz}
        for name, r in (("call_run", call), ("silent_run", silent)):
            row[name] = {
                "total_spikes": r["summary"]["total_spikes_all_neurons"],
                "kc": r["kc"],
                "motor_rate_hz": r["motor_rate_hz"],
                "motor_spikes": {k: r["summary"]["motor_spikes_during_call"][k] + r["summary"]["motor_spikes_in_tail"][k]
                                 for k in r["motor_rate_hz"]},
                "regions": r["regions"],
            }
        row["jo_input_spikes_during_call"] = jo_in
        row["spikes_beyond_jo_input_call_minus_silent"] = (call["summary"]["total_spikes_all_neurons"]
                                                           - silent["summary"]["total_spikes_all_neurons"] - jo_in)
        levels.append(row)

    zero = levels[0]
    result = {
        "phase": "S",
        "gate": "1a-B",
        **provenance(SEED),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "kc_kc_scale": 1.0,
        "region_definition": sim_pool.REGION_OF_SUPERCLASS,
        "region_note": "kc = type name starting KC; sensory = any superclass containing 'sensory', plus ENS; "
                       "other = no superclass annotated",
        "saturated_above_hz": sim_pool.SATURATED_ABOVE_HZ,
        "levels": levels,
        "call_alone_wakes_motor": any(v > 0 for v in zero["call_run"]["motor_spikes"].values()),
        "call_alone_neurons_fired_pct": zero["call_run"]["regions"]["all"]["pct_fired"],
    }
    out = ROOT / "results" / "wwry_wake_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    for lv in levels:
        for name in ("call_run", "silent_run"):
            r = lv[name]
            print(f"drive {lv['tonic_drive_hz']:4.1f} Hz {name:10}: total {r['total_spikes']:>9} | KC fired "
                  f"{r['kc']['kc_pct_fired_at_least_once']}% per16th {r['kc']['kc_pct_active_per_16th_mean']}% "
                  f"{r['kc']['kc_mean_rate_hz']} Hz | motor Hz {r['motor_rate_hz']}")
            print("     >200Hz % by region:", {k: v["pct_above_200hz"] for k, v in r["regions"].items()},
                  "| fired %:", {k: v["pct_fired"] for k, v in r["regions"].items()})
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
