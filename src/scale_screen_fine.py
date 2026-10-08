"""Finer global weight scale screen (Phase S, D2b).

Same three runs per scale as src/scale_screen.py (silent, one kick from rest,
full call; no tonic drive, dt 2 ms), on scales 0.12-0.18, with the D2 scales
0.1 / 0.2 / 0.3 rerun alongside so one table holds them all.

Screen pass (stricter than D2): no runaway after a single hit AND the kick and
the snare motor groups both spike during the call AND activity ends within
500 ms of the last hit of the call. If two or more scales pass on seed 200 they
are rerun on seeds 201-203.

The scale is screened on this only; coupling and F1 play no part.

Real vs. chosen: wiring REAL. CHOSEN: the scale factors and every threshold here.

Usage (from the repo root):  python -m src.scale_screen_fine
"""

import json
import multiprocessing
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constants as c
from src import encode, scale_screen, sim_pool, transient
from src.provenance import provenance
from src.timing import Timing

NEW_SCALES = [0.12, 0.14, 0.16, 0.18]
D2_SCALES = [0.1, 0.2, 0.3]
SCREEN_SEED = 200
MORE_SEEDS = [201, 202, 203]
GATE_VOICES = ("Kick", "Snare")


def checks_for(level):
    single, call = level["single_kick_from_rest"], level["call_run"]
    return {
        "no_runaway_after_single_hit": single["time_to_runaway_ms"] is None,
        "kick_and_snare_motor_spikes_on_call": all(call["motor_spikes"][v] > 0 for v in GATE_VOICES),
        "activity_ends_within_500ms_of_last_hit": call["after_last_hit_onset"]["self_sustained"] is False,
    }


def finish(level):
    level["mv_per_synapse"] = round(0.275 * level["scale"], 6)
    call = level["call_run"]
    last = call["after_last_hit_onset"]["last_spike_s"]
    level["call_activity_ends_s"] = last
    level["call_activity_ends_ms_after_last_hit"] = None if last is None else round(1000.0 * (last - call["last_hit_onset_s"]), 1)
    level["call_spikes_later_than_500ms_after_last_hit"] = call["after_last_hit_onset"]["spikes_after"]
    level["checks"] = checks_for(level)
    level["screen_pass"] = all(level["checks"].values())
    return level


def run(args):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(sim_pool.PERFORMANCE_CORES, len(args))) as pool:
        return [finish(lv) for lv in pool.map(scale_screen.run_scale, args, chunksize=1)]


def main():
    timing = Timing.for_call(scale_screen.CALL)
    t0 = time.time()
    scales = sorted(NEW_SCALES + D2_SCALES)
    levels = run([(s, SCREEN_SEED) for s in scales])
    passing = [lv["scale"] for lv in levels if lv["screen_pass"]]

    more = []
    if len(passing) >= 2:
        more = run([(s, seed) for s in passing for seed in MORE_SEEDS])

    result = {
        "phase": "S",
        "step": "D2b",
        **provenance([SCREEN_SEED, *MORE_SEEDS] if more else SCREEN_SEED),
        "call": str(scale_screen.CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "tonic_drive_hz": scale_screen.TONIC_DRIVE_HZ,
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "modelling_choice": "all synaptic weights multiplied by one global factor in a wrapper (src/weight_scale.py); CHOSEN",
        "screen_rule": "no runaway after a single kick from rest (never more than "
                       f"{transient.RUNAWAY_KC_PCT:g} % of KCs within one 4 ms window, {scale_screen.SINGLE_HIT_MS:g} ms simulated) AND "
                       "kick-group and snare-group motor spikes both > 0 during the call run AND no spike of any neuron later than "
                       "500 ms after the onset of the call's last hit",
        "new_scales": NEW_SCALES,
        "rerun_d2_scales": D2_SCALES,
        "screen_seed": SCREEN_SEED,
        "levels_screen_seed": levels,
        "passing_scales_screen_seed": passing,
        "more_seeds": MORE_SEEDS if more else [],
        "levels_more_seeds": more,
        "passing_on_all_seeds": [s for s in passing if all(lv["screen_pass"] for lv in more if lv["scale"] == s)] if more else passing,
        "total_wall_time_s": round(time.time() - t0, 1),
    }
    out = ROOT / "results" / "wwry_scale_screen_fine.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    for lv in levels + more:
        s, cl = lv["single_kick_from_rest"], lv["call_run"]
        print(f"scale {lv['scale']:.2f} seed {lv['seed']}: silent {lv['silent_run']['total_spikes']} | kick: {s['total_spikes']} sp, KC {s['pct_kc_fired']}%, "
              f"runaway {s['time_to_runaway_ms']}, last {s['last_spike_ms']} ms | call: {cl['total_spikes']} sp, KC/16th {cl['kc_pct_active_per_16th_mean']}% "
              f"(max {cl['kc_pct_active_per_16th_max']}), >200Hz {cl['pct_all_neurons_above_200hz']}%, runaway {cl['time_to_runaway_ms']}, "
              f"motor Hz K {cl['motor_rate_hz']['Kick']} S {cl['motor_rate_hz']['Snare']}, spikes K {cl['motor_spikes']['Kick']} S {cl['motor_spikes']['Snare']}, "
              f"late {lv['call_spikes_later_than_500ms_after_last_hit']}, ends {lv['call_activity_ends_ms_after_last_hit']} ms after last hit, "
              f"{cl['wall_time_s']} s | {lv['checks']} {'PASS' if lv['screen_pass'] else 'fail'}")
    print("passing seed 200:", passing, "| on all seeds:", result["passing_on_all_seeds"], f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
