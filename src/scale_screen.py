"""Global weight scale screen (Phase S, D2): at what overall synaptic strength does a hit stop igniting the network?

All synaptic weights are multiplied by one factor through the wrapper in
src/weight_scale.py (flysim.py is not edited). For each factor, seed 200, no
tonic drive, dt 2 ms:

  (a) silent run of the call's length;
  (b) one kick hit presented to a network at rest, 500 ms simulated;
  (c) the full wwry call.

Screen pass: (b) shows no runaway AND (c) has nonzero motor activity. The scale
is screened on this only; coupling and F1 play no part.

Real vs. chosen: wiring REAL. CHOSEN: the scale factors, tonic drive 0 Hz, the
encoder, the 500 ms single-hit window and the runaway threshold.

Usage (from the repo root):  python -m src.scale_screen
"""

import json
import multiprocessing
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constants as c
from src import encode, reset_transient, sim_pool, transient
from src.provenance import provenance
from src.timing import Timing

CALL = ROOT / "calls" / "wwry.mid"
SEED = 200
SCALES = [0.5, 0.3, 0.2, 0.1, 0.05]
TONIC_DRIVE_HZ = 0.0
SINGLE_HIT_MS = 500.0
SINGLE_HIT_STEP = 0  # the first kick of the call
GATE_VOICES = ("Kick", "Snare")
KC_SPARSE_BELOW_PCT = 20.0


def run_row(r, timing, duration_ms):
    """The reported numbers of one full-length run (silent or call)."""
    n_kc, dt_ms = r["kc"]["n_kc"], r["dt_ms"]
    row = {
        "total_spikes": r["summary"]["total_spikes_all_neurons"],
        "kc_pct_active_per_16th_mean": r["kc"]["kc_pct_active_per_16th_mean"],
        "kc_pct_active_per_16th_max": r["kc"]["kc_pct_active_per_16th_max"],
        "kc_mean_rate_hz": r["kc"]["kc_mean_rate_hz"],
        "pct_all_neurons_above_200hz": r["regions"]["all"]["pct_above_200hz"],
        "pct_all_neurons_fired": r["regions"]["all"]["pct_fired"],
        "time_to_runaway_ms": transient.first_time_above_ms(r["kc_step_counts"], n_kc, 2, dt_ms),
        "motor_rate_hz": r["motor_rate_hz"],
        "motor_spikes": {name: int(r["motor_step_counts"][i].sum()) for i, name in enumerate(r["motor_rate_hz"])},
        "wall_time_s": r["summary"]["runtime_s"],
    }
    if len(r["hit_t"]):
        row["input_spikes_delivered"] = r["summary"]["input_spikes_delivered"]
        row["after_last_hit"] = transient.persistence(r["all_step_counts"], float(np.max(r["hit_t"])) + duration_ms / 1000.0, dt_ms)
    return row


def run_scale(scale):
    """All three runs for one scale, in one worker."""
    from src import run_fly

    sim_pool._init()
    fb = sim_pool._fb
    timing = Timing.for_call(CALL)
    cfg = encode.load_encoder_config()
    base = {"seed": SEED, "bpm": timing.bpm, "drive_hz": TONIC_DRIVE_HZ, "weight_scale": scale, "want": ["step_counts"]}
    silent = sim_pool.run_job({**base, "call": None})
    call = sim_pool.run_job({**base, "call": str(CALL)})

    # (b) one hit from rest; the weights are still scaled (the wrapper wrote them into fb.wdata).
    dt_ms = float(fb.p.dt)
    n_steps = round(SINGLE_HIT_MS / dt_ms)
    jo_bodies = encode.load_jo_groups()
    voices, jo, motor = run_fly.load_groups(fb, jo_bodies)
    hits = encode.read_hits(CALL, voices, timing)
    enc = encode.encode(hits, jo_bodies, config=cfg, seed=SEED, voices=voices)
    sel, rel = reset_transient.step_inputs(enc["spike_t"], SINGLE_HIT_STEP, timing.step_sec, cfg["burst_duration_ms"] / 1000.0)
    input_step = np.floor(rel * 1000.0 / dt_ms + 1e-9).astype(np.int64)
    input_neuron = np.array([fb.body_to_i[int(b)] for b in enc["spike_body"][sel]], dtype=np.int64)
    t0 = time.time()
    spike_step, spike_neuron = run_fly.simulate(fb, np.concatenate(jo), input_step, input_neuron, SEED, n_steps, TONIC_DRIVE_HZ)
    wall = time.time() - t0
    kc = sim_pool._kc
    is_kc = np.zeros(fb.n, dtype=bool)
    is_kc[kc] = True
    is_jo = np.zeros(fb.n, dtype=bool)
    is_jo[np.concatenate(jo)] = True
    kc_sel = is_kc[spike_neuron]
    motor_counts = run_fly.group_step_counts(spike_step, spike_neuron, motor, fb.n, n_steps)
    all_counts = np.bincount(spike_step, minlength=n_steps)
    single = {
        "voice_hit": voices[next(h for h in hits if round(h["t"] / timing.step_sec) == SINGLE_HIT_STEP)["voice"]]["name"],
        "sim_ms": SINGLE_HIT_MS,
        "input_spikes_delivered": len(set(zip(input_step.tolist(), input_neuron.tolist()))),
        "total_spikes": len(spike_step),
        "spikes_outside_jo": int((~is_jo[spike_neuron]).sum()),
        "neurons_fired": len(np.unique(spike_neuron)),
        "pct_kc_fired": round(100.0 * len(np.unique(spike_neuron[kc_sel])) / len(kc), 2),
        "time_to_runaway_ms": transient.first_time_above_ms(np.bincount(spike_step[kc_sel], minlength=n_steps), len(kc), int(fb.refr_steps), dt_ms),
        "last_spike_ms": None if not len(spike_step) else float(spike_step.max() * dt_ms),
        "spikes_in_last_100ms": int(all_counts[-round(100.0 / dt_ms):].sum()),
        "motor_spikes": {v["name"]: int(motor_counts[i].sum()) for i, v in enumerate(voices)},
        "wall_time_s": round(wall, 2),
    }
    return {"scale": scale, "silent_run": run_row(silent, timing, cfg["burst_duration_ms"]),
            "single_kick_from_rest": single, "call_run": run_row(call, timing, cfg["burst_duration_ms"])}


def checks_for(level):
    single, call = level["single_kick_from_rest"], level["call_run"]
    return {
        "no_runaway_after_single_hit": single["time_to_runaway_ms"] is None,
        "call_evokes_motor_activity": sum(call["motor_spikes"].values()) > 0,
        # Reported, not part of the screen rule:
        "info_call_kick_and_snare_motor_both_fire": all(call["motor_spikes"][v] > 0 for v in GATE_VOICES),
        "info_call_run_no_runaway": call["time_to_runaway_ms"] is None,
        "info_call_run_kc_sparse": call["kc_pct_active_per_16th_mean"] < KC_SPARSE_BELOW_PCT,
        "info_call_run_not_self_sustained": call["after_last_hit"]["self_sustained"] is False,
    }


def main(scales=SCALES, out_name="wwry_scale_screen.json"):
    timing = Timing.for_call(CALL)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(sim_pool.PERFORMANCE_CORES, len(scales))) as pool:
        levels = pool.map(run_scale, scales, chunksize=1)
    for lv in levels:
        lv["mv_per_synapse"] = round(0.275 * lv["scale"], 6)
        lv["checks"] = checks_for(lv)
        lv["screen_pass"] = lv["checks"]["no_runaway_after_single_hit"] and lv["checks"]["call_evokes_motor_activity"]

    result = {
        "phase": "S",
        "step": "D2",
        **provenance(SEED),
        "call": str(CALL.relative_to(ROOT)),
        "bpm": timing.bpm,
        "dt_ms": c.DT_MS,
        "tonic_drive_hz": TONIC_DRIVE_HZ,
        "encoder": encode.load_encoder_config(),
        "jo_set": "all 554 (build/jo_groups.json)",
        "modelling_choice": "all synaptic weights multiplied by one global factor in a wrapper (src/weight_scale.py); CHOSEN, "
                            "not stated by the connectome",
        "d1_implied_scale": None,
        "d1_implied_scale_note": "not computed: no FlyWire numbers available locally (results/input_ratio.json)",
        "screen_rule": "no runaway after a single kick from rest (never more than "
                       f"{transient.RUNAWAY_KC_PCT:g} % of KCs firing within one 4 ms refractory cycle, {SINGLE_HIT_MS:g} ms simulated) "
                       "AND the full call evokes nonzero motor spikes (any of the 6 motor groups)",
        "levels": levels,
        "passing_scales": [lv["scale"] for lv in levels if lv["screen_pass"]],
        "screen_pass": any(lv["screen_pass"] for lv in levels),
        "total_wall_time_s": round(time.time() - t0, 1),
    }
    out = ROOT / "results" / out_name
    out.write_text(json.dumps(result, indent=1) + "\n")

    for lv in levels:
        s, cl, sl = lv["single_kick_from_rest"], lv["call_run"], lv["silent_run"]
        print(f"scale {lv['scale']}: silent {sl['total_spikes']} | single kick: spikes {s['total_spikes']} (outside JO {s['spikes_outside_jo']}), "
              f"KC fired {s['pct_kc_fired']}%, runaway {s['time_to_runaway_ms']}, last spike {s['last_spike_ms']} ms, motor {s['motor_spikes']}")
        print(f"     call: spikes {cl['total_spikes']} | KC/16th {cl['kc_pct_active_per_16th_mean']}% (max {cl['kc_pct_active_per_16th_max']}) | "
              f">200Hz {cl['pct_all_neurons_above_200hz']}% | fired {cl['pct_all_neurons_fired']}% | runaway {cl['time_to_runaway_ms']} | "
              f"late spikes {cl['after_last_hit']['spikes_after']} | motor Hz K {cl['motor_rate_hz']['Kick']} S {cl['motor_rate_hz']['Snare']} | "
              f"motor spikes {cl['motor_spikes']} | {cl['wall_time_s']} s | {'PASS' if lv['screen_pass'] else 'fail'}")
    print("passing:", result["passing_scales"], f"-> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
