"""Run the fly: play a call into its antennal neurons and record what comes out.

PLAN.md Step 1.4 (play-along). Simulates the call plus a short tail with the
encoder's input spikes, and a silent baseline with the same seed and no input.
Call and tail lengths follow the call's tempo (src/timing.py; 4.0 s + 0.5 s at
the default 120 BPM).

Real vs. chosen
  REAL:   the wiring, the neuron model and its parameters (upstream flysim.py,
          imported and not edited), and the neurons we stimulate and read.
  CHOSEN: * the input spikes (src/encode.py);
          * the tonic drive onto every vnc_intrinsic neuron, kept exactly as in
            upstream fly_drums_sim.py (DRIVE_HZ). Without it the motor side is
            silent; it is an artificial "wake-up" current, not something the
            connectome provides;
          * the 2.0 ms timestep (validated default for this model is 0.2 ms).

How input is delivered without editing the simulator: flysim's only input is a
per-neuron Poisson rate. A rate of 1000/dt Hz makes the per-step probability
exactly 1, so a neuron given that rate for one step fires in that step. Each
encoder spike is delivered that way, in the step its time falls in.

Why the silent baseline is a fair comparison: every step of every run draws the
same number of random values (tonic neurons + all JO neurons, the JO ones at
rate 0 when there is no input). The call run and the silent run therefore use
identical random numbers for the tonic drive, and any difference between them
is caused by the call.

Usage (from the repo root):  python -m src.run_fly calls/call_01.mid --seed 0
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fly_drums_sim import DRIVE_HZ  # upstream's tonic drive rate, unmodified
from flysim import FlyBrain, Params  # upstream, unmodified
from src import constants as c
from src import encode
from src.provenance import provenance
from src.timing import DEFAULT, Timing

RESULTS = ROOT / "results"
MOTOR_ORDER = None  # filled from config/voices.json: motor group of voice 0..5


class SimParams(Params):
    dt = c.DT_MS


def total_steps(timing=DEFAULT, dt_ms=c.DT_MS):
    # Rounded up, so the last sixteenth step of the tail is always simulated in full.
    return math.ceil(timing.sim_sec * 1000.0 / dt_ms - 1e-9)


def call_steps(timing=DEFAULT, dt_ms=c.DT_MS):
    return round(timing.call_sec * 1000.0 / dt_ms)


def load_groups(fb, jo_bodies=None):
    """Index arrays (into the simulator) for the 6 JO groups and 6 motor groups, in voice order.

    jo_bodies: 6 bodyId arrays to use instead of build/jo_groups.json (Phase S input-coupling grid).
    """
    voices = encode.load_voices()
    jo_bodies = encode.load_jo_groups() if jo_bodies is None else jo_bodies
    motor_bodies = json.loads((ROOT / "build" / "motor_groups.json").read_text())["groups"]
    jo = [np.array([fb.body_to_i[int(b)] for b in jo_bodies[v["jo_group"]]], dtype=np.int64) for v in voices]
    motor = [np.array([fb.body_to_i[int(b)] for b in motor_bodies[v["motor_group"]]], dtype=np.int64) for v in voices]
    return voices, jo, motor


def simulate(fb, jo_all, input_step, input_neuron, seed, n_steps=None, drive_hz=DRIVE_HZ):
    """Simulate n_steps steps (default total_steps()). input_step/input_neuron: one entry per input spike to deliver.

    Returns (spike_step, spike_neuron) for every spike of every neuron.
    """
    n_steps = total_steps() if n_steps is None else n_steps
    vnc_int = fb.where(superclass="vnc_intrinsic")
    tonic_key = tuple(vnc_int.tolist())
    jo_key = tuple(jo_all.tolist())
    pos_in_jo = {int(n): i for i, n in enumerate(jo_all)}
    force_hz = 1000.0 / fb.p.dt  # per-step probability exactly 1
    zero = np.zeros(len(jo_all), dtype=np.float32)

    by_step = {}
    for s, n in zip(input_step.tolist(), input_neuron.tolist()):
        by_step.setdefault(s, set()).add(pos_in_jo[n])

    spike_steps, spike_neurons = [], []
    state, step = None, 0
    while step < n_steps:
        if step in by_step:
            rates = zero.copy()
            rates[sorted(by_step[step])] = force_hz
            length = 1
        else:
            rates = zero
            nxt = min((s for s in by_step if s > step), default=n_steps)
            length = min(nxt, n_steps) - step
        out = fb.run({tonic_key: drive_hz, jo_key: rates}, steps=length, seed=seed, spike_log=True, state=state)
        state = out["_state"]
        for k, fired in enumerate(out["_spikes"]):
            if len(fired):
                spike_steps.append(np.full(len(fired), step + k, dtype=np.int32))
                spike_neurons.append(fired.astype(np.int32))
        step += length

    if spike_steps:
        return np.concatenate(spike_steps), np.concatenate(spike_neurons)
    return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)


def group_step_counts(spike_step, spike_neuron, groups, n_neurons, n_steps=None):
    """(6, n_steps) array: spikes per simulation step for each group."""
    n_steps = total_steps() if n_steps is None else n_steps
    counts = np.zeros((len(groups), n_steps), dtype=np.int32)
    for g, idx in enumerate(groups):
        member = np.zeros(n_neurons, dtype=bool)
        member[idx] = True
        counts[g] = np.bincount(spike_step[member[spike_neuron]], minlength=n_steps)
    return counts


def run(fb, call_path, seed, timing=None, encoder_config=None, jo_bodies=None, drive_hz=DRIVE_HZ):
    """One run. call_path=None is the silent baseline. Returns (arrays for the npz, summary dict).

    timing:         tempo; default is the call's configured tempo (120 BPM for a silent run,
                    so pass the call's timing to get a silent baseline of the same length).
    encoder_config: overrides config/encoder.json.
    jo_bodies:      6 bodyId arrays overriding build/jo_groups.json.
    drive_hz:       tonic drive onto vnc_intrinsic (CHOSEN); default is upstream's DRIVE_HZ.
    """
    timing = timing or Timing.for_call(call_path)
    dt_ms = float(fb.p.dt)  # the brain's own timestep (c.DT_MS unless the caller built it otherwise)
    n_steps = total_steps(timing, dt_ms)
    jo_bodies = encode.load_jo_groups() if jo_bodies is None else jo_bodies
    voices, jo, motor = load_groups(fb, jo_bodies)
    jo_all = np.concatenate(jo)

    if call_path is None:
        hits, input_step, input_neuron = [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        hits = encode.read_hits(call_path, voices, timing)
        enc = encode.encode(hits, jo_bodies, config=encoder_config, seed=seed, voices=voices)
        input_step = np.floor(enc["spike_t"] * 1000.0 / dt_ms + 1e-9).astype(np.int64)
        input_neuron = np.array([fb.body_to_i[int(b)] for b in enc["spike_body"]], dtype=np.int64)
        keep = input_step < n_steps
        input_step, input_neuron = input_step[keep], input_neuron[keep]

    t0 = time.time()
    spike_step, spike_neuron = simulate(fb, jo_all, input_step, input_neuron, seed, n_steps, drive_hz)
    runtime = time.time() - t0

    jo_counts = group_step_counts(spike_step, spike_neuron, jo, fb.n, n_steps)
    motor_counts = group_step_counts(spike_step, spike_neuron, motor, fb.n, n_steps)
    in_call = call_steps(timing, dt_ms)

    arrays = {
        "spike_step": spike_step,            # every spike of every neuron: simulation step ...
        "spike_neuron": spike_neuron,        # ... and simulator neuron index (see `bodies`)
        "bodies": fb.bodies,                 # neuron index -> bodyId
        "jo_step_counts": jo_counts,         # (6 voices, steps)
        "motor_step_counts": motor_counts,   # (6 voices, steps)
        "hit_t": np.array([h["t"] for h in hits], dtype=np.float64),
        "hit_voice": np.array([h["voice"] for h in hits], dtype=np.int16),
        "hit_vel": np.array([h["vel"] for h in hits], dtype=np.int16),
        "dt_ms": np.float64(dt_ms),
        "bpm": np.float64(timing.bpm),
        "seed": np.int64(seed),
    }
    summary = {
        "call": None if call_path is None else str(pathlib.Path(call_path).resolve().relative_to(ROOT)),
        "seed": seed,
        "runtime_s": round(runtime, 2),
        "bpm": timing.bpm,
        "tonic_drive_hz": drive_hz,
        "dt_ms": dt_ms,
        "sim_seconds": n_steps * dt_ms / 1000.0,
        "steps": n_steps,
        "n_hits": len(hits),
        "input_spikes_delivered": len(set(zip(input_step.tolist(), input_neuron.tolist()))),
        "total_spikes_all_neurons": len(spike_step),
        "jo_spikes_during_call": {v["name"]: int(jo_counts[i, :in_call].sum()) for i, v in enumerate(voices)},
        "motor_spikes_during_call": {v["name"]: int(motor_counts[i, :in_call].sum()) for i, v in enumerate(voices)},
        "motor_spikes_in_tail": {v["name"]: int(motor_counts[i, in_call:].sum()) for i, v in enumerate(voices)},
    }
    return arrays, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fb = FlyBrain(p=SimParams())
    timing = Timing.for_call(args.call)
    call_arrays, call_summary = run(fb, args.call, args.seed, timing)
    silent_arrays, silent_summary = run(fb, None, args.seed, timing)

    call_npz = RESULTS / f"run_{args.call.stem}_seed{args.seed}.npz"
    silent_npz = RESULTS / f"silent_seed{args.seed}.npz"
    np.savez_compressed(call_npz, **call_arrays)
    np.savez_compressed(silent_npz, **silent_arrays)

    a, b = call_summary["motor_spikes_during_call"], silent_summary["motor_spikes_during_call"]
    differs = a != b or not np.array_equal(call_arrays["motor_step_counts"], silent_arrays["motor_step_counts"])
    meta = {
        "step": "1.4",
        **provenance(args.seed),
        "dt_ms": c.DT_MS,
        "tonic_drive": {"target": "superclass vnc_intrinsic", "rate_hz": DRIVE_HZ, "note": "chosen, kept from upstream"},
        "call_run": {**call_summary, "npz": str(call_npz.relative_to(ROOT))},
        "silent_run": {**silent_summary, "npz": str(silent_npz.relative_to(ROOT))},
        "motor_step_counts_identical": bool(np.array_equal(call_arrays["motor_step_counts"], silent_arrays["motor_step_counts"])),
        "pass": bool(differs),
    }
    (RESULTS / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"call run:   {call_summary['runtime_s']} s, motor spikes during call {a}")
    print(f"silent run: {silent_summary['runtime_s']} s, motor spikes during call {b}")
    print("pass (motor output differs from silent):", meta["pass"])
    if not meta["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
