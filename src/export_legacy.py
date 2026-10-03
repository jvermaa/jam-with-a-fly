"""Export one scored run into upstream's viewer format (PLAN.md Step 1.7).

Writes fly_drums_export.json in the schema documented in
docs/legacy_export_schema.md, so the unmodified real-brain.html can play the
fly's answer. The original file is kept as fly_drums_export.original.json.

The hits are the decoded answer exactly as scored: nothing is added, removed or
moved. Known limits of the old viewer (it pulses a motor group on each hit
rather than showing real spikes, it has no open-hat piece) are listed in the
schema doc. Playback tempo comes from jam.bpm.

Usage (from the repo root):  python -m src.export_legacy untrained_call_01_seed0
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flysim import FlyBrain  # upstream, unmodified
from mushroom import MushroomBody  # upstream, unmodified (read-only use: stats)
from src import constants as c
from src import encode, run_fly
from src.provenance import provenance

ANNOTATIONS = ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

# The viewer's six hard-coded channels, in the order of its overlay codes.
VIEWER_CHANNELS = ["kick", "snare", "hihat", "cymbal", "tom", "chord"]
VOICE_TO_CHANNEL = {"Kick": "kick", "Snare": "snare", "Closed hat": "hihat",
                    "Open hat": "hihat", "Tom": "tom", "Crash": "cymbal"}
HIHAT_OPEN_ABOVE = 0.75  # the viewer plays the open hi-hat sound when vel > 0.75


def viewer_velocity(voice_name, midi_vel):
    vel = midi_vel / 127.0
    if voice_name == "Closed hat":
        vel = min(vel, HIHAT_OPEN_ABOVE)
    elif voice_name == "Open hat":
        vel = max(vel, HIHAT_OPEN_ABOVE + 0.01)
    return round(vel, 3)


def neuron_cloud(voices):
    """Same packing as upstream fly_drums_sim.py, with channel_code from our motor groups."""
    ann = pd.read_feather(ANNOTATIONS)
    traced = ann[(ann.status == "Traced") & (ann.statusLabel != "Glia")]
    traced = traced.loc[traced["somaLocation"].notna()]
    coords = np.array(list(traced["somaLocation"]), dtype=np.float32)
    names = sorted(traced["superclass"].dropna().unique().tolist())
    code = {s: i for i, s in enumerate(names)}
    sc_codes = traced["superclass"].map(lambda s: code.get(s, -1)).to_numpy()

    motor = json.loads((ROOT / "build" / "motor_groups.json").read_text())["groups"]
    channel_of_body = {}
    for v in voices:
        ch = VIEWER_CHANNELS.index(VOICE_TO_CHANNEL[v["name"]]) + 1
        for b in motor[v["motor_group"]]:
            channel_of_body[b] = ch
    channel_code = traced["bodyId"].map(lambda b: channel_of_body.get(int(b), 0)).to_numpy()
    return {
        "superclass_names": names,
        "channel_names": VIEWER_CHANNELS,
        "xyz": np.round(coords, 0).astype(int).tolist(),
        "superclass_code": [int(x) for x in sc_codes],
        "channel_code": [int(x) for x in channel_code],
    }


def main():
    run_id = sys.argv[1]
    score = json.loads((ROOT / "results" / f"score_{run_id}.json").read_text())
    batch = json.loads((ROOT / "results" / "batch_untrained.json").read_text())
    batch_row = next(r for r in batch["runs"] if r["run_id"] == run_id)
    voices = encode.load_voices()
    name = {v["idx"]: v["name"] for v in voices}

    hits = [{
        "bar": h["step"] // c.STEPS_PER_BAR,
        "section": "answer",
        "step": h["step"] % c.STEPS_PER_BAR,
        "ch": VOICE_TO_CHANNEL[name[h["voice"]]],
        "vel": viewer_velocity(name[h["voice"]], h["vel"]),
    } for h in score["answer_hits"]]

    fb = FlyBrain(p=run_fly.SimParams())
    mb_stats = MushroomBody(fb).stats()  # untrained: no gain has been changed

    subclasses = {ch: [] for ch in VIEWER_CHANNELS}
    for v in voices:
        subclasses[VOICE_TO_CHANNEL[v["name"]]].append(v["motor_group"])

    export = {
        "meta": {
            "neurons": int(fb.n),
            "edges": int(fb.W.nnz),
            "dt_ms": c.DT_MS,
            "drive_hz": run_fly.DRIVE_HZ,
            "generations": 0,
            "bars": c.CALL_BARS,
            "channel_subclass": {ch: " + ".join(s) if s else "no" for ch, s in subclasses.items()},
            "wall_clock_s": batch_row["runtime_s"],
        },
        "reward_history": [],
        "bar_fitness": [],
        "target_grid": {},
        "hits": hits,
        "mushroom_stats": mb_stats,
        "neuron_cloud": neuron_cloud(voices),
        "jam": {
            **provenance(score["seed"]),
            "run_id": run_id,
            "source_score": f"results/score_{run_id}.json",
            "answer_mid": score["answer_mid"],
            "bpm": c.BPM,
            "f1": score["f1"],
            "baseline_f1": score["baseline_f1"],
            "lag_steps": score["lag_steps"],
            "selection": f"rank {batch_row['rank']} of {batch['n_runs']} untrained runs",
        },
    }
    out = ROOT / "fly_drums_export.json"
    out.write_text(json.dumps(export))

    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB) for {run_id}")
    print("hit list (bar.step  viewer channel  vel   <- voice, MIDI velocity):")
    for h, src in zip(hits, score["answer_hits"]):
        print(f"  {h['bar']}.{h['step']:<2}  {h['ch']:7} {h['vel']:<5}  <- {name[src['voice']]}, {src['vel']}")


if __name__ == "__main__":
    main()
