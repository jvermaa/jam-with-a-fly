"""Encoder: MIDI drum hits -> input spike trains on the fly's antennal neurons.

PLAN.md Step 1.3. For each hit at time t on voice i, every neuron in JO group i
gets a Poisson spike burst: duration 30 ms, rate = 150 Hz x (velocity / 127)
(defaults in config/encoder.json).

Real vs. chosen
  REAL:   the neurons that receive the spikes (Johnston's organ bodies from the
          MaleCNS annotations, build/jo_groups.json).
  CHOSEN: everything else here. A real antenna does not get a private burst per
          drum; "voice i -> JO group i", the burst length and the rate are our
          encoding, picked so each voice has its own input channel.

Spike times are in seconds, in continuous time, and do not depend on the
simulator's timestep. src/run_fly.py is what bins them into simulation steps.

Usage (from the repo root):  python -m src.encode calls/call_01.mid
"""

import json
import pathlib
import sys

import mido
import numpy as np

from src import constants as c

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SEED = 0

# Categorical colours, one per voice, in fixed voice order.
VOICE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def load_voices():
    return json.loads((ROOT / "config" / "voices.json").read_text())["voices"]


def load_encoder_config():
    cfg = json.loads((ROOT / "config" / "encoder.json").read_text())
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def load_jo_groups():
    """List of 6 sorted bodyId arrays, index = JO group."""
    groups = json.loads((ROOT / "build" / "jo_groups.json").read_text())["groups"]
    return [np.array(g["bodyIds"], dtype=np.int64) for g in sorted(groups, key=lambda g: g["group"])]


def read_hits(path, voices=None):
    """Read a call file -> list of {"t": seconds, "voice": idx, "vel": 1..127}, sorted by time.

    Notes that are not in the voice table are ignored here; src/check_calls.py is
    what rejects files containing them.
    """
    voices = voices or load_voices()
    note_to_voice = {v["midi_note"]: v["idx"] for v in voices}
    mid = mido.MidiFile(path)
    hits = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in note_to_voice:
                # Fixed 120 BPM (PLAN.md locked decision), so ticks map straight to seconds.
                t = tick / mid.ticks_per_beat * 60.0 / c.BPM
                hits.append({"t": t, "voice": note_to_voice[msg.note], "vel": int(msg.velocity)})
    return sorted(hits, key=lambda h: (h["t"], h["voice"]))


def encode(hits, jo_groups, config=None, seed=DEFAULT_SEED, voices=None):
    """Turn hits into Poisson spike bursts on the JO groups.

    Returns a dict:
      "bursts": one entry per hit: {"t", "voice", "vel", "group", "onset", "offset", "rate_hz"}
      "spike_t":     float64 array, spike times in seconds, sorted
      "spike_body":  int64 array, bodyId of the neuron that spikes
      "spike_group": int8 array, JO group of that neuron
    Same hits + same seed -> same spikes.
    """
    config = config or load_encoder_config()
    voices = voices or load_voices()
    group_of_voice = {v["idx"]: v["jo_group"] for v in voices}
    duration = config["burst_duration_ms"] / 1000.0
    rng = np.random.default_rng(seed)

    bursts, t_parts, body_parts, group_parts = [], [], [], []
    for h in hits:
        g = group_of_voice[h["voice"]]
        bodies = jo_groups[g]
        rate = config["max_rate_hz"] * h["vel"] / config["velocity_full_scale"]
        bursts.append({**h, "group": g, "onset": h["t"], "offset": h["t"] + duration, "rate_hz": rate})

        # Poisson process per neuron: Poisson count, then uniform times in the window.
        n_spikes = rng.poisson(rate * duration, size=len(bodies))
        total = int(n_spikes.sum())
        t_parts.append(h["t"] + rng.random(total) * duration)
        body_parts.append(np.repeat(bodies, n_spikes))
        group_parts.append(np.full(total, g, dtype=np.int8))

    if t_parts:
        t = np.concatenate(t_parts)
        body = np.concatenate(body_parts)
        group = np.concatenate(group_parts)
        order = np.argsort(t, kind="stable")
        t, body, group = t[order], body[order], group[order]
    else:
        t, body, group = np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8)
    return {"bursts": bursts, "spike_t": t, "spike_body": body, "spike_group": group}


def plot_input_raster(encoded, jo_groups, out_path, title, voices=None):
    """Raster: 6 rows = JO groups (one dot per input spike), MIDI hits overlaid as ticks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voices = voices or load_voices()
    by_group = {v["jo_group"]: v for v in voices}
    ink, muted, surface = "#0b0b0b", "#898781", "#fcfcfb"

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for g, bodies in enumerate(jo_groups):
        sel = encoded["spike_group"] == g
        # Each neuron gets its own height inside the row, so a burst shows as a block of dots.
        rank = np.searchsorted(bodies, encoded["spike_body"][sel])
        y = g + 0.12 + 0.62 * rank / max(len(bodies) - 1, 1)
        ax.scatter(encoded["spike_t"][sel], y, s=1.2, color=VOICE_COLORS[g], linewidths=0)
        # MIDI hit times for this group: a dark tick just above the row's dots.
        hit_t = [b["t"] for b in encoded["bursts"] if b["group"] == g]
        ax.vlines(hit_t, g + 0.80, g + 0.97, color=ink, linewidth=1.6)

    for beat in np.arange(0, c.CALL_SEC + 1e-9, 60.0 / c.BPM):
        ax.axvline(beat, color=muted, linewidth=0.4, alpha=0.5, zorder=0)
    ax.axvline(c.BAR_SEC, color=muted, linewidth=0.9, zorder=0)

    ax.set_yticks([g + 0.45 for g in range(len(jo_groups))])
    ax.set_yticklabels([f"{by_group[g]['name']}\nJO group {g} ({len(jo_groups[g])} neurons)" for g in range(len(jo_groups))], color=ink, fontsize=8)
    ax.set_ylim(len(jo_groups), 0)
    ax.set_xlim(-0.05, c.CALL_SEC + 0.05)
    ax.set_xlabel("time (s)", color=ink)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.set_title(title, color=ink, loc="left", fontsize=11)
    ax.text(1.0, 1.02, "black tick = MIDI hit    coloured dots = input spikes (one row of dots per neuron)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)


def main():
    call = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "calls" / "call_01.mid"
    jo_groups = load_jo_groups()
    hits = read_hits(call)
    encoded = encode(hits, jo_groups, seed=DEFAULT_SEED)
    out = ROOT / "results" / f"{call.stem}_input.png"
    plot_input_raster(encoded, jo_groups, out, f"{call.stem}: what the fly's antennal neurons receive (seed {DEFAULT_SEED})")
    print(f"{call.name}: {len(hits)} hits -> {len(encoded['spike_t'])} input spikes; wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
