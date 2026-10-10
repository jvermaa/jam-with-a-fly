"""Check our WWRY call against a third-party MIDI arrangement (Phase S, Gate 0).

Reads the drum track of the reference file (track 0, MIDI channel 10), takes
bars 1-7 (the plain stomp-stomp-clap, before the arrangement adds cymbals) and
compares their sixteenth-step grid with calls/wwry.mid, with the reference's
hand clap (note 39) read as our snare voice (note 38).

The reference file is third-party and is NOT in the repo: it lives in the
gitignored reference/ folder. Only the hit grid found in it is written out.

Real vs. chosen: the target pattern is CHOSEN. So are the 39 -> 38 mapping (our
six-voice table has no clap) and the tempo: we use 82 BPM from a drum score,
the reference file itself says 85. Tempo does not affect the step grid.

Usage (from the repo root):  python -m src.wwry_reference
"""

import hashlib
import json
import pathlib

import mido

from src import constants as c
from src import encode
from src.provenance import provenance

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference" / "Queen_-_We_Will_Rock_You_n.mid"
CALL = ROOT / "calls" / "wwry.mid"
DRUM_TRACK = 0
DRUM_CHANNEL = 9      # MIDI channel 10, zero-based
BARS = range(7)    # bars 1-7, zero-based
NOTE_MAP = {39: 38}   # hand clap -> our snare voice


def reference_bars(path=REFERENCE):
    """{bar (0-based): {note: sorted sixteenth steps}} for the drum track, plus file facts.

    Steps are floats; a hit off the sixteenth grid keeps its fractional step.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    bar_ticks, step_ticks = tpb * c.BEATS_PER_BAR, tpb * c.BEATS_PER_BAR / c.STEPS_PER_BAR
    bars, tempos, tick = {}, [], 0
    for msg in mid.tracks[DRUM_TRACK]:
        tick += msg.time
        if msg.type == "set_tempo":
            tempos.append(mido.tempo2bpm(msg.tempo))
        elif msg.type == "note_on" and msg.velocity > 0 and msg.channel == DRUM_CHANNEL:
            bar = tick // bar_ticks
            bars.setdefault(bar, {}).setdefault(msg.note, []).append((tick - bar * bar_ticks) / step_ticks)
    bars = {b: {n: sorted(s) for n, s in sorted(notes.items())} for b, notes in sorted(bars.items())}
    return bars, {"ticks_per_beat": tpb, "n_tracks": len(mid.tracks), "file_bpm": sorted(set(tempos))}


def mapped_bar_grid(bar_notes):
    """One reference bar -> {our MIDI note: sorted steps}, with NOTE_MAP applied."""
    out = {}
    for note, steps in bar_notes.items():
        out.setdefault(NOTE_MAP.get(note, note), []).extend(steps)
    return {n: sorted(s) for n, s in sorted(out.items())}


def call_bar_grids(path=CALL):
    """Our call -> list of CALL_BARS dicts {MIDI note: sorted steps}."""
    voices = encode.load_voices()
    note = {v["idx"]: v["midi_note"] for v in voices}
    mid = mido.MidiFile(path)
    step_ticks = mid.ticks_per_beat * c.BEATS_PER_BAR / c.STEPS_PER_BAR
    grids = [{} for _ in range(c.CALL_BARS)]
    note_to_voice = {v["midi_note"]: v["idx"] for v in voices}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in note_to_voice:
                step = tick / step_ticks
                bar = int(step // c.STEPS_PER_BAR)
                grids[bar].setdefault(note[note_to_voice[msg.note]], []).append(step - bar * c.STEPS_PER_BAR)
    return [{n: sorted(s) for n, s in sorted(g.items())} for g in grids]


def main():
    bars, facts = reference_bars()
    ours = call_bar_grids()
    found = {b: bars.get(b, {}) for b in BARS}
    mapped = {b: mapped_bar_grid(notes) for b, notes in found.items()}
    as_json = lambda g: {str(n): s for n, s in g.items()}

    all_bars_same = all(found[b] == found[BARS[0]] for b in BARS)
    our_bars_same = all(g == ours[0] for g in ours)
    match = all(mapped[b] == ours[0] for b in BARS) and our_bars_same
    result = {
        "phase": "S",
        "gate": 0,
        **provenance(None),
        "seed_note": "comparison only; no random numbers used",
        "reference_file": str(REFERENCE.relative_to(ROOT)),
        "reference_note": "third-party arrangement, local only (gitignored), not redistributed",
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        **facts,
        "drum_track": DRUM_TRACK,
        "drum_channel_1_based": DRUM_CHANNEL + 1,
        "bars_1_based": [b + 1 for b in BARS],
        "reference_grid_per_bar": {str(b + 1): as_json(found[b]) for b in BARS},
        "reference_bars_identical": all_bars_same,
        "note_map": {str(k): v for k, v in NOTE_MAP.items()},
        "our_call": str(CALL.relative_to(ROOT)),
        "our_grid_per_bar": {str(i + 1): as_json(g) for i, g in enumerate(ours)},
        "our_bpm": 82,
        "pass": bool(match),
    }
    out = ROOT / "results" / "wwry_reference.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"reference: {facts['n_tracks']} tracks, file tempo {facts['file_bpm']} BPM")
    for b in BARS:
        print(f"  bar {b + 1}: {found[b]}")
    print(f"our call, per bar: {ours[0]}")
    print("match (with 39 -> 38):", match)
    if not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
