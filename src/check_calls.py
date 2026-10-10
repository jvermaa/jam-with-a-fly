"""Validate the human-made call files (PLAN.md Step 1.2).

Pass per file: tempo 120 BPM, length exactly CALL_SEC (+-1 tick), only the notes
in config/voices.json. Pass for the set: all 6 voices used at least once.
The files are only read, never changed.

Real vs. chosen: the calls are CHOSEN (a human programmed them in Ableton).

Usage (from the repo root):  python -m src.check_calls
"""

import hashlib
import json
import pathlib

import mido

from src import constants as c
from src.provenance import provenance

ROOT = pathlib.Path(__file__).resolve().parent.parent
CALLS = ROOT / "calls"


def check_file(path, voices):
    note_to_voice = {v["midi_note"]: v for v in voices}
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat

    tempos, hits, other_notes, end_tick = [], [], [], 0
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append(mido.tempo2bpm(msg.tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                if msg.note in note_to_voice:
                    hits.append((tick, msg.note, msg.velocity))
                else:
                    other_notes.append(msg.note)
        end_tick = max(end_tick, tick)

    # No tempo event means the MIDI default, 120 BPM.
    tempo_source = "file" if tempos else "midi_default"
    bpm_values = sorted(set(tempos)) or [120.0]

    expected_ticks = c.CALL_BARS * c.BEATS_PER_BAR * tpb
    ticks_per_step = tpb * c.BEATS_PER_BAR / c.STEPS_PER_BAR
    off_grid = [t for t, _, _ in hits if t % ticks_per_step != 0]

    grid = {v["name"]: ["."] * c.CALL_STEPS for v in voices}
    for t, note, _ in hits:
        step = int(t // ticks_per_step)
        if step < c.CALL_STEPS:
            grid[note_to_voice[note]["name"]][step] = "x"

    checks = {
        "tempo_is_120": bpm_values == [float(c.BPM)],
        "length_is_4s_within_1_tick": abs(end_tick - expected_ticks) <= 1,
        "only_allowed_notes": not other_notes,
    }
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pass": all(checks.values()),
        "checks": checks,
        "bpm": bpm_values,
        "tempo_source": tempo_source,
        "ticks_per_beat": tpb,
        "end_tick": end_tick,
        "expected_end_tick": expected_ticks,
        "length_sec": mid.length,
        "disallowed_notes": sorted(set(other_notes)),
        "n_hits": len(hits),
        "hits_per_voice": {v["name"]: sum(1 for _, n, _ in hits if n == v["midi_note"]) for v in voices},
        "velocities": sorted({vel for _, _, vel in hits}),
        "off_grid_hits": len(off_grid),
        "step_grid": {name: "".join(row) for name, row in grid.items() if "x" in row},
    }


def main():
    voices = json.loads((ROOT / "config" / "voices.json").read_text())["voices"]
    files = sorted(CALLS.glob("call_*.mid"))
    reports = [check_file(f, voices) for f in files]

    used = {name for r in reports for name, n in r["hits_per_voice"].items() if n > 0}
    set_checks = {
        "five_or_more_files": len(files) >= 5,
        "every_file_passes": bool(reports) and all(r["pass"] for r in reports),
        "all_six_voices_used_across_set": used == {v["name"] for v in voices},
    }
    result = {
        "step": "1.2",
        **provenance(None),
        "seed_note": "validation only; no random numbers used",
        "pass": all(set_checks.values()),
        "set_checks": set_checks,
        "files": reports,
    }
    out = ROOT / "results" / "calls_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    for r in reports:
        print(f"{r['file']}: {'PASS' if r['pass'] else 'FAIL'}  bpm={r['bpm']} length={r['length_sec']}s "
              f"hits={r['n_hits']} off_grid={r['off_grid_hits']} vel={r['velocities']}")
        for name, row in r["step_grid"].items():
            print(f"    {name:11} {row[:16]} | {row[16:]}")
    print("set:", set_checks)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
