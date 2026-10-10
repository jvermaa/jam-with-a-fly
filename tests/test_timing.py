"""Per-call tempo (Phase S): src/timing.py and the `timing` argument threaded
through encode, run_fly, decode, check_calls and make_calls. Default timing must
give exactly the 120 BPM numbers of src/constants.py. No simulation is run; the
only files written are .mid files under tmp_path. The regression test at the end
reads gitignored recordings in results/ and skips without them."""

import dataclasses
import json
import pathlib

import mido
import numpy as np
import pytest

from src import check_calls, decode, encode, make_calls, run_fly, timing
from src import constants as c
from src.timing import Timing

ROOT = pathlib.Path(__file__).resolve().parent.parent
CALLS = ROOT / "calls"
VOICES = encode.load_voices()
T82 = Timing(82)
KICK_STEPS = [0, 2, 8, 10, 16, 18, 24, 26]
SNARE_STEPS = [4, 12, 20, 28]


def tempo_and_end_tick(path):
    mid = mido.MidiFile(path)
    tempos, end = [], 0
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append(mido.tempo2bpm(msg.tempo))
        end = max(end, tick)
    return tempos, end, mid.ticks_per_beat


# --- Timing ----------------------------------------------------------------

def test_default_timing_reproduces_constants_exactly():
    t = Timing()
    assert t.bpm == 120 == c.BPM
    assert t.beat_sec == 0.5
    assert t.step_sec == 0.125 == c.STEP_SEC
    assert t.bar_sec == 2.0 == c.BAR_SEC
    assert t.call_sec == 4.0 == c.CALL_SEC
    assert t.tail_sec == 0.5 == c.SIM_TAIL_SEC
    assert t.sim_sec == 4.5
    assert timing.DEFAULT == t


def test_timing_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Timing().bpm = 100


def test_timing_at_82_bpm():
    assert T82.call_sec == pytest.approx(8 * 60 / 82)
    assert T82.call_sec == pytest.approx(5.8537, abs=1e-4)
    assert T82.step_sec == pytest.approx(0.18293, abs=1e-5)
    assert c.CALL_STEPS * T82.step_sec == pytest.approx(T82.call_sec, abs=1e-12)
    assert T82.bar_sec == pytest.approx(T82.call_sec / 2)
    assert T82.tail_sec == pytest.approx(4 * T82.step_sec, abs=1e-12)
    assert T82.sim_sec == pytest.approx(36 * T82.step_sec, abs=1e-12)


def test_for_call_looks_the_tempo_up_by_file_stem():
    assert Timing.for_call("calls/wwry.mid").bpm == 82
    assert Timing.for_call(CALLS / "wwry.mid").bpm == 82
    assert Timing.for_call("calls/call_01.mid").bpm == 120
    assert Timing.for_call(None).bpm == 120
    assert timing.configured_bpm("calls/wwry.mid") == 82
    assert timing.configured_bpm("calls/call_01.mid") == 120
    assert timing.configured_bpm(None) == 120


# --- run_fly step counts, decode.bin_steps ---------------------------------

def test_default_step_counts_are_unchanged():
    assert run_fly.total_steps() == 2250
    assert run_fly.call_steps() == 2000
    assert run_fly.total_steps(Timing()) == 2250
    assert run_fly.call_steps(Timing()) == 2000


def test_bin_steps_gives_36_columns_at_default_and_at_82_bpm():
    default = np.zeros((6, run_fly.total_steps()), dtype=np.int64)
    assert decode.bin_steps(default, c.DT_MS).shape == (6, 36)
    slow = np.zeros((6, run_fly.total_steps(T82)), dtype=np.int64)
    assert decode.bin_steps(slow, c.DT_MS, T82).shape == (6, 36)


def test_82_bpm_run_covers_the_call_and_the_whole_tail():
    assert run_fly.call_steps(T82) == round(T82.call_sec * 1000 / c.DT_MS)
    assert run_fly.total_steps(T82) * c.DT_MS / 1000 >= T82.sim_sec - 1e-9
    assert run_fly.total_steps(T82) > run_fly.total_steps()


# --- encode.read_hits, decode.call_grid_from_hits --------------------------

def test_read_hits_wwry_is_on_the_82_bpm_grid():
    hits = encode.read_hits(CALLS / "wwry.mid")
    assert len(hits) == 12
    kick = [h["t"] for h in hits if h["voice"] == 0]
    snare = [h["t"] for h in hits if h["voice"] == 1]
    assert {h["voice"] for h in hits} == {0, 1}
    assert kick == pytest.approx([s * T82.step_sec for s in KICK_STEPS], abs=1e-9)
    assert snare == pytest.approx([s * T82.step_sec for s in SNARE_STEPS], abs=1e-9)


def test_call_grid_from_wwry_hits_at_82_bpm():
    hits = encode.read_hits(CALLS / "wwry.mid")
    grid = decode.call_grid_from_hits([h["t"] for h in hits], [h["voice"] for h in hits], timing=T82)
    expected = np.zeros((6, 32), dtype=bool)
    expected[0, KICK_STEPS] = True
    expected[1, SNARE_STEPS] = True
    assert grid.shape == (6, 32)
    assert np.array_equal(grid, expected)


def test_read_hits_call_06_is_still_on_the_120_bpm_grid():
    hits = encode.read_hits(CALLS / "call_06.mid")
    assert hits
    steps = np.array([h["t"] for h in hits]) / 0.125
    assert np.allclose(steps, np.round(steps), atol=1e-9)
    assert max(h["t"] for h in hits) < c.CALL_SEC
    explicit = encode.read_hits(CALLS / "call_06.mid", timing=Timing())
    assert explicit == hits


# --- decode.answer_hits, decode.write_midi ---------------------------------

def test_answer_hits_times_follow_the_timing():
    grid = np.zeros((6, 32), dtype=bool)
    grid[0, 8] = grid[1, 12] = True
    vel = np.full((6, 32), 100)
    assert [h["t"] for h in decode.answer_hits(grid, vel)] == [1.0, 1.5]
    slow = decode.answer_hits(grid, vel, T82)
    assert [h["t"] for h in slow] == pytest.approx([8 * T82.step_sec, 12 * T82.step_sec], abs=1e-12)
    assert [h["step"] for h in slow] == [8, 12]


def test_write_midi_at_82_bpm(tmp_path):
    hits = [{"step": 0, "voice": 0, "vel": 100}, {"step": 4, "voice": 1, "vel": 90}]
    path = tmp_path / "answer_82.mid"
    decode.write_midi(hits, path, timing=T82)
    tempos, end, tpb = tempo_and_end_tick(path)
    assert len(tempos) == 1
    assert tempos[0] == pytest.approx(82, abs=0.01)
    assert end == 2 * c.BEATS_PER_BAR * tpb


def test_write_midi_default_is_120_bpm(tmp_path):
    hits = [{"step": 0, "voice": 0, "vel": 100}, {"step": 4, "voice": 1, "vel": 90}]
    path = tmp_path / "answer_120.mid"
    decode.write_midi(hits, path)
    tempos, end, tpb = tempo_and_end_tick(path)
    assert tempos == [pytest.approx(120, abs=1e-9)]
    assert end == 2 * c.BEATS_PER_BAR * tpb


# --- check_calls.check_file, make_calls ------------------------------------

def test_check_file_passes_wwry_at_82_bpm():
    r = check_calls.check_file(CALLS / "wwry.mid", VOICES)
    assert r["expected_bpm"] == 82
    assert r["checks"]["tempo_as_configured"]
    assert r["checks"]["length_is_2_bars_within_1_tick"]
    assert r["checks"]["only_allowed_notes"]
    assert r["pass"]
    assert r["n_hits"] == 12 and r["off_grid_hits"] == 0


def test_check_file_still_passes_call_01():
    r = check_calls.check_file(CALLS / "call_01.mid", VOICES)
    assert r["expected_bpm"] == 120
    assert r["pass"]
    assert all(r["checks"].values())


def wwry_grid():
    spec = (CALLS / "call_spec_wwry.txt").read_text()
    return make_calls.parse_spec(spec, {v["name"] for v in VOICES})["wwry"]


def test_wwry_spec_parses_to_the_stomp_stomp_clap_grid():
    grid = wwry_grid()
    assert set(grid) == {"Kick", "Snare"}
    assert [i for i, x in enumerate(grid["Kick"]) if x] == KICK_STEPS
    assert [i for i, x in enumerate(grid["Snare"]) if x] == SNARE_STEPS


def test_check_file_rejects_wwry_written_at_120_bpm(tmp_path, monkeypatch):
    # check_file reports the path relative to its ROOT, so point ROOT at the tmp dir.
    monkeypatch.setattr(check_calls, "ROOT", tmp_path)
    path = tmp_path / "wwry.mid"
    make_calls.build_midi(wwry_grid(), VOICES, 120).save(path)
    r = check_calls.check_file(path, VOICES)
    assert r["expected_bpm"] == 82
    assert r["checks"]["tempo_as_configured"] is False
    assert r["checks"]["length_is_2_bars_within_1_tick"]
    assert r["checks"]["only_allowed_notes"]
    assert r["pass"] is False


def test_check_file_accepts_wwry_built_at_82_bpm(tmp_path, monkeypatch):
    monkeypatch.setattr(check_calls, "ROOT", tmp_path)
    path = tmp_path / "wwry.mid"
    make_calls.build_midi(wwry_grid(), VOICES, 82).save(path)
    r = check_calls.check_file(path, VOICES)
    assert r["pass"]
    assert r["step_grid"] == {"Kick": "x.x.....x.x....." * 2, "Snare": "....x.......x..." * 2}


# --- regression: an existing 120 BPM recording decodes as before -----------

RUN = ROOT / "results" / "run_call_06_seed0.npz"
SILENT = ROOT / "results" / "silent_seed0.npz"
SCORE = ROOT / "results" / "score_untrained_call_06_seed0.json"


@pytest.mark.skipif(not (RUN.exists() and SILENT.exists() and SCORE.exists()),
                    reason="recordings in results/ are gitignored and not present")
def test_default_timing_reproduces_the_committed_call_06_score():
    run, silent = np.load(RUN), np.load(SILENT)
    want = json.loads(SCORE.read_text())
    dt_ms = float(run["dt_ms"])
    call_grid = decode.call_grid_from_hits(run["hit_t"], run["hit_voice"])
    d = decode.decode(run["motor_step_counts"], silent["motor_step_counts"], call_grid, dt_ms)
    assert d["score"]["overall"]["f1"] == want["f1"]
    assert d["lag_steps"] == want["lag_steps"]
    rows = lambda g: ["".join("x" if x else "." for x in row) for row in g]
    assert rows(d["answer_grid"]) == [p["answer_grid"] for p in want["per_voice"]]
    assert rows(call_grid) == [p["call_grid"] for p in want["per_voice"]]
