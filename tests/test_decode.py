"""Step 1.5 decoder: motor spike counts -> hits -> .mid. Uses made-up motor
spike counts, so no connectome data, build/, results/ or answers/ files are
needed. The only file written is a .mid under tmp_path."""

import mido
import numpy as np
import pytest

from src import constants as c
from src import decode, encode

SEED = 0
VOICES = encode.load_voices()
SIM_STEPS = round((c.CALL_SEC + c.SIM_TAIL_SEC) * 1000 / c.DT_MS)  # 4.5 s at 2.0 ms = 2250
N_BINS = 36  # 4.5 s / 125 ms
BURST = 50   # extra spikes in the bin of a hit

# (voice, step); the crash on the last step (31) makes the lag unambiguous.
CALL_HITS = ([(0, s) for s in range(0, 32, 8)] + [(1, s) for s in range(4, 32, 8)]
             + [(2, s) for s in range(2, 32, 4)] + [(5, 31)])


def call_grid():
    g = np.zeros((6, c.CALL_STEPS), dtype=bool)
    for v, s in CALL_HITS:
        g[v, s] = True
    return g


def silent_motor():
    """Steady low rate: one spike per sim step in every group (62 or 63 per 125 ms bin)."""
    return np.ones((6, SIM_STEPS), dtype=np.int64)


def call_motor(delay_steps=0):
    """The silent counts plus a large burst 50 ms into the bin of each call hit."""
    motor = silent_motor()
    for v, s in CALL_HITS:
        t = (s + delay_steps) * c.STEP_SEC + 0.05
        motor[v, int(t * 1000 / c.DT_MS)] += BURST
    return motor


# --- bin_steps -------------------------------------------------------------

def test_bin_steps_gives_36_bins_and_conserves_spikes():
    counts = np.random.default_rng(SEED).poisson(0.3, size=(6, SIM_STEPS))
    binned = decode.bin_steps(counts, c.DT_MS)
    assert binned.shape == (6, N_BINS)
    assert np.array_equal(binned.sum(axis=1), counts.sum(axis=1))


def test_bin_steps_puts_a_spike_in_its_125ms_bin():
    counts = np.zeros((6, SIM_STEPS), dtype=np.int64)
    counts[0, 0] = 1      # t = 0.000 s -> bin 0
    counts[1, 62] = 2     # t = 0.124 s -> bin 0
    counts[2, 63] = 3     # t = 0.126 s -> bin 1
    counts[3, 125] = 4    # t = 0.250 s -> bin 2
    counts[4, 2249] = 5   # t = 4.498 s -> bin 35
    binned = decode.bin_steps(counts, c.DT_MS)
    assert binned[0, 0] == 1
    assert binned[1, 0] == 2
    assert binned[2, 1] == 3
    assert binned[3, 2] == 4
    assert binned[4, 35] == 5
    assert binned.sum() == 15


def test_bin_steps_drops_a_trailing_part_bin():
    counts = np.random.default_rng(SEED).poisson(0.3, size=(6, SIM_STEPS + 10))
    binned = decode.bin_steps(counts, c.DT_MS)
    assert binned.shape == (6, N_BINS)
    assert np.array_equal(binned.sum(axis=1), counts[:, :SIM_STEPS].sum(axis=1))


# --- thresholds / call grid / velocities -----------------------------------

def test_thresholds_are_mean_plus_two_std():
    silent = np.random.default_rng(SEED).poisson(5.0, size=(6, N_BINS))
    thr, mean, std = decode.thresholds(silent)
    assert mean == pytest.approx(silent.mean(axis=1))
    assert std == pytest.approx(silent.std(axis=1))
    assert thr == pytest.approx(silent.mean(axis=1) + 2 * silent.std(axis=1))
    assert thr.shape == (6,)

    # Hand-worked: counts 2,4,4,4,5,5,7,9 have mean 5 and std 2 -> threshold 9.
    thr, mean, std = decode.thresholds(np.array([[2, 4, 4, 4, 5, 5, 7, 9]]))
    assert (thr[0], mean[0], std[0]) == pytest.approx((9.0, 5.0, 2.0))


def test_call_grid_from_hits():
    # The hit at 4.0 s is step 32, outside the 2-bar call, and is dropped.
    grid = decode.call_grid_from_hits([0.0, 0.5, 3.875, 4.0], [0, 1, 5, 2])
    assert grid.shape == (6, c.CALL_STEPS)
    assert grid.dtype == bool
    assert sorted(zip(*(a.tolist() for a in np.nonzero(grid)))) == [(0, 0), (1, 4), (5, 31)]


def test_velocities_stay_in_range_and_grow_with_count():
    thr, std = np.array([10.0, 20.0]), np.array([2.0, 4.0])
    # Columns: below threshold, at threshold, +1 std, +2 std, +3 std (full scale), +10 std.
    binned = np.array([[0, 10, 12, 14, 16, 30],
                       [0, 20, 24, 28, 32, 60]])
    vel = decode.velocities(binned, thr, std)
    assert vel.shape == binned.shape
    assert vel.min() >= 40 and vel.max() <= 127
    assert (np.diff(vel, axis=1) >= 0).all()
    assert vel.tolist() == [[40, 40, 69, 98, 127, 127]] * 2


def test_velocities_with_zero_std_do_not_divide_by_zero():
    vel = decode.velocities(np.array([[0, 1, 5]]), np.array([0.0]), np.array([0.0]))
    assert vel.tolist() == [[40, 69, 127]]


# --- decode(), end to end --------------------------------------------------

def test_decode_recovers_a_synthetic_echo():
    grid = call_grid()
    d = decode.decode(call_motor(), silent_motor(), grid, c.DT_MS)
    assert d["binned"].shape == (6, N_BINS)
    assert d["raw_grid"].shape == (6, N_BINS)
    assert d["threshold"] == pytest.approx(d["silent_mean"] + 2 * d["silent_std"])
    assert d["lag_steps"] == 0
    assert d["answer_grid"].shape == grid.shape
    assert np.array_equal(d["answer_grid"], grid)
    assert d["answer_vel"].shape == grid.shape
    assert d["score"]["overall"]["f1"] == 1.0
    for v in range(6):
        if grid[v].any():
            assert d["score"]["per_voice"][v]["f1"] == 1.0


def test_decode_finds_lag_one_when_bursts_are_one_16th_late():
    grid = call_grid()
    d = decode.decode(call_motor(delay_steps=1), silent_motor(), grid, c.DT_MS)
    assert d["lag_steps"] == 1
    assert np.array_equal(d["answer_grid"], grid)
    assert d["score"]["overall"]["f1"] == 1.0
    # The velocities are shifted by the same lag as the hits.
    assert (d["answer_vel"][grid] == 127).all()
    assert (d["answer_vel"][~grid] == 40).all()


def test_decode_silent_run_gives_no_hits():
    d = decode.decode(silent_motor(), silent_motor(), call_grid(), c.DT_MS)
    assert not d["raw_grid"].any()
    assert d["lag_steps"] == 0
    assert d["score"]["overall"]["f1"] == 0.0
    assert decode.answer_hits(d["answer_grid"], d["answer_vel"]) == []


# --- answer_hits / write_midi ----------------------------------------------

def test_answer_hits_are_sorted_with_time_from_step():
    grid = call_grid()
    vel = np.full(grid.shape, 40)
    vel[5, 31] = 127
    hits = decode.answer_hits(grid, vel)
    assert len(hits) == len(CALL_HITS)
    assert [(h["step"], h["voice"]) for h in hits] == sorted((s, v) for v, s in CALL_HITS)
    for h in hits:
        assert set(h) == {"step", "t", "voice", "vel"}
        assert h["t"] == h["step"] * c.STEP_SEC
    assert hits[-1] == {"step": 31, "t": 3.875, "voice": 5, "vel": 127}
    assert all(h["vel"] == 40 for h in hits[:-1])


def test_write_midi_round_trip(tmp_path):
    # One hit per voice, two voices on the same step, and one hit on the last step.
    hits = [
        {"step": 0, "t": 0.0, "voice": 0, "vel": 100},
        {"step": 0, "t": 0.0, "voice": 2, "vel": 80},
        {"step": 4, "t": 0.5, "voice": 1, "vel": 127},
        {"step": 6, "t": 0.75, "voice": 3, "vel": 40},
        {"step": 16, "t": 2.0, "voice": 4, "vel": 64},
        {"step": 31, "t": 3.875, "voice": 5, "vel": 90},
    ]
    path = tmp_path / "answer.mid"
    decode.write_midi(hits, path, VOICES)

    mid = mido.MidiFile(path)
    assert mid.ticks_per_beat == 480
    assert len(mid.tracks) == 1
    assert mid.length == pytest.approx(4.0)

    tick, tempos, ons, offs = 0, [], [], []
    for msg in mid.tracks[0]:
        tick += msg.time
        if msg.type == "set_tempo":
            tempos.append(msg.tempo)
        elif msg.type == "note_on" and msg.velocity > 0:
            ons.append((tick, msg.note, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            offs.append((tick, msg.note))
    assert tick == 2 * 4 * 480  # exactly 2 bars
    assert [mido.tempo2bpm(t) for t in tempos] == pytest.approx([120.0])

    note = {v["idx"]: v["midi_note"] for v in VOICES}
    assert sorted(ons) == sorted((h["step"] * 120, note[h["voice"]], h["vel"]) for h in hits)
    # Every note lasts one 16th (120 ticks).
    assert sorted(offs) == sorted((h["step"] * 120 + 120, note[h["voice"]]) for h in hits)


def test_write_midi_with_no_hits_is_still_two_bars(tmp_path):
    path = tmp_path / "empty.mid"
    decode.write_midi([], path, VOICES)
    mid = mido.MidiFile(path)
    assert mid.length == pytest.approx(4.0)
    assert not [m for m in mid.tracks[0] if m.type in ("note_on", "note_off")]
