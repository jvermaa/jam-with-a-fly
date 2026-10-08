"""Phase S gate 1 input-coupling metric: post-hit window spike counts, call vs
silent. Uses made-up motor spike counts, so no connectome data, build/ or
results/ files are needed and no simulation is run."""

import numpy as np
import pytest

from src import coupling

DT_MS = 2.0
WINDOW_STEPS = 50          # 100 ms at 2.0 ms
N_STEPS = 2250             # 4.5 s at 2.0 ms
VOICES = ["kick", "snare", "hat", "tom", "ride", "crash"]
STEP_82 = 0.18292682926829268  # one 16th note at 82 BPM, in seconds


def zeros():
    return np.zeros((len(VOICES), N_STEPS), dtype=np.int64)


def flat(rate):
    """`rate` spikes per sim step in every group."""
    return np.full((len(VOICES), N_STEPS), rate, dtype=np.int64)


# --- hit_window_counts -----------------------------------------------------

def test_window_is_50_steps_from_the_hit_step_at_dt_2ms():
    start, stop = coupling.hit_window_counts(zeros(), [0.0, 0.5, 1.001], DT_MS)
    assert start.tolist() == [0, 250, 500]
    assert stop.tolist() == [50, 300, 550]


def test_window_length_follows_window_ms_and_dt():
    start, stop = coupling.hit_window_counts(zeros(), [1.0], DT_MS, window_ms=40.0)
    assert (start.tolist(), stop.tolist()) == ([500], [520])
    start, stop = coupling.hit_window_counts(zeros(), [1.0], 0.5)
    assert (start.tolist(), stop.tolist()) == ([2000], [2200])


def test_window_is_clipped_at_the_end_of_the_run():
    t_late = (N_STEPS - 10) * DT_MS / 1000.0
    start, stop = coupling.hit_window_counts(zeros(), [t_late], DT_MS)
    assert (start.tolist(), stop.tolist()) == ([N_STEPS - 10], [N_STEPS])


@pytest.mark.parametrize("k", [1, 2, 3, 4, 7, 8, 15, 16, 31, 41])
def test_82bpm_grid_times_land_in_the_expected_sim_step(k):
    expected = (k * 7500) // 82   # floor(k * 15/82 s * 1000 / 2 ms), in exact integers
    start, stop = coupling.hit_window_counts(zeros(), [k * STEP_82], DT_MS)
    assert start.tolist() == [expected]
    assert stop.tolist() == [min(expected + WINDOW_STEPS, N_STEPS)]


def test_82bpm_step_41_is_exactly_7_5_seconds():
    # 41 * 15/82 s = 7.5 s -> step 3750, whatever the float rounding of the product.
    start, _ = coupling.hit_window_counts(np.zeros((6, 5000)), [41 * STEP_82], DT_MS)
    assert start.tolist() == [3750]


# --- coupling: what is counted ---------------------------------------------

def test_counts_first_and_last_step_of_the_window_only():
    call = zeros()
    call[0, 249] = 100    # one step before the window
    call[0, 250] = 1      # first step of the window
    call[0, 299] = 2      # last step of the window
    call[0, 300] = 100    # one step after the window
    out = coupling.coupling(call, flat(1), [0.5], [0], VOICES, DT_MS)
    assert out["kick"]["call_spikes"] == 3
    assert out["kick"]["silent_spikes"] == WINDOW_STEPS


def test_only_the_matching_voice_row_is_counted():
    call = flat(1)
    call[1, 250:300] += 7   # a burst in the snare row, inside the kick hit's window
    out = coupling.coupling(call, flat(1), [0.5], [0], VOICES, DT_MS)
    assert out["kick"]["call_spikes"] == WINDOW_STEPS
    assert out["kick"]["change_pct"] == 0.0


def test_82bpm_hit_counts_the_spike_in_its_own_step():
    k = 3                              # 0.5488 s -> step 274
    call = zeros()
    call[2, 273] = 100                 # the step before: outside
    call[2, 274] = 5                   # the hit's step: inside
    out = coupling.coupling(call, flat(1), [k * STEP_82], [2], VOICES, DT_MS)
    assert out["hat"]["call_spikes"] == 5


def test_custom_window_ms_is_used():
    call = zeros()
    call[0, 250:300] = 1
    out = coupling.coupling(call, flat(1), [0.5], [0], VOICES, DT_MS, window_ms=40.0)
    assert out["kick"]["call_spikes"] == 20
    assert out["kick"]["silent_spikes"] == 20


# --- coupling: the numbers -------------------------------------------------

def test_change_pct_silent_10_call_12_is_plus_20():
    silent = zeros()
    call = zeros()
    silent[0, 250:260] = 1   # 10 spikes in the window
    call[0, 250:262] = 1     # 12 spikes in the window
    out = coupling.coupling(call, silent, [0.5], [0], VOICES, DT_MS)
    assert out == {"kick": {"hits": 1, "call_spikes": 12, "silent_spikes": 10,
                            "change_pct": 20.0, "per_hit_change_pct": [20.0]}}


def test_change_pct_can_be_negative():
    silent = zeros()
    call = zeros()
    silent[1, 250:260] = 1
    call[1, 250:255] = 1
    out = coupling.coupling(call, silent, [0.5], [1], VOICES, DT_MS)
    assert out["snare"]["change_pct"] == -50.0


def test_several_hits_are_summed_and_per_hit_keeps_hit_order():
    silent = zeros()
    call = zeros()
    for step in (0, 500, 1000):      # kick hits at 0.0, 1.0, 2.0 s
        silent[0, step:step + 10] = 1
    call[0, 0:12] = 1                # 12 vs 10 -> +20
    call[0, 500:510] = 1             # 10 vs 10 ->   0
    call[0, 1000:1015] = 1           # 15 vs 10 -> +50
    silent[1, 250:270] = 1           # snare hit at 0.5 s: 20 silent
    call[1, 250:260] = 1             # 10 call -> -50
    # Hits interleaved across voices and not in time order for the kick.
    hit_t = [2.0, 0.5, 0.0, 1.0]
    hit_voice = [0, 1, 0, 0]
    out = coupling.coupling(call, silent, hit_t, hit_voice, VOICES, DT_MS)
    assert out["kick"] == {"hits": 3, "call_spikes": 37, "silent_spikes": 30,
                           "change_pct": 23.33, "per_hit_change_pct": [50.0, 20.0, 0.0]}
    assert out["snare"] == {"hits": 1, "call_spikes": 10, "silent_spikes": 20,
                            "change_pct": -50.0, "per_hit_change_pct": [-50.0]}


def test_identical_call_and_silent_give_zero_for_every_voice_with_hits():
    motor = np.random.default_rng(0).poisson(3.0, size=(len(VOICES), N_STEPS)) + 1
    hit_t = [i * STEP_82 for i in range(12)]
    hit_voice = [i % len(VOICES) for i in range(12)]
    out = coupling.coupling(motor, motor.copy(), hit_t, hit_voice, VOICES, DT_MS)
    assert set(out) == set(VOICES)
    for name in VOICES:
        assert out[name]["hits"] == 2
        assert out[name]["call_spikes"] == out[name]["silent_spikes"] > 0
        assert out[name]["change_pct"] == 0.0
        assert out[name]["per_hit_change_pct"] == [0.0, 0.0]


def test_voices_without_hits_are_absent():
    out = coupling.coupling(flat(1), flat(1), [0.5, 1.0], [0, 2], VOICES, DT_MS)
    assert list(out) == ["kick", "hat"]


def test_no_hits_at_all_gives_an_empty_result():
    assert coupling.coupling(flat(1), flat(1), [], [], VOICES, DT_MS) == {}


# --- coupling: edge cases --------------------------------------------------

def test_zero_silent_spikes_gives_none_not_an_exception():
    call = zeros()
    call[0, 250:255] = 1
    out = coupling.coupling(call, zeros(), [0.5], [0], VOICES, DT_MS)
    assert out["kick"] == {"hits": 1, "call_spikes": 5, "silent_spikes": 0,
                           "change_pct": None, "per_hit_change_pct": [None]}


def test_one_empty_silent_window_is_none_per_hit_but_total_is_defined():
    silent = zeros()
    call = zeros()
    silent[0, 500:510] = 1           # only the second hit has silent spikes
    call[0, 0:4] = 1
    call[0, 500:511] = 1
    out = coupling.coupling(call, silent, [0.0, 1.0], [0, 0], VOICES, DT_MS)
    assert out["kick"]["per_hit_change_pct"] == [None, 10.0]
    assert out["kick"]["call_spikes"] == 15
    assert out["kick"]["silent_spikes"] == 10
    assert out["kick"]["change_pct"] == 50.0


def test_hit_window_past_the_end_is_clipped_not_an_error():
    t_late = (N_STEPS - 10) * DT_MS / 1000.0   # only 10 steps left in the run
    out = coupling.coupling(flat(3), flat(2), [t_late], [5], VOICES, DT_MS)
    assert out["crash"] == {"hits": 1, "call_spikes": 30, "silent_spikes": 20,
                            "change_pct": 50.0, "per_hit_change_pct": [50.0]}


def test_plain_lists_are_accepted_and_counts_are_python_ints():
    call = flat(2).tolist()
    silent = flat(1).tolist()
    out = coupling.coupling(call, silent, [0.5], [0], VOICES, DT_MS)
    assert out["kick"]["change_pct"] == 100.0
    assert type(out["kick"]["call_spikes"]) is int
    assert type(out["kick"]["silent_spikes"]) is int
    assert type(out["kick"]["hits"]) is int
