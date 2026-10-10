"""Phase S C3/C4 pure helpers: transient.* (runaway time, persistence, window sums),
reset_transient.step_inputs / kind_of / coupling_for and input_strength.encoder /
grid_steps / screen_checks. Uses made-up spike counts, so no simulation is run. Only
the jo_sets test reads build/ files and only the two sanity tests read results/ files;
each is skipped when its files are absent."""

import json
import pathlib

import numpy as np
import pytest

from src import encode, input_strength, reset_transient, transient
from src.timing import Timing

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / "build" / "graph.npz"
JO_GROUPS = ROOT / "build" / "jo_groups.json"
RESULTS = ROOT / "results"

DT_MS = 2.0

# ---- transient.first_time_above_ms -----------------------------------------------------------

N_KC = 100
CYCLE_STEPS = 2


def test_first_time_above_none_when_never_above():
    assert transient.first_time_above_ms(np.zeros(20, dtype=int), N_KC, CYCLE_STEPS, DT_MS) is None
    assert transient.first_time_above_ms([10, 10, 10, 10, 10], N_KC, CYCLE_STEPS, DT_MS) is None


def test_first_time_above_exactly_at_threshold_is_not_above():
    # 25 + 25 = 50 of 100 KCs = exactly 50 %, which is not "more than".
    assert transient.first_time_above_ms([0, 25, 25, 0], N_KC, CYCLE_STEPS, DT_MS) is None
    assert transient.first_time_above_ms([0, 25, 26, 0], N_KC, CYCLE_STEPS, DT_MS) == 2.0


def test_first_time_above_crossing_spread_over_two_adjacent_steps():
    # Neither step alone exceeds 50 %; the window starting at step 3 holds 51.
    counts = [0, 0, 0, 26, 25, 0]
    assert transient.first_time_above_ms(counts, N_KC, CYCLE_STEPS, DT_MS) == 6.0


def test_first_time_above_scales_with_dt():
    counts = [0, 0, 0, 26, 25, 0]
    assert transient.first_time_above_ms(counts, N_KC, CYCLE_STEPS, 0.2) == pytest.approx(0.6)
    assert transient.first_time_above_ms(counts, N_KC, CYCLE_STEPS, 1.0) == 3.0


def test_first_time_above_returns_the_first_of_several_crossings():
    counts = [0, 30, 30, 0, 0, 40, 40, 0]
    assert transient.first_time_above_ms(counts, N_KC, CYCLE_STEPS, DT_MS) == 2.0


def test_first_time_above_custom_pct():
    counts = [0, 0, 6, 5, 0]
    assert transient.first_time_above_ms(counts, N_KC, CYCLE_STEPS, DT_MS) is None
    assert transient.first_time_above_ms(counts, N_KC, CYCLE_STEPS, DT_MS, pct=10.0) == 4.0


def test_first_time_above_none_when_shorter_than_one_cycle():
    assert transient.first_time_above_ms([100], N_KC, CYCLE_STEPS, DT_MS) is None
    assert transient.first_time_above_ms([], N_KC, CYCLE_STEPS, DT_MS) is None


# ---- transient.persistence -------------------------------------------------------------------

LAST_INPUT_END_S = 0.1  # + 500 ms -> the late window starts at step 300 (dt 2 ms)
PERSIST_KEYS = {"last_spike_s", "spikes_after_ms", "spikes_after", "late_window_s", "self_sustained"}


def test_persistence_activity_stops_before_the_late_window():
    counts = np.zeros(500, dtype=int)
    counts[10] = 7
    counts[200] = 4
    counts[299] = 1  # the last step before the late window
    out = transient.persistence(counts, LAST_INPUT_END_S, DT_MS)
    assert set(out) == PERSIST_KEYS
    assert out == {
        "last_spike_s": 0.598,
        "spikes_after_ms": 500.0,
        "spikes_after": 0,
        "late_window_s": 0.4,  # steps 300..499
        "self_sustained": False,
    }


def test_persistence_activity_in_the_late_window():
    counts = np.zeros(500, dtype=int)
    counts[10] = 7
    counts[350] = 3
    counts[499] = 2
    out = transient.persistence(counts, LAST_INPUT_END_S, DT_MS)
    assert out["self_sustained"] is True
    assert out["spikes_after"] == 5
    assert out["last_spike_s"] == 0.998
    assert out["late_window_s"] == 0.4


def test_persistence_after_ms_moves_the_late_window():
    counts = np.zeros(500, dtype=int)
    counts[200] = 4
    out = transient.persistence(counts, LAST_INPUT_END_S, DT_MS, after_ms=100.0)  # late from step 100
    assert out["spikes_after_ms"] == 100.0
    assert out["spikes_after"] == 4
    assert out["late_window_s"] == 0.8
    assert out["self_sustained"] is True


def test_persistence_no_spikes_at_all():
    out = transient.persistence(np.zeros(500, dtype=int), LAST_INPUT_END_S, DT_MS)
    assert out["last_spike_s"] is None
    assert out["spikes_after"] == 0
    assert out["self_sustained"] is False


def test_persistence_run_too_short_for_a_late_window():
    counts = np.zeros(300, dtype=int)  # ends exactly where the late window would start
    counts[100] = 5
    out = transient.persistence(counts, LAST_INPUT_END_S, DT_MS)
    assert out["self_sustained"] is None
    assert out["spikes_after"] == 0
    assert out["late_window_s"] == 0.0
    assert out["last_spike_s"] == 0.2
    assert transient.persistence(np.ones(50, dtype=int), LAST_INPUT_END_S, DT_MS)["self_sustained"] is None


# ---- transient.window_sum --------------------------------------------------------------------

def test_window_sum_per_window():
    assert transient.window_sum(np.arange(10), [0, 4], 3) == [0 + 1 + 2, 4 + 5 + 6]


def test_window_sum_is_clipped_at_the_end_of_the_row():
    assert transient.window_sum(np.arange(10), [0, 8], 4) == [6, 8 + 9]
    assert transient.window_sum(np.arange(10), [9], 4) == [9]


def test_window_sum_no_windows():
    assert transient.window_sum(np.arange(10), [], 4) == []


# ---- transient.pct_change --------------------------------------------------------------------

def test_pct_change_none_when_baseline_is_zero():
    assert transient.pct_change(3, 0) is None
    assert transient.pct_change(0, 0) is None
    assert transient.pct_change(3.5, 0.0) is None


def test_pct_change_values():
    assert transient.pct_change(15, 10) == 50.0
    assert transient.pct_change(10, 10) == 0.0
    assert transient.pct_change(5, 10) == -50.0


def test_pct_change_is_rounded_to_two_decimals():
    assert transient.pct_change(4, 3) == 33.33
    assert transient.pct_change(1, 3) == -66.67


# ---- transient.own_hit_vs_quiet --------------------------------------------------------------

# 2-step windows. Hit windows start at 0 and 4, quiet windows at 2, 6 and 8.
CALL_ROW = [3, 1, 1, 0, 2, 2, 0, 1, 1, 0, 0, 0]
SILENT_ROW = [1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
HIT_STEPS = [0, 4]
QUIET_STEPS = [2, 6, 8]


def test_own_hit_vs_quiet_all_keys():
    out = transient.own_hit_vs_quiet(CALL_ROW, SILENT_ROW, HIT_STEPS, QUIET_STEPS, 2)
    assert out == {
        "hits": 2,
        "call_spikes": 8,                      # 4 + 4
        "silent_spikes": 4,                    # 1 + 3
        "change_pct_vs_silent": 100.0,
        "quiet_windows": 3,
        "call_spikes_per_hit_window": 4.0,
        "call_spikes_per_quiet_window": 1.0,   # (1 + 1 + 1) / 3
        "change_pct_vs_quiet_windows": 300.0,  # per-window means 4 vs 1; sums 8 vs 3 would give 166.67
    }


def test_own_hit_vs_quiet_silent_row_all_zeros():
    out = transient.own_hit_vs_quiet(CALL_ROW, [0] * len(CALL_ROW), HIT_STEPS, QUIET_STEPS, 2)
    assert out["silent_spikes"] == 0
    assert out["change_pct_vs_silent"] is None
    assert out["call_spikes"] == 8
    assert out["change_pct_vs_quiet_windows"] == 300.0


def test_own_hit_vs_quiet_negative_change():
    # The group fires less after its hits (1 per window) than in quiet windows (2 per window).
    call = [1, 0, 2, 0, 0, 1, 1, 1, 2, 0, 0, 0]
    out = transient.own_hit_vs_quiet(call, SILENT_ROW, HIT_STEPS, QUIET_STEPS, 2)
    assert out["call_spikes_per_hit_window"] == 1.0
    assert out["call_spikes_per_quiet_window"] == 2.0
    assert out["change_pct_vs_quiet_windows"] == -50.0
    assert out["change_pct_vs_silent"] == -50.0  # 2 vs 4


# ---- reset_transient.step_inputs -------------------------------------------------------------

STEP_SEC = 0.125
BURST_SEC = 0.03


def test_step_inputs_selects_only_the_chosen_step():
    spike_t = np.array([0.0, 0.01, 0.25, 0.26, 0.2799, 0.375, 0.38])
    sel, rel = reset_transient.step_inputs(spike_t, 2, STEP_SEC, BURST_SEC)
    assert sel.dtype == bool
    assert sel.tolist() == [False, False, True, True, True, False, False]
    assert rel == pytest.approx([0.0, 0.01, 0.0299])
    assert len(rel) == sel.sum()


def test_step_inputs_step_zero():
    sel, rel = reset_transient.step_inputs([0.0, 0.02, 0.125, 0.13], 0, STEP_SEC, BURST_SEC)
    assert sel.tolist() == [True, True, False, False]
    assert rel == pytest.approx([0.0, 0.02])


def test_step_inputs_spike_exactly_at_step_start_is_included():
    sel, rel = reset_transient.step_inputs([0.25], 2, STEP_SEC, BURST_SEC)
    assert sel.tolist() == [True]
    assert rel.tolist() == [0.0]


def test_step_inputs_hair_negative_spike_is_included_at_time_zero():
    sel, rel = reset_transient.step_inputs([0.25 - 1e-12], 2, STEP_SEC, BURST_SEC)
    assert sel.tolist() == [True]
    assert rel.tolist() == [0.0]  # clamped, never negative


def test_step_inputs_spike_at_burst_end_is_excluded():
    sel, rel = reset_transient.step_inputs([0.25 + BURST_SEC], 2, STEP_SEC, BURST_SEC)
    assert sel.tolist() == [False]
    assert len(rel) == 0
    sel, _ = reset_transient.step_inputs([BURST_SEC], 0, STEP_SEC, BURST_SEC)
    assert sel.tolist() == [False]


def test_step_inputs_clearly_earlier_spike_is_excluded():
    sel, _ = reset_transient.step_inputs([0.25 - 1e-6], 2, STEP_SEC, BURST_SEC)
    assert sel.tolist() == [False]


# ---- reset_transient.kind_of -----------------------------------------------------------------

def test_kind_of():
    assert reset_transient.kind_of({"voices_hit": []}) == "silent"
    assert reset_transient.kind_of({"voices_hit": ["Kick"]}) == "Kick"


# ---- reset_transient.coupling_for ------------------------------------------------------------

TABLE = {
    "Kick": {"mean_motor_spikes": {"Kick": 10.0, "Snare": 4.0}},
    "Snare": {"mean_motor_spikes": {"Kick": 8.0, "Snare": 6.0}},
    "silent": {"mean_motor_spikes": {"Kick": 0.0, "Snare": 2.0}},
}


def test_coupling_for_silent_baseline_zero():
    assert reset_transient.coupling_for(TABLE, "Kick", "Snare") == {
        "own_hit_mean_spikes": 10.0,
        "silent_step_mean_spikes": 0.0,
        "other_voice_hit_mean_spikes": 8.0,      # kick motor group on snare-hit steps
        "change_pct_vs_silent_step": None,
        "change_pct_vs_other_voice_hit": 25.0,
    }


def test_coupling_for_silent_baseline_nonzero():
    assert reset_transient.coupling_for(TABLE, "Snare", "Kick") == {
        "own_hit_mean_spikes": 6.0,
        "silent_step_mean_spikes": 2.0,
        "other_voice_hit_mean_spikes": 4.0,      # snare motor group on kick-hit steps
        "change_pct_vs_silent_step": 200.0,
        "change_pct_vs_other_voice_hit": 50.0,
    }


# ---- input_strength.encoder ------------------------------------------------------------------

def test_encoder_overrides_rate_and_duration_only():
    default = encode.load_encoder_config()
    cfg = input_strength.encoder(50.0, 15.0)
    assert cfg["max_rate_hz"] == 50.0
    assert cfg["burst_duration_ms"] == 15.0
    assert set(cfg) == set(default) | {"max_rate_hz", "burst_duration_ms"}
    for key, value in default.items():
        if key not in {"max_rate_hz", "burst_duration_ms"}:
            assert cfg[key] == value


def test_encoder_does_not_change_the_default_config():
    before = encode.load_encoder_config()
    input_strength.encoder(1.0, 2.0)
    assert encode.load_encoder_config() == before


# ---- input_strength.grid_steps ---------------------------------------------------------------

def test_grid_steps_at_120_bpm():
    timing = Timing(120.0)
    assert timing.step_sec == pytest.approx(0.125)
    by_voice, quiet = input_strength.grid_steps([0.0, 0.25, 0.5], [0, 0, 1], timing, DT_MS)
    assert by_voice == {0: [0, 125], 1: [250]}
    assert len(quiet) == 32 - 3
    assert not {0, 125, 250} & set(quiet)
    assert 62 in quiet  # grid step 1 starts at 62.5 sim steps -> floor
    assert quiet == sorted(quiet)
    assert quiet[-1] == 1937  # grid step 31 -> floor(1937.5)


# ---- input_strength.screen_checks ------------------------------------------------------------

def measurement(self_sustained=False, kc_pct=5.0, kick=(10, 2), snare=(7, 3)):
    return {
        "after_last_hit": {"self_sustained": self_sustained},
        "kc_pct_active_per_16th_mean": kc_pct,
        "motor_after_own_hits_0_100ms": {
            "Kick": {"call_spikes": kick[0], "silent_spikes": kick[1]},
            "Snare": {"call_spikes": snare[0], "silent_spikes": snare[1]},
        },
    }


def test_screen_checks_all_pass():
    assert input_strength.screen_checks(measurement()) == {
        "not_self_sustained": True,
        "kc_sparse_per_16th": True,
        "no_runaway": True,
        "motor_rises_after_kick_and_snare_hits": True,
    }


def test_screen_checks_self_sustained_is_a_runaway():
    out = input_strength.screen_checks(measurement(self_sustained=True))
    assert out["not_self_sustained"] is False
    assert out["kc_sparse_per_16th"] is True
    assert out["no_runaway"] is False


def test_screen_checks_dense_kcs_are_a_runaway():
    out = input_strength.screen_checks(measurement(kc_pct=35.0))
    assert out["not_self_sustained"] is True
    assert out["kc_sparse_per_16th"] is False
    assert out["no_runaway"] is False


def test_screen_checks_kc_threshold_is_strict():
    assert not input_strength.screen_checks(measurement(kc_pct=20.0))["no_runaway"]
    assert input_strength.screen_checks(measurement(kc_pct=19.99))["no_runaway"]


def test_screen_checks_unknown_persistence_does_not_pass():
    out = input_strength.screen_checks(measurement(self_sustained=None))
    assert out["not_self_sustained"] is False
    assert out["no_runaway"] is False


@pytest.mark.parametrize("kick, snare, expected", [
    ((10, 2), (7, 3), True),
    ((2, 2), (7, 3), False),   # kick equal to silent: not a rise
    ((10, 2), (1, 3), False),  # snare below silent
    ((0, 0), (0, 0), False),
    ((1, 0), (1, 0), True),
])
def test_screen_checks_motor_rises_needs_both_voices(kick, snare, expected):
    out = input_strength.screen_checks(measurement(kick=kick, snare=snare))
    assert out["motor_rises_after_kick_and_snare_hits"] is expected
    assert out["no_runaway"] is True  # independent of the motor check


# ---- input_strength.jo_sets (reads build/) ---------------------------------------------------

@pytest.mark.skipif(not (GRAPH.exists() and JO_GROUPS.exists()), reason="build/graph.npz or build/jo_groups.json absent")
def test_jo_sets_auditory_is_a_subset_of_all():
    sets = input_strength.jo_sets()
    assert set(sets) == {"all", "auditory"}
    assert len(sets["all"]) == 6
    assert len(sets["auditory"]) == 6
    assert sum(len(g) for g in sets["all"]) == 554
    for full, auditory in zip(sets["all"], sets["auditory"]):
        assert set(auditory) <= set(full)
    assert 0 < sum(len(g) for g in sets["auditory"]) < 554


# ---- results files ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["wwry_reset_transient.json", "wwry_input_strength.json"])
def test_results_file_has_provenance_and_no_absolute_paths(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"results/{name} absent")
    text = path.read_text()
    data = json.loads(text)
    for key in ("seed", "git_sha", "timestamp"):
        assert key in data
    assert "/Users/" not in text
    assert "/home/" not in text
