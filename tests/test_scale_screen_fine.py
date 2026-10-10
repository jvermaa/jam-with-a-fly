"""Phase S D2b pure helpers: scale_screen_fine.checks_for / finish, its module constants, and
scale_screen.run_row on a made-up run dict. No simulation is run and no connectome data or
build/ files are read. Only the sanity test reads a results/ file; it is skipped when the
file is absent."""

import copy
import json
import pathlib

import numpy as np
import pytest

from src import scale_screen, scale_screen_fine
from src.timing import Timing

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ---- scale_screen_fine.checks_for ---------------------------------------------------------------


def make_level(runaway_ms=None, kick=3, snare=2, self_sustained=False, last_spike_s=5.04, spikes_after=0):
    return {
        "scale": 0.14,
        "single_kick_from_rest": {"time_to_runaway_ms": runaway_ms},
        "call_run": {
            "motor_spikes": {"Kick": kick, "Snare": snare},
            "last_hit_onset_s": 5.0,
            "after_last_hit_onset": {"last_spike_s": last_spike_s, "spikes_after": spikes_after,
                                     "self_sustained": self_sustained},
        },
    }


def test_checks_for_has_exactly_the_three_screen_checks():
    assert set(scale_screen_fine.checks_for(make_level())) == {
        "no_runaway_after_single_hit",
        "kick_and_snare_motor_spikes_on_call",
        "activity_ends_within_500ms_of_last_hit",
    }


@pytest.mark.parametrize("runaway_ms, expected", [(None, True), (0.0, False), (12.0, False)])
def test_no_runaway_only_when_time_to_runaway_is_none(runaway_ms, expected):
    checks = scale_screen_fine.checks_for(make_level(runaway_ms=runaway_ms))
    assert checks["no_runaway_after_single_hit"] is expected


@pytest.mark.parametrize("kick, snare, expected", [(3, 2, True), (3, 0, False), (0, 2, False), (0, 0, False)])
def test_kick_and_snare_must_both_spike(kick, snare, expected):
    checks = scale_screen_fine.checks_for(make_level(kick=kick, snare=snare))
    assert checks["kick_and_snare_motor_spikes_on_call"] is expected


def test_other_motor_groups_do_not_count_for_the_kick_and_snare_check():
    level = make_level(kick=0, snare=0)
    level["call_run"]["motor_spikes"]["Hat"] = 50
    assert scale_screen_fine.checks_for(level)["kick_and_snare_motor_spikes_on_call"] is False


@pytest.mark.parametrize("self_sustained, expected", [(False, True), (True, False), (None, False)])
def test_activity_ends_only_when_self_sustained_is_exactly_false(self_sustained, expected):
    checks = scale_screen_fine.checks_for(make_level(self_sustained=self_sustained))
    assert checks["activity_ends_within_500ms_of_last_hit"] is expected


# ---- scale_screen_fine.finish -------------------------------------------------------------------


def test_finish_adds_the_reported_fields_and_passes():
    level = scale_screen_fine.finish(make_level())
    assert level["mv_per_synapse"] == round(0.275 * 0.14, 6)
    assert level["call_activity_ends_s"] == 5.04
    assert level["call_activity_ends_ms_after_last_hit"] == 40.0
    assert level["call_spikes_later_than_500ms_after_last_hit"] == 0
    assert level["checks"] == {
        "no_runaway_after_single_hit": True,
        "kick_and_snare_motor_spikes_on_call": True,
        "activity_ends_within_500ms_of_last_hit": True,
    }
    assert level["screen_pass"] is True


def test_finish_returns_the_same_level_and_keeps_its_inputs():
    level = make_level()
    before = copy.deepcopy(level)
    out = scale_screen_fine.finish(level)
    assert out is level
    for key, value in before.items():
        assert out[key] == value


@pytest.mark.parametrize("kwargs, failed", [
    ({"runaway_ms": 8.0}, "no_runaway_after_single_hit"),
    ({"kick": 0}, "kick_and_snare_motor_spikes_on_call"),
    ({"snare": 0}, "kick_and_snare_motor_spikes_on_call"),
    ({"self_sustained": True, "spikes_after": 7}, "activity_ends_within_500ms_of_last_hit"),
    ({"self_sustained": None}, "activity_ends_within_500ms_of_last_hit"),
])
def test_finish_fails_the_screen_when_any_one_check_fails(kwargs, failed):
    level = scale_screen_fine.finish(make_level(**kwargs))
    assert level["screen_pass"] is False
    assert [name for name, ok in level["checks"].items() if not ok] == [failed]


def test_finish_reports_the_late_spike_count():
    level = scale_screen_fine.finish(make_level(self_sustained=True, spikes_after=7))
    assert level["call_spikes_later_than_500ms_after_last_hit"] == 7


def test_finish_with_no_spike_at_all_has_no_end_time():
    level = scale_screen_fine.finish(make_level(last_spike_s=None))
    assert level["call_activity_ends_s"] is None
    assert level["call_activity_ends_ms_after_last_hit"] is None


# ---- scale_screen_fine constants ----------------------------------------------------------------


def test_module_constants():
    assert scale_screen_fine.NEW_SCALES == [0.12, 0.14, 0.16, 0.18]
    assert scale_screen_fine.D2_SCALES == [0.1, 0.2, 0.3]
    assert scale_screen_fine.SCREEN_SEED == 200
    assert scale_screen_fine.MORE_SEEDS == [201, 202, 203]


# ---- scale_screen.run_row -----------------------------------------------------------------------

N_STEPS = 1000  # 2 s at dt 2 ms
DT_MS = 2.0
DURATION_MS = 30.0
TIMING = Timing(82.0)


def make_run(hit_t=(0.5, 1.0), spike_steps=(), motor=None):
    """A made-up run dict; spike_steps are the steps at which one neuron spikes."""
    all_counts = np.zeros(N_STEPS, dtype=np.int64)
    for step in spike_steps:
        all_counts[step] += 1
    motor_counts = np.zeros((2, N_STEPS), dtype=np.int64)
    for (row, step), count in (motor or {}).items():
        motor_counts[row, step] = count
    return {
        "kc": {"n_kc": 10, "kc_pct_active_per_16th_mean": 1.5, "kc_pct_active_per_16th_max": 4.0,
               "kc_mean_rate_hz": 0.25},
        "dt_ms": DT_MS,
        "summary": {"total_spikes_all_neurons": int(all_counts.sum()), "runtime_s": 1.23,
                    "input_spikes_delivered": 42},
        "regions": {"all": {"pct_above_200hz": 0.0, "pct_fired": 0.5}},
        "kc_step_counts": np.zeros(N_STEPS, dtype=np.int64),
        "motor_rate_hz": {"Kick": 1.5, "Snare": 0.5},
        "motor_step_counts": motor_counts,
        "all_step_counts": all_counts,
        "hit_t": np.array(hit_t, dtype=float),
    }


def test_run_row_silent_run_has_no_after_last_hit_fields():
    row = scale_screen.run_row(make_run(hit_t=()), TIMING, DURATION_MS)
    for key in ("after_last_hit", "after_last_hit_onset", "last_hit_onset_s", "input_spikes_delivered"):
        assert key not in row


def test_run_row_call_run_has_both_persistence_measures():
    row = scale_screen.run_row(make_run(), TIMING, DURATION_MS)
    assert "after_last_hit" in row
    assert "after_last_hit_onset" in row
    assert row["last_hit_onset_s"] == 1.0
    assert row["input_spikes_delivered"] == 42


def test_run_row_last_hit_is_the_latest_hit_whatever_the_order():
    row = scale_screen.run_row(make_run(hit_t=(1.0, 0.5)), TIMING, DURATION_MS)
    assert row["last_hit_onset_s"] == 1.0


def test_run_row_onset_window_starts_earlier_than_the_burst_end_window():
    # 2 s run, last hit at 1.0 s: late window is 0.5 s from the onset, 0.47 s from the burst end.
    row = scale_screen.run_row(make_run(), TIMING, DURATION_MS)
    assert row["after_last_hit_onset"]["late_window_s"] == pytest.approx(0.5)
    assert row["after_last_hit"]["late_window_s"] == pytest.approx(0.47)


def test_run_row_spike_between_the_two_windows_counts_from_the_onset_only():
    # step 760 = 1.52 s: later than 1.0 s + 500 ms, earlier than 1.0 s + 30 ms + 500 ms.
    row = scale_screen.run_row(make_run(spike_steps=[760]), TIMING, DURATION_MS)
    assert row["after_last_hit_onset"]["self_sustained"] is True
    assert row["after_last_hit_onset"]["spikes_after"] == 1
    assert row["after_last_hit"]["self_sustained"] is False
    assert row["after_last_hit"]["spikes_after"] == 0
    assert row["after_last_hit_onset"]["last_spike_s"] == 1.52


def test_run_row_spike_before_500ms_after_onset_is_not_self_sustained():
    row = scale_screen.run_row(make_run(spike_steps=[700]), TIMING, DURATION_MS)
    assert row["after_last_hit_onset"]["self_sustained"] is False
    assert row["after_last_hit"]["self_sustained"] is False


def test_run_row_spike_after_both_windows_is_self_sustained_in_both():
    row = scale_screen.run_row(make_run(spike_steps=[900]), TIMING, DURATION_MS)
    assert row["after_last_hit_onset"]["self_sustained"] is True
    assert row["after_last_hit"]["self_sustained"] is True


def test_run_row_motor_spikes_sums_each_motor_row_in_dict_order():
    run = make_run(motor={(0, 10): 2, (0, 500): 3, (1, 999): 4})
    row = scale_screen.run_row(run, TIMING, DURATION_MS)
    assert row["motor_spikes"] == {"Kick": 5, "Snare": 4}
    assert list(row["motor_spikes"]) == ["Kick", "Snare"]
    assert all(type(v) is int for v in row["motor_spikes"].values())
    assert row["motor_rate_hz"] == {"Kick": 1.5, "Snare": 0.5}


def test_run_row_copies_the_summary_numbers():
    row = scale_screen.run_row(make_run(spike_steps=[5, 6, 6]), TIMING, DURATION_MS)
    assert row["total_spikes"] == 3
    assert row["kc_pct_active_per_16th_mean"] == 1.5
    assert row["kc_pct_active_per_16th_max"] == 4.0
    assert row["kc_mean_rate_hz"] == 0.25
    assert row["pct_all_neurons_above_200hz"] == 0.0
    assert row["pct_all_neurons_fired"] == 0.5
    assert row["wall_time_s"] == 1.23
    assert row["time_to_runaway_ms"] is None


# ---- results file: provenance and no personal paths ---------------------------------------------


def test_results_file_has_provenance_and_no_absolute_paths():
    name = "wwry_scale_screen_fine.json"
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"results/{name} absent")
    text = path.read_text()
    result = json.loads(text)
    for key in ("seed", "git_sha", "timestamp"):
        assert key in result
    assert "/Users/" not in text
    assert "/home/" not in text
