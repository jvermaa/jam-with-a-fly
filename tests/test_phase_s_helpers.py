"""Phase S pure helpers: sign_audit.tabulate, sim_pool.rate_ceiling_hz / region_stats,
run_fly.total_steps / call_steps. Uses made-up edges, spikes and region labels, so no
connectome data, build/ or results/ files are needed and no simulation is run."""

import math
import types

import numpy as np
import pytest

from src import run_fly, sign_audit, sim_pool
from src.timing import Timing

# ---- sign_audit.tabulate ---------------------------------------------------------------------

# (label, weight in mV, applied sign)
EDGES = [
    ("gaba", -1.5, -1),
    ("acetylcholine", 2.0, 1),
    ("gaba", -0.5, -1),
    ("dopamine", 3.0, 0),
    ("acetylcholine", 0.25, 1),
    ("gaba", 4.0, 1),          # a mislabelled edge: its own sign bucket under the same label
    ("dopamine", -1.0, 0),
    ("acetylcholine", 1.0, 1),
]


def tabulated():
    labels = [e[0] for e in EDGES]
    weight = np.array([e[1] for e in EDGES])
    sign = np.array([e[2] for e in EDGES])
    return sign_audit.tabulate(labels, weight, sign)


def test_tabulate_counts_and_sums():
    assert tabulated() == {
        "acetylcholine": {"excitatory": {"edges": 3, "sum_abs_mv": 3.2}},  # 3.25 rounded to 1 dp
        "dopamine": {"dropped": {"edges": 2, "sum_abs_mv": 4.0}},
        "gaba": {
            "excitatory": {"edges": 1, "sum_abs_mv": 4.0},
            "inhibitory": {"edges": 2, "sum_abs_mv": 2.0},
        },
    }


def test_tabulate_negative_weights_count_as_absolute_value():
    out = sign_audit.tabulate(["gaba", "gaba"], np.array([-2.0, -3.0]), np.array([-1, -1]))
    assert out == {"gaba": {"inhibitory": {"edges": 2, "sum_abs_mv": 5.0}}}


def test_tabulate_sign_names():
    out = sign_audit.tabulate(["x", "x", "x"], np.array([1.0, -1.0, 1.0]), np.array([1, -1, 0]))
    assert set(out["x"]) == {"excitatory", "inhibitory", "dropped"}
    assert all(v == {"edges": 1, "sum_abs_mv": 1.0} for v in out["x"].values())


def test_tabulate_label_with_only_dropped_edges():
    assert list(tabulated()["dopamine"]) == ["dropped"]


def test_tabulate_keys_sorted_by_label():
    out = tabulated()
    assert list(out) == sorted(out)
    assert list(out) == ["acetylcholine", "dopamine", "gaba"]


def test_tabulate_edge_total_is_preserved():
    total = sum(v["edges"] for by_sign in tabulated().values() for v in by_sign.values())
    assert total == len(EDGES)


# ---- sim_pool.rate_ceiling_hz ----------------------------------------------------------------

def fake_brain(refr_steps, dt):
    return types.SimpleNamespace(refr_steps=refr_steps, p=types.SimpleNamespace(dt=dt))


def test_rate_ceiling_at_wide_timestep():
    assert sim_pool.rate_ceiling_hz(fake_brain(2, 2.0)) == 250.0


def test_rate_ceiling_at_fine_timestep():
    assert sim_pool.rate_ceiling_hz(fake_brain(11, 0.2)) == pytest.approx(1000.0 / 2.2)
    assert sim_pool.rate_ceiling_hz(fake_brain(11, 0.2)) == pytest.approx(454.545, abs=1e-3)


# ---- sim_pool.region_stats -------------------------------------------------------------------

SIM_SEC = 2.0
CEILING_HZ = 250.0  # "at ceiling" from 0.95 x 250 = 237.5 Hz
# neuron index -> (region, spikes in SIM_SEC); rate = spikes / 2
NEURONS = [
    ("kc", 400),     # 200 Hz exactly: fired, NOT above 200
    ("kc", 480),     # 240 Hz: above 200 and at the ceiling
    ("kc", 0),       # silent
    ("motor", 500),  # 250 Hz: above 200 and at the ceiling
    ("motor", 200),  # 100 Hz
    ("other", 0),    # silent
]
EMPTY = {"neurons": 0, "pct_fired": None, "pct_above_200hz": None, "pct_at_ceiling": None, "mean_rate_hz": None}


@pytest.fixture
def stats(monkeypatch):
    region = np.array([sim_pool.REGIONS.index(r) for r, _ in NEURONS], dtype=np.int8)
    monkeypatch.setattr(sim_pool, "_region", region)
    spikes = np.repeat(np.arange(len(NEURONS)), [k for _, k in NEURONS])
    rng = np.random.default_rng(0)
    return sim_pool.region_stats(rng.permutation(spikes), len(NEURONS), SIM_SEC, CEILING_HZ)


def test_region_stats_has_all_and_every_region(stats):
    assert list(stats) == ["all"] + sim_pool.REGIONS


def test_region_stats_kc(stats):
    assert stats["kc"] == {
        "neurons": 3,
        "pct_fired": 66.67,
        "pct_above_200hz": 33.33,
        "pct_at_ceiling": 33.33,
        "mean_rate_hz": 146.67,  # (200 + 240 + 0) / 3
    }


def test_region_stats_motor(stats):
    assert stats["motor"] == {
        "neurons": 2,
        "pct_fired": 100.0,
        "pct_above_200hz": 50.0,
        "pct_at_ceiling": 50.0,
        "mean_rate_hz": 175.0,
    }


def test_region_stats_silent_region(stats):
    assert stats["other"] == {
        "neurons": 1, "pct_fired": 0.0, "pct_above_200hz": 0.0, "pct_at_ceiling": 0.0, "mean_rate_hz": 0.0,
    }


def test_region_stats_all(stats):
    assert stats["all"] == {
        "neurons": 6,
        "pct_fired": 66.67,
        "pct_above_200hz": 33.33,
        "pct_at_ceiling": 33.33,
        "mean_rate_hz": 131.67,  # 1580 spikes / 6 neurons / 2 s
    }


def test_region_stats_region_without_neurons_is_none(stats):
    for name in sim_pool.REGIONS:
        if name not in {"kc", "motor", "other"}:
            assert stats[name] == EMPTY


def test_region_stats_neuron_counts_add_up(stats):
    assert sum(stats[r]["neurons"] for r in sim_pool.REGIONS) == stats["all"]["neurons"]


def test_region_stats_exactly_200hz_is_not_above(monkeypatch):
    monkeypatch.setattr(sim_pool, "_region", np.zeros(2, dtype=np.int8))  # both in REGIONS[0]
    # 1 s: neuron 0 at exactly 200 Hz, neuron 1 at 201 Hz.
    neuron = np.repeat(np.array([0, 1]), [200, 201])
    out = sim_pool.region_stats(neuron, 2, 1.0, 250.0)
    assert out["all"]["pct_above_200hz"] == 50.0
    assert out[sim_pool.REGIONS[0]]["pct_above_200hz"] == 50.0


def test_region_stats_ceiling_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr(sim_pool, "_region", np.zeros(3, dtype=np.int8))
    # 1 s, ceiling 200 Hz -> threshold 190 Hz: 189 below, 190 at (counted), 200 at.
    neuron = np.repeat(np.array([0, 1, 2]), [189, 190, 200])
    out = sim_pool.region_stats(neuron, 3, 1.0, 200.0)
    assert out["all"]["pct_at_ceiling"] == 66.67


def test_region_stats_no_spikes(monkeypatch):
    monkeypatch.setattr(sim_pool, "_region", np.zeros(4, dtype=np.int8))
    out = sim_pool.region_stats(np.array([], dtype=np.int64), 4, 1.0, 250.0)
    assert out["all"] == {
        "neurons": 4, "pct_fired": 0.0, "pct_above_200hz": 0.0, "pct_at_ceiling": 0.0, "mean_rate_hz": 0.0,
    }


# ---- run_fly.total_steps / call_steps --------------------------------------------------------

def test_default_steps_unchanged():
    assert run_fly.total_steps() == 2250  # 4.5 s at 120 BPM, dt 2.0 ms
    assert run_fly.call_steps() == 2000   # 4.0 s


def test_default_steps_match_explicit_arguments():
    assert run_fly.total_steps(Timing(120.0), 2.0) == 2250
    assert run_fly.call_steps(Timing(120.0), 2.0) == 2000


def test_total_steps_at_82_bpm_fine_timestep():
    t = Timing(82)
    assert t.sim_sec == pytest.approx(13.5 / 2.05)  # 6.585365... s
    assert run_fly.total_steps(t, 0.2) == math.ceil(t.sim_sec / 0.0002 - 1e-9) == 32927
    assert run_fly.total_steps(t, 2.0) == 3293


def test_total_steps_fine_timestep_is_about_ten_times_the_wide_one():
    t = Timing(82)
    fine, wide = run_fly.total_steps(t, 0.2), run_fly.total_steps(t, 2.0)
    assert fine / wide == pytest.approx(10.0, rel=1e-3)
    assert 10 * (wide - 1) < fine <= 10 * wide  # both are the same duration, rounded up


def test_total_steps_cover_the_whole_simulation():
    t = Timing(82)
    for dt_ms in (0.2, 2.0):
        n = run_fly.total_steps(t, dt_ms)
        assert n * dt_ms / 1000.0 >= t.sim_sec - 1e-9
        assert (n - 1) * dt_ms / 1000.0 < t.sim_sec


def test_call_steps_at_82_bpm():
    t = Timing(82)
    assert run_fly.call_steps(t, 2.0) == 2927   # 5.8536... s / 2 ms, rounded
    assert run_fly.call_steps(t, 0.2) == 29268
    assert run_fly.call_steps(t, 0.2) < run_fly.total_steps(t, 0.2)
