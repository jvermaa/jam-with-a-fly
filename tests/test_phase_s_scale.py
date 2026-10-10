"""Phase S D1-D3 pure helpers: weight_scale.GlobalScale (on a fake brain that is only a
weight array), input_ratio.stats / region_masks / input_counts, scale_screen.checks_for
and taste_search.search. Uses made-up weights, counts and annotations, so no simulation
is run and no connectome data or build/ files are read. Only the sanity tests read
results/ files; each is skipped when its file is absent."""

import json
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

from src import input_ratio, scale_screen, taste_search
from src.weight_scale import GlobalScale

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ---- weight_scale.GlobalScale ------------------------------------------------------------------

# mV per presynaptic spike; positive and negative, like FlyBrain.wdata
WEIGHTS = [0.825, -0.825, 1.375, -4.4, 2.2, 5.5, -1.925, 3.3]


def make_brain(weights=WEIGHTS):
    return types.SimpleNamespace(wdata=np.array(weights, dtype=np.float32))


def test_apply_half_halves_every_weight():
    fb = make_brain()
    original = fb.wdata.copy()
    GlobalScale(fb).apply(0.5)
    np.testing.assert_allclose(fb.wdata, original * 0.5, rtol=1e-6)


def test_apply_writes_in_place():
    fb = make_brain()
    array_before = fb.wdata
    GlobalScale(fb).apply(0.5)
    assert fb.wdata is array_before
    assert fb.wdata.dtype == np.float32
    assert array_before[0] == pytest.approx(0.825 * 0.5)


@pytest.mark.parametrize("scale", [0.5, 0.3, 0.2, 0.1, 0.05])
def test_apply_preserves_signs(scale):
    fb = make_brain()
    original = fb.wdata.copy()
    GlobalScale(fb).apply(scale)
    np.testing.assert_array_equal(np.sign(fb.wdata), np.sign(original))
    assert (fb.wdata != 0).all()


@pytest.mark.parametrize("scale", [0.5, 0.3, 0.2, 0.1, 0.05])
def test_apply_preserves_ratios_between_weights(scale):
    fb = make_brain()
    original = fb.wdata.copy()
    GlobalScale(fb).apply(scale)
    np.testing.assert_allclose(fb.wdata / fb.wdata[0], original / original[0], rtol=1e-5)


def test_apply_one_restores_original_exactly():
    fb = make_brain()
    original = fb.wdata.copy()
    gs = GlobalScale(fb)
    gs.apply(0.3)
    assert not np.array_equal(fb.wdata, original)
    gs.apply(1.0)
    np.testing.assert_array_equal(fb.wdata, original)


def test_two_scales_in_a_row_are_not_cumulative():
    fb = make_brain()
    original = fb.wdata.copy()
    gs = GlobalScale(fb)
    gs.apply(0.5)
    gs.apply(0.2)
    np.testing.assert_allclose(fb.wdata, original * 0.2, rtol=1e-6)
    assert not np.allclose(fb.wdata, original * 0.1)


def test_scale_is_one_before_any_apply_and_weights_untouched():
    fb = make_brain()
    original = fb.wdata.copy()
    gs = GlobalScale(fb)
    assert gs.scale == 1.0
    np.testing.assert_array_equal(fb.wdata, original)


def test_stats_keys_and_values():
    fb = make_brain()
    gs = GlobalScale(fb)
    assert set(gs.stats()) == {"edges", "scale", "mv_per_synapse", "original_sum_abs_mv"}
    assert gs.stats()["edges"] == len(WEIGHTS)
    assert gs.stats()["scale"] == 1.0
    assert gs.stats()["mv_per_synapse"] == pytest.approx(0.275)
    assert gs.stats()["original_sum_abs_mv"] == pytest.approx(sum(abs(w) for w in WEIGHTS), abs=0.06)  # rounded to 1 decimal


def test_stats_follow_the_scale_but_original_sum_does_not():
    fb = make_brain()
    gs = GlobalScale(fb)
    before = gs.stats()["original_sum_abs_mv"]
    gs.apply(0.2)
    s = gs.stats()
    assert s["scale"] == 0.2
    assert s["mv_per_synapse"] == pytest.approx(0.275 * 0.2)
    assert s["edges"] == len(WEIGHTS)
    assert s["original_sum_abs_mv"] == before


# ---- input_ratio.stats ---------------------------------------------------------------------------

def test_stats_on_a_small_array():
    s = input_ratio.stats([0, 0, 10, 20, 70])
    assert set(s) == {"neurons", "mean", "median", "p10", "p90", "neurons_with_zero_input", "total_synapses"}
    assert s["neurons"] == 5
    assert s["mean"] == 20.0
    assert s["median"] == 10.0
    assert s["p10"] == pytest.approx(0.0)
    assert s["p90"] == pytest.approx(50.0)
    assert s["neurons_with_zero_input"] == 2
    assert s["total_synapses"] == 100


def test_stats_without_zero_input_neurons():
    s = input_ratio.stats(np.array([3, 5, 7]))
    assert s["neurons_with_zero_input"] == 0
    assert s["mean"] == 5.0
    assert s["median"] == 5.0
    assert s["total_synapses"] == 15


# ---- input_ratio.region_masks ----------------------------------------------------------------------

SUPERCLASSES = ["cb_intrinsic", "ol_intrinsic", "visual_projection", "descending_neuron",
                "vnc_intrinsic", "vnc_motor", "ascending_neuron", ""]


def test_region_masks_keys_and_shapes():
    masks = input_ratio.region_masks(SUPERCLASSES)
    assert set(masks) == {"whole_cns", "brain_only", "central_brain_without_optic_lobe", "vnc"}
    for m in masks.values():
        assert m.dtype == bool
        assert m.shape == (len(SUPERCLASSES),)


def test_region_masks_whole_cns_is_everything():
    assert input_ratio.region_masks(SUPERCLASSES)["whole_cns"].all()


def test_region_masks_brain_only_is_the_first_four():
    brain = input_ratio.region_masks(SUPERCLASSES)["brain_only"]
    assert brain.tolist() == [True, True, True, True, False, False, False, False]


def test_region_masks_central_brain_excludes_optic_lobe():
    central = input_ratio.region_masks(SUPERCLASSES)["central_brain_without_optic_lobe"]
    assert central.tolist() == [True, False, False, True, False, False, False, False]


def test_region_masks_vnc_is_the_two_vnc_superclasses():
    vnc = input_ratio.region_masks(SUPERCLASSES)["vnc"]
    assert vnc.tolist() == [False, False, False, False, True, True, False, False]


def test_region_masks_accepts_a_numpy_array():
    masks = input_ratio.region_masks(np.array(SUPERCLASSES))
    assert masks["brain_only"].sum() == 4
    assert masks["vnc"].sum() == 2


# ---- input_ratio.input_counts ----------------------------------------------------------------------

def make_index():
    """Body ids 10, 11, 12 are neurons 0, 1, 2; every other body id is not in the graph (-1)."""
    index_of = np.full(20, -1, dtype=np.int64)
    index_of[[10, 11, 12]] = [0, 1, 2]
    return index_of


def test_input_counts_sums_weights_per_postsynaptic_neuron():
    post = np.array([10, 10, 11, 12, 12, 12])
    weight = np.array([3, 4, 5, 1, 1, 6])
    counts = input_ratio.input_counts(post, weight, make_index(), 3)
    assert counts.tolist() == [7.0, 5.0, 8.0]


def test_input_counts_ignores_bodies_not_in_the_index():
    post = np.array([10, 5, 11, 19, 5])
    weight = np.array([3, 100, 5, 200, 300])
    counts = input_ratio.input_counts(post, weight, make_index(), 3)
    assert counts.tolist() == [3.0, 5.0, 0.0]


def test_input_counts_has_length_n_even_when_last_neurons_get_nothing():
    counts = input_ratio.input_counts(np.array([10]), np.array([9]), make_index(), 5)
    assert counts.tolist() == [9.0, 0.0, 0.0, 0.0, 0.0]


def test_input_counts_all_bodies_unknown_gives_zeros():
    counts = input_ratio.input_counts(np.array([1, 2, 3]), np.array([9, 9, 9]), make_index(), 3)
    assert counts.tolist() == [0.0, 0.0, 0.0]


# ---- scale_screen.checks_for -----------------------------------------------------------------------

VOICES = ("Kick", "Snare", "HiHat", "Tom", "Crash", "Ride")
CHECK_KEYS = {"no_runaway_after_single_hit", "call_evokes_motor_activity",
              "info_call_kick_and_snare_motor_both_fire", "info_call_run_no_runaway",
              "info_call_run_kc_sparse", "info_call_run_not_self_sustained"}


def make_level(single_runaway=None, motor=None, call_runaway=None, kc_pct=5.0, self_sustained=False):
    spikes = {v: 0 for v in VOICES}
    spikes.update(motor or {})
    return {
        "single_kick_from_rest": {"time_to_runaway_ms": single_runaway},
        "call_run": {"motor_spikes": spikes, "time_to_runaway_ms": call_runaway,
                     "kc_pct_active_per_16th_mean": kc_pct, "after_last_hit": {"self_sustained": self_sustained}},
    }


def test_checks_for_keys_and_plain_bools():
    checks = scale_screen.checks_for(make_level(motor={"Kick": 3, "Snare": 2}))
    assert set(checks) == CHECK_KEYS
    for value in checks.values():
        assert value is True or value is False
    json.dumps(checks)


def test_no_runaway_after_single_hit_only_when_time_is_none():
    assert scale_screen.checks_for(make_level(single_runaway=None))["no_runaway_after_single_hit"] is True
    assert scale_screen.checks_for(make_level(single_runaway=8.0))["no_runaway_after_single_hit"] is False
    # A runaway at 0 ms is still a runaway: the check is "is None", not falsiness.
    assert scale_screen.checks_for(make_level(single_runaway=0.0))["no_runaway_after_single_hit"] is False


def test_single_hit_check_ignores_the_call_run_runaway():
    checks = scale_screen.checks_for(make_level(single_runaway=None, call_runaway=12.0))
    assert checks["no_runaway_after_single_hit"] is True
    assert checks["info_call_run_no_runaway"] is False


def test_call_evokes_motor_activity_false_when_all_groups_silent():
    assert scale_screen.checks_for(make_level())["call_evokes_motor_activity"] is False


@pytest.mark.parametrize("voice", VOICES)
def test_call_evokes_motor_activity_true_when_any_group_fires(voice):
    checks = scale_screen.checks_for(make_level(motor={voice: 1}))
    assert checks["call_evokes_motor_activity"] is True


def test_kick_and_snare_both_fire_needs_both():
    both = scale_screen.checks_for(make_level(motor={"Kick": 1, "Snare": 1}))
    kick_only = scale_screen.checks_for(make_level(motor={"Kick": 9}))
    snare_only = scale_screen.checks_for(make_level(motor={"Snare": 9}))
    others_only = scale_screen.checks_for(make_level(motor={"HiHat": 9, "Tom": 9}))
    assert both["info_call_kick_and_snare_motor_both_fire"] is True
    assert kick_only["info_call_kick_and_snare_motor_both_fire"] is False
    assert snare_only["info_call_kick_and_snare_motor_both_fire"] is False
    assert others_only["info_call_kick_and_snare_motor_both_fire"] is False
    # ... while motor activity as such is present in all four.
    for checks in (both, kick_only, snare_only, others_only):
        assert checks["call_evokes_motor_activity"] is True


def test_kc_sparse_is_strictly_below_twenty_percent():
    assert scale_screen.KC_SPARSE_BELOW_PCT == 20.0
    assert scale_screen.checks_for(make_level(kc_pct=19.99))["info_call_run_kc_sparse"] is True
    assert scale_screen.checks_for(make_level(kc_pct=20.0))["info_call_run_kc_sparse"] is False
    assert scale_screen.checks_for(make_level(kc_pct=55.0))["info_call_run_kc_sparse"] is False


def test_not_self_sustained_follows_after_last_hit():
    assert scale_screen.checks_for(make_level(self_sustained=False))["info_call_run_not_self_sustained"] is True
    assert scale_screen.checks_for(make_level(self_sustained=True))["info_call_run_not_self_sustained"] is False


# ---- taste_search.search ---------------------------------------------------------------------------

def make_annotations(**columns):
    """A frame with every column in TEXT_COLUMNS; columns not given are all None."""
    n = len(next(iter(columns.values())))
    data = {col: [None] * n for col in taste_search.TEXT_COLUMNS}
    data.update(columns)
    return pd.DataFrame(data)


def test_search_returns_value_counts_only_for_columns_with_matches():
    ann = make_annotations(
        type=["LB1a", "LB1a", "MN9", "KCg-m"],
        synonyms=["sugar projection", None, None, "sugar projection"],
        **{"class": ["gustatory", "gustatory", None, "Kenyon cell"]},
    )
    assert taste_search.search(ann, r"sugar") == {"synonyms": {"sugar projection": 2}}
    assert taste_search.search(ann, r"gustat") == {"class": {"gustatory": 2}}
    assert taste_search.search(ann, r"LB1|sugar") == {"type": {"LB1a": 2}, "synonyms": {"sugar projection": 2}}


def test_search_no_match_gives_empty_dict():
    ann = make_annotations(type=["LB1a", "KCg-m"])
    assert taste_search.search(ann, r"bitter") == {}


def test_search_is_case_insensitive():
    ann = make_annotations(synonyms=["Sugar sensing", "SWEET taste", "gr5a positive", "bitter"])
    found = taste_search.search(ann, taste_search.PATTERNS["sugar"])
    assert found == {"synonyms": {"Sugar sensing": 1, "SWEET taste": 1, "gr5a positive": 1}}


def test_search_counts_are_plain_ints_with_string_keys():
    ann = make_annotations(type=["MN9", "MN9", "MN9"])
    found = taste_search.search(ann, taste_search.PATTERNS["MN9"])
    assert found == {"type": {"MN9": 3}}
    (value, count), = found["type"].items()
    assert type(value) is str
    assert type(count) is int
    json.dumps(found)


def test_search_all_none_frame_matches_nothing():
    ann = make_annotations(type=[None, None])
    assert taste_search.search(ann, r".") == {}


def test_mn9_pattern_respects_word_boundaries():
    ann = make_annotations(type=["MN9", "MN9x", "XMN9", "MN90"],
                           instance=["MN9_R", "MN9(L)", "mn9 left", "preMN9"])
    found = taste_search.search(ann, taste_search.PATTERNS["MN9"])
    assert found["type"] == {"MN9": 1}
    # "_" is a word character, so "MN9_R" is not a whole-word match either.
    assert found["instance"] == {"MN9(L)": 1, "mn9 left": 1}
    assert set(found) == {"type", "instance"}


def test_patterns_cover_the_three_populations():
    assert {"sugar", "bitter", "MN9"} <= set(taste_search.PATTERNS)


# ---- results files: provenance and no personal paths -------------------------------------------------

@pytest.mark.parametrize("name", ["input_ratio.json", "wwry_scale_screen.json", "taste_search.json"])
def test_results_file_has_provenance_and_no_absolute_paths(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"results/{name} absent")
    text = path.read_text()
    result = json.loads(text)
    for key in ("seed", "git_sha", "timestamp"):
        assert key in result
    assert "/Users/" not in text
    assert "/home/" not in text
