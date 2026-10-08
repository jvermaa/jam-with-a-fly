"""Phase S, Gate 1a wrapper that scales KC -> KC weights. Uses a hand-built
8-neuron fake brain (CSC arrays like flysim.FlyBrain), so no connectome data,
build/ files or simulator are needed."""

import types

import numpy as np
import pytest
import scipy.sparse as sp

from src.kc_recurrence import KCRecurrence

TYPES = ["KCab-s", "KCg-m", "KCab-s", "APL", "MBON01", "MBON-KClike", "DNa02", "KCg-m"]
KC = {0, 1, 2, 7}

# (pre, post, weight in mV per presynaptic spike)
EDGES = [
    # KC -> KC (one of them negative, to check signs survive scaling)
    (0, 1, 0.825), (0, 2, 1.375), (1, 0, 2.2), (2, 7, -0.825), (7, 0, 3.3), (7, 7, 1.1),
    # KC -> non-KC
    (0, 4, 5.5), (1, 3, 1.65), (2, 5, 0.825), (7, 6, 2.75),
    # non-KC -> KC
    (3, 0, -4.4), (3, 1, -1.925), (5, 2, 0.825), (6, 7, 1.1),
    # non-KC -> non-KC
    (3, 4, -0.825), (4, 6, 3.025), (5, 4, 1.375), (6, 3, 0.825),
]
N_KC_KC = 6


def make_brain(type_names=TYPES, edges=EDGES):
    """Fake brain in CSC layout: column = presynaptic neuron, indices = postsynaptic targets."""
    n = len(type_names)
    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    w = np.array([e[2] for e in edges], dtype=np.float32)
    m = sp.csc_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    return types.SimpleNamespace(n=n, types=np.array(type_names), indptr=m.indptr.copy(),
                                 indices=m.indices.copy(), wdata=m.data.astype(np.float32).copy())


def edge_list(fb):
    """(pre, post) of every stored weight, in wdata order."""
    pre = np.repeat(np.arange(fb.n), np.diff(fb.indptr))
    return pre, np.asarray(fb.indices)


def kc_kc_mask(fb, kc=KC):
    pre, post = edge_list(fb)
    return np.isin(pre, list(kc)) & np.isin(post, list(kc))


# --- which edges are addressed ---------------------------------------------

def test_fake_brain_has_all_four_edge_kinds():
    fb = make_brain()
    pre, post = edge_list(fb)
    pre_kc, post_kc = np.isin(pre, list(KC)), np.isin(post, list(KC))
    for a in (True, False):
        for b in (True, False):
            assert ((pre_kc == a) & (post_kc == b)).any()


def test_pos_addresses_exactly_the_kc_to_kc_edges():
    fb = make_brain()
    r = KCRecurrence(fb)
    expected = np.flatnonzero(kc_kc_mask(fb))
    assert len(expected) == N_KC_KC
    assert np.array_equal(np.sort(r.pos), expected)
    assert len(np.unique(r.pos)) == len(r.pos)
    assert r.stats()["kc_to_kc_edges"] == N_KC_KC


def test_kc_matching_is_by_name_prefix():
    fb = make_brain()
    r = KCRecurrence(fb)
    assert sorted(r.kc.tolist()) == sorted(KC)
    # "MBON-KClike" (5) and "APL" (3) are not Kenyon cells: none of their edges are addressed.
    pre, post = edge_list(fb)
    assert not np.isin(pre[r.pos], [3, 5]).any()
    assert not np.isin(post[r.pos], [3, 5]).any()


def test_stats_describe_the_original_weights():
    fb = make_brain()
    r = KCRecurrence(fb)
    r.apply(0.25)
    s = r.stats()
    assert s["scale"] == 0.25
    assert s["original_excitatory_edges"] == 5
    assert s["original_inhibitory_edges"] == 1
    assert s["original_mv_per_spike_total"] == pytest.approx(8.0, abs=0.06)


# --- apply -----------------------------------------------------------------

def test_apply_zero_zeroes_only_kc_to_kc_weights():
    fb = make_brain()
    original = fb.wdata.copy()
    mask = kc_kc_mask(fb)
    KCRecurrence(fb).apply(0)
    assert np.all(fb.wdata[mask] == 0)
    assert fb.wdata[~mask].tobytes() == original[~mask].tobytes()
    assert np.all(fb.wdata[~mask] != 0)


@pytest.mark.parametrize("scale", [0.25, 0.5])
def test_apply_scales_the_original_weights(scale):
    fb = make_brain()
    original = fb.wdata.copy()
    mask = kc_kc_mask(fb)
    KCRecurrence(fb).apply(scale)
    assert np.array_equal(fb.wdata[mask], original[mask] * np.float32(scale))
    assert np.allclose(fb.wdata[mask], scale * original[mask].astype(float), rtol=1e-6)
    assert fb.wdata[~mask].tobytes() == original[~mask].tobytes()
    assert fb.wdata.dtype == np.float32


def test_apply_does_not_compound():
    fb = make_brain()
    original = fb.wdata.copy()
    mask = kc_kc_mask(fb)
    r = KCRecurrence(fb)
    r.apply(0.5)
    once = fb.wdata.copy()
    r.apply(0.5)
    assert fb.wdata.tobytes() == once.tobytes()
    assert np.array_equal(fb.wdata[mask], original[mask] * np.float32(0.5))
    # and going through 0 loses nothing
    r.apply(0)
    r.apply(0.25)
    assert np.array_equal(fb.wdata[mask], original[mask] * np.float32(0.25))


def test_apply_one_restores_the_original_exactly():
    fb = make_brain()
    original = fb.wdata.copy()
    r = KCRecurrence(fb)
    for scale in (0, 0.25, 0.5):
        r.apply(scale)
    r.apply(1.0)
    assert fb.wdata.tobytes() == original.tobytes()
    assert r.stats()["scale"] == 1.0


def test_construction_alone_changes_nothing():
    fb = make_brain()
    original = fb.wdata.copy()
    r = KCRecurrence(fb)
    assert fb.wdata.tobytes() == original.tobytes()
    assert r.stats()["scale"] == 1.0


# --- no edge added, removed or re-signed -----------------------------------

@pytest.mark.parametrize("scale", [0, 0.25, 0.5, 1.0])
def test_structure_is_untouched(scale):
    fb = make_brain()
    indptr, indices, n_weights = fb.indptr.copy(), fb.indices.copy(), len(fb.wdata)
    KCRecurrence(fb).apply(scale)
    assert np.array_equal(fb.indptr, indptr)
    assert np.array_equal(fb.indices, indices)
    assert len(fb.wdata) == n_weights


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0])
def test_signs_are_unchanged_for_nonzero_scale(scale):
    fb = make_brain()
    signs = np.sign(fb.wdata).copy()
    KCRecurrence(fb).apply(scale)
    assert np.array_equal(np.sign(fb.wdata), signs)
    assert (signs[kc_kc_mask(fb)] < 0).any()  # the fake brain does hold a negative KC -> KC weight


# --- brains with nothing to scale ------------------------------------------

def test_brain_without_kenyon_cells():
    names = ["APL", "MBON01", "MBON-KClike", "DNa02"]
    fb = make_brain(names, [(0, 1, -0.825), (1, 3, 1.1), (2, 1, 2.2), (3, 0, 0.825), (2, 2, 1.375)])
    original, indptr, indices = fb.wdata.copy(), fb.indptr.copy(), fb.indices.copy()
    r = KCRecurrence(fb)
    assert len(r.kc) == 0
    assert len(r.pos) == 0
    assert r.stats()["kc_to_kc_edges"] == 0
    r.apply(0)
    assert fb.wdata.tobytes() == original.tobytes()
    assert np.array_equal(fb.indptr, indptr)
    assert np.array_equal(fb.indices, indices)


def test_brain_with_kenyon_cells_but_no_kc_to_kc_edges():
    names = ["KCab-s", "KCg-m", "APL", "MBON01"]
    fb = make_brain(names, [(0, 3, 5.5), (1, 2, 1.65), (2, 0, -4.4), (2, 1, -1.925), (3, 2, 0.825)])
    original = fb.wdata.copy()
    r = KCRecurrence(fb)
    assert sorted(r.kc.tolist()) == [0, 1]
    assert len(r.pos) == 0
    assert r.stats()["kc_to_kc_edges"] == 0
    r.apply(0)
    assert fb.wdata.tobytes() == original.tobytes()
    assert r.stats()["scale"] == 0.0
