"""Phase S, Gate 0: calls/wwry.mid is the stomp-stomp-clap grid, and it equals
bars 1-7 of the reference arrangement's drum track with the hand clap (39) read
as our snare (38). The reference file is third-party, gitignored and absent in
CI, so the comparison tests skip without it. Nothing is written."""

import pathlib

import mido
import pytest

from src import wwry_reference as w

ROOT = pathlib.Path(__file__).resolve().parent.parent
CALL = ROOT / "calls" / "wwry.mid"
KICK, SNARE = 36, 38
EXPECTED_BAR = {KICK: [0, 2, 8, 10], SNARE: [4, 12]}

needs_reference = pytest.mark.skipif(
    not w.REFERENCE.exists(), reason="third-party reference MIDI is not in the repo")


# --- our call on its own (no reference file needed) ------------------------

def test_wwry_call_is_stomp_stomp_clap_in_both_bars():
    grids = w.call_bar_grids(CALL)
    assert len(grids) == 2
    for grid in grids:
        assert grid == EXPECTED_BAR


def test_wwry_call_has_no_other_notes():
    mid = mido.MidiFile(CALL)
    step_ticks = mid.ticks_per_beat // 4
    hits = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                hits.append((msg.note, tick / step_ticks))
    expected = sorted((note, float(bar * 16 + s)) for note, steps in EXPECTED_BAR.items()
                      for bar in range(2) for s in steps)
    assert sorted(hits) == expected


def test_constants_of_the_comparison():
    assert w.CALL == CALL
    assert list(w.BARS) == [0, 1, 2, 3, 4, 5, 6]
    assert w.NOTE_MAP == {39: 38}


def test_mapped_bar_grid_maps_clap_to_snare_and_keeps_the_rest():
    assert w.mapped_bar_grid({36: [0.0, 2.0], 39: [4.0]}) == {36: [0.0, 2.0], 38: [4.0]}
    # a clap and a snare in the same bar are merged, sorted
    assert w.mapped_bar_grid({39: [12.0], 38: [4.0]}) == {38: [4.0, 12.0]}
    # other notes are not dropped, so an extra cymbal would break the match
    assert w.mapped_bar_grid({49: [0.0]}) == {49: [0.0]}


# --- against the reference file --------------------------------------------

@needs_reference
def test_reference_has_all_seven_bars():
    bars, _ = w.reference_bars(w.REFERENCE)
    for b in w.BARS:
        assert bars.get(b), f"reference bar {b + 1} has no drum hits"


@needs_reference
@pytest.mark.parametrize("bar", list(w.BARS))
def test_reference_bar_equals_each_of_our_bars(bar):
    bars, _ = w.reference_bars(w.REFERENCE)
    mapped = w.mapped_bar_grid(bars.get(bar, {}))
    ours = w.call_bar_grids(CALL)
    assert len(ours) == 2
    for our_bar in ours:
        assert mapped == our_bar  # same notes, same steps, nothing else


@needs_reference
def test_reference_bars_hold_only_kick_and_clap():
    bars, _ = w.reference_bars(w.REFERENCE)
    for b in w.BARS:
        assert set(bars[b]) == {36, 39}
        assert bars[b] == {36: [0, 2, 8, 10], 39: [4, 12]}
