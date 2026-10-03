"""The timing constants in PLAN.md section 1 must agree with each other and
with config/voices.json."""

import json
import pathlib

import pytest

from src import constants as c

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_timing_constants_are_consistent():
    assert c.BAR_SEC == pytest.approx(c.BEATS_PER_BAR * 60.0 / c.BPM)
    assert c.CALL_SEC == pytest.approx(c.CALL_BARS * c.BAR_SEC)
    assert c.STEP_SEC == pytest.approx(c.BAR_SEC / c.STEPS_PER_BAR)
    assert c.CALL_STEPS == c.CALL_BARS * c.STEPS_PER_BAR


def test_voice_table():
    voices = json.loads((ROOT / "config" / "voices.json").read_text())["voices"]
    assert [v["idx"] for v in voices] == list(range(6))
    assert len({v["midi_note"] for v in voices}) == 6
    assert sorted(v["motor_group"] for v in voices) == ["fl", "hl", "hm", "ml", "nm", "wm"]
    # Echo requires input voice i <-> output voice i.
    assert [v["jo_group"] for v in voices] == list(range(6))
