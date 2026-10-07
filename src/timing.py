"""Per-call tempo (Phase S). Everything that turns beats into seconds goes through here.

PLAN.md locked every call to 120 BPM (src/constants.py). The WWRY sprint needs one
call at a different tempo, so the tempo is now looked up per call in
config/calls.json; a call that is not listed there runs at the default 120 BPM and
gets exactly the numbers in src/constants.py.

Real vs. chosen: all of this is CHOSEN (tempo, grid, tail length). None of it
comes from the connectome.
"""

import dataclasses
import json
import pathlib

from src import constants as c

ROOT = pathlib.Path(__file__).resolve().parent.parent
CALLS_CONFIG = ROOT / "config" / "calls.json"

# Sim time after the call, in sixteenth steps. 4 steps = SIM_TAIL_SEC (0.5 s) at
# 120 BPM; counting it in steps keeps the decoder's lag search the same room at
# any tempo.
TAIL_STEPS = 4


@dataclasses.dataclass(frozen=True)
class Timing:
    bpm: float = float(c.BPM)

    @property
    def beat_sec(self):
        return 60.0 / self.bpm

    @property
    def step_sec(self):
        return self.beat_sec * c.BEATS_PER_BAR / c.STEPS_PER_BAR

    @property
    def bar_sec(self):
        return self.beat_sec * c.BEATS_PER_BAR

    @property
    def call_sec(self):
        return self.bar_sec * c.CALL_BARS

    @property
    def tail_sec(self):
        return self.step_sec * TAIL_STEPS

    @property
    def sim_sec(self):
        return self.call_sec + self.tail_sec

    @classmethod
    def for_call(cls, path):
        """Timing of a call file, by its stem (calls/wwry.mid -> "wwry"). None -> default."""
        return cls(float(configured_bpm(path)))


def configured_bpm(path):
    cfg = json.loads(CALLS_CONFIG.read_text())
    if path is None:
        return cfg["default_bpm"]
    return cfg["calls"].get(pathlib.Path(path).stem, {}).get("bpm", cfg["default_bpm"])


DEFAULT = Timing()
