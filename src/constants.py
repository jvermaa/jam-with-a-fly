"""Timing constants for Fly Drum Jam (PLAN.md section 1, "Timing constants").

Real vs. chosen: everything in this file is CHOSEN by us (tempo, grid, scoring
tolerances, sim timestep). None of it comes from the connectome.
"""

BPM = 120
BEATS_PER_BAR = 4  # 4/4

BAR_SEC = 2.0
CALL_BARS = 2
CALL_SEC = 4.0

STEPS_PER_BAR = 16  # 16th notes
STEP_SEC = 0.125
CALL_STEPS = 32

# Extra sim time after the call, for late responses.
SIM_TAIL_SEC = 0.5

# The upstream repo's speed tradeoff. The validated default for this kind of
# LIF model (Shiu et al. 2024) is 0.2 ms; 2.0 ms is 10x coarser. Must be noted
# in the credits.
DT_MS = 2.0

HIT_TOLERANCE_STEPS = 1
MAX_LAG_STEPS = 2
