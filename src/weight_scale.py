"""Wrapper that scales every synaptic weight by one global factor (Phase S, D2).

Why: with the graph as built, any spark ignites a self-sustained runaway state
(results/wwry_reset_transient.json): a single drum hit on a resting network has
more than half the Kenyon cells firing within 6-10 ms. The 0.275 mV per synapse
the model uses was calibrated by Shiu et al. 2024 on a different connectome
(FlyWire, female brain); nothing says it carries over to MaleCNS unchanged.

Real vs. chosen
  REAL:   every edge, its sign and its synapse count. This file adds, removes
          and re-signs nothing; the ratio between any two weights is unchanged.
  CHOSEN: the factor. One number multiplies all weights, i.e. it replaces
          0.275 mV per synapse by scale x 0.275 mV. Any result produced with a
          scale other than 1 must say so.

flysim.py is not edited. Like upstream mushroom.MushroomBody.apply(), this
writes into FlyBrain.wdata, the weight array the running simulation reads, and
keeps the original values so the change is exactly reversible.
"""

import numpy as np


class GlobalScale:
    def __init__(self, fb):
        self.fb = fb
        self.base = fb.wdata.copy()
        self.scale = 1.0

    def apply(self, scale):
        """Set every weight to scale x its original value. scale=1 restores the graph as built."""
        self.scale = float(scale)
        self.fb.wdata[:] = self.base * np.float32(self.scale)

    def stats(self):
        return {
            "edges": len(self.base),
            "scale": self.scale,
            "mv_per_synapse": round(0.275 * self.scale, 6),
            "original_sum_abs_mv": round(float(np.abs(self.base).sum()), 1),
        }
