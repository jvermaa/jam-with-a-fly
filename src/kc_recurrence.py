"""Wrapper that scales the Kenyon-cell -> Kenyon-cell weights (Phase S, Gate 1a).

Why: in the graph as built by upstream build_graph.py, a KC -> KC contact is an
ordinary excitatory (cholinergic) synapse like any other. With those edges at
full strength every Kenyon cell holds every other at the model's maximum rate,
at any tonic drive, with or without sound (results/wwry_saturation.json).

Real vs. chosen
  REAL:   the KC -> KC contacts exist in the connectome, and this file does not
          add, remove or re-sign any edge.
  CHOSEN: how strongly they excite. Scaling them down (0 = no effect) treats
          KC-KC contacts as non-excitatory. Real KC-KC contacts are mostly
          axo-axonic, so reading them as plain spike-to-spike excitation is
          itself an assumption; turning that excitation down is our modelling
          choice, not something the connectome states. Any result produced
          with a scale other than 1 must say so.

flysim.py is not edited. Like upstream mushroom.MushroomBody.apply(), this
writes into FlyBrain.wdata, the weight array the running simulation reads, and
keeps the original values so the change is exactly reversible.
"""

import numpy as np

KC_TYPE_PREFIX = "KC"


class KCRecurrence:
    def __init__(self, fb):
        self.fb = fb
        types = np.asarray(fb.types).astype(str)
        self.kc = np.flatnonzero(np.char.startswith(types, KC_TYPE_PREFIX))
        is_kc = np.zeros(fb.n, dtype=bool)
        is_kc[self.kc] = True
        # fb is CSC: column k holds the targets of presynaptic neuron k.
        pos = [np.arange(fb.indptr[k], fb.indptr[k + 1])[is_kc[fb.indices[fb.indptr[k]:fb.indptr[k + 1]]]]
               for k in self.kc]
        self.pos = np.concatenate(pos).astype(np.int64) if pos else np.zeros(0, dtype=np.int64)
        self.base = fb.wdata[self.pos].copy()
        self.scale = 1.0

    def apply(self, scale):
        """Set every KC -> KC weight to scale x its original value. scale=1 restores the graph as built."""
        self.scale = float(scale)
        if len(self.pos):
            self.fb.wdata[self.pos] = self.base * np.float32(self.scale)

    def stats(self):
        return {
            "kc_to_kc_edges": len(self.pos),
            "scale": self.scale,
            "original_mv_per_spike_total": round(float(self.base.sum()), 1),
            "original_excitatory_edges": int((self.base > 0).sum()),
            "original_inhibitory_edges": int((self.base < 0).sum()),
        }
