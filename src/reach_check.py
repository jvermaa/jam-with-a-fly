"""How well can each candidate "ear" reach each motor group through the wiring?

Evidence for the Step 1.1 checkpoint questions (which JO neurons to use, how to
balance them, whether the foreleg group keeps its descending neurons). This is
STRUCTURE ONLY: shortest path length in synaptic hops over the signed graph W.
It says whether a route exists and how short it is. It does not say the route
is strong enough to make a motor neuron fire; only the simulation (Step 1.4)
can show that.

Real vs. chosen: the wiring and the annotations are REAL. Which neurons we call
an "ear" and which we read as a "drum" are CHOSEN.

Usage (from the repo root):  python -m src.reach_check
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flysim import FlyBrain  # upstream, unmodified
from src.provenance import provenance

MAX_HOPS = 6
MOTOR_SUBCLASSES = ["fl", "ml", "hl", "wm", "hm", "nm"]
MOTOR_SUPERCLASSES = ["vnc_motor", "cb_motor"]


def hop_distance(fb, sources, excitatory_only):
    """Fewest synaptic hops from any source to every neuron (-1 = unreachable in MAX_HOPS)."""
    dist = np.full(fb.n, -1, dtype=np.int16)
    dist[sources] = 0
    frontier = np.asarray(sources)
    for hop in range(1, MAX_HOPS + 1):
        nxt = []
        for j in frontier:
            lo, hi = fb.indptr[j], fb.indptr[j + 1]
            tgt = fb.indices[lo:hi]
            if excitatory_only:
                tgt = tgt[fb.wdata[lo:hi] > 0]
            nxt.append(tgt)
        if not nxt:
            break
        new = np.unique(np.concatenate(nxt))
        new = new[dist[new] < 0]
        if len(new) == 0:
            break
        dist[new] = hop
        frontier = new
    return dist


def describe(dist, idx):
    d = dist[idx]
    out = {"n": len(idx), "reached": int((d >= 0).sum())}
    for h in range(1, MAX_HOPS + 1):
        out[f"within_{h}_hops"] = int(((d >= 0) & (d <= h)).sum())
    return out


def main():
    fb = FlyBrain()
    has_output = np.diff(fb.indptr) > 0
    jo = fb.where(type_re=r"^JO")

    ears = {
        "all_JO": jo,
        "all_JO_with_output": jo[has_output[jo]],
        "auditory_JO": jo[fb.subclass[jo] == "auditory"],
        "auditory_JO_with_output": jo[(fb.subclass[jo] == "auditory") & has_output[jo]],
        "wind_gravity_JO_with_output": jo[(fb.subclass[jo] == "wind_gravity") & has_output[jo]],
    }
    drums = {}
    for sub in MOTOR_SUBCLASSES:
        idx = fb.where(subclass=sub)
        is_motor = np.isin(fb.superclass[idx], MOTOR_SUPERCLASSES)
        drums[f"{sub}_as_upstream"] = idx
        if not is_motor.all():
            drums[f"{sub}_motor_only"] = idx[is_motor]
            drums[f"{sub}_descending_only"] = idx[~is_motor]

    result = {"step": "1.1", **provenance(0), "seed_note": "no random numbers used",
              "max_hops": MAX_HOPS, "ears": {}}
    for name, src in ears.items():
        entry = {"n": len(src)}
        for label, exc in (("any_sign", False), ("excitatory_only", True)):
            dist = hop_distance(fb, src, exc)
            entry[label] = {d: describe(dist, idx) for d, idx in drums.items()}
        result["ears"][name] = entry

    out = ROOT / "results" / "reach_check.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    for name, entry in result["ears"].items():
        print(f"\n{name} (n={entry['n']}), excitatory-only paths: reached within 2 / 3 / 4 hops of n")
        for d, s in entry["excitatory_only"].items():
            print(f"  {d:20} {s['within_2_hops']:4} {s['within_3_hops']:4} {s['within_4_hops']:4}  of {s['n']}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
