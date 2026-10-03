"""Find the 6 hearing-input groups and the 6 motor-output groups (PLAN.md Step 1.1).

Real vs. chosen
  REAL:   which bodies are Johnston's organ (JO) neurons, their subtype, side and
          function label, and which bodies carry each motor subclass. All of that
          is read from the MaleCNS v1.0 annotations, nothing is invented.
  CHOSEN: cutting the JO population into 6 groups, and which subtypes land in
          which group. The fly has no "kick ear" or "snare ear"; the 6-way split
          exists only so that each drum voice has its own input channel. Most JO
          neurons are annotated wind/gravity, not auditory (see the summary file).

Motor groups are selected exactly the way upstream fly_drums_sim.py does it:
FlyBrain.where(subclass=...). flysim.py is imported, not edited.

Usage (from the repo root):  python -m src.groups
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flysim import FlyBrain  # upstream, unmodified
from src.provenance import provenance

SEED = 0
N_GROUPS = 6
JO_TYPE_RE = r"^JO"
MOTOR_SUBCLASSES = ["fl", "ml", "hl", "wm", "hm", "nm"]  # upstream's 6 motor groups
MIN_JO_TOTAL = 50  # PLAN.md stop condition: fewer than this means a wrong filter

ANNOTATIONS = ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
RESULTS = ROOT / "results"
BUILD = ROOT / "build"


def write_annotation_columns(ann):
    """All column names + 20 sample rows, so nothing downstream assumes a column."""
    lines = [f"body-annotations: {ann.shape[0]} rows x {ann.shape[1]} columns", "", "COLUMNS (name: dtype, non-null count)"]
    for c in ann.columns:
        lines.append(f"  {c}: {ann[c].dtype}, {int(ann[c].notna().sum())}")
    lines += ["", "FIRST 20 ROWS", ann.head(20).to_string(max_colwidth=40), ""]
    lines += ["FIRST 20 ROWS WHOSE type STARTS WITH 'JO'", ann[ann["type"].astype("string").str.match(JO_TYPE_RE, na=False)].head(20).to_string(max_colwidth=40), ""]
    (RESULTS / "annotation_columns.txt").write_text("\n".join(lines))


def counts(values):
    names, n = np.unique(np.asarray(values, dtype=str), return_counts=True)
    return {str(k) if k else "(none)": int(v) for k, v in zip(names, n)}


def balanced_groups(type_counts, n_groups):
    """Largest subtype first, each into the currently smallest group.

    Deterministic (ties broken by name, then by lowest group index), so no random
    numbers are needed; SEED is recorded only for the results-file contract.
    """
    groups = [[] for _ in range(n_groups)]
    totals = [0] * n_groups
    for name, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        g = min(range(n_groups), key=lambda i: (totals[i], i))
        groups[g].append(name)
        totals[g] += n
    return groups


def main():
    RESULTS.mkdir(exist_ok=True)
    ann = pd.read_feather(ANNOTATIONS)
    write_annotation_columns(ann)

    fb = FlyBrain()
    ann_i = ann.drop_duplicates(subset=["bodyId"]).set_index("bodyId")
    has_output = np.diff(fb.indptr) > 0  # neuron has >=1 outgoing edge in W

    # ---- hearing input: Johnston's organ ---------------------------------
    jo = fb.where(type_re=JO_TYPE_RE)
    jo_types = fb.types[jo]
    type_counts = counts(jo_types)
    print(f"JO neurons in the graph: {len(jo)} across {len(type_counts)} subtypes")
    for name, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name:14} {n}")
    if len(jo) < MIN_JO_TOTAL:
        raise SystemExit(f"STOP: only {len(jo)} JO neurons (< {MIN_JO_TOTAL}); wrong filter")

    # Side comes from the annotated instance name ("JO-FV_L" / "JO-FV_R").
    side = ann_i["instance"].reindex(fb.bodies[jo]).astype("string").str.extract(r"_([LR])$")[0].fillna("unknown").to_numpy()

    if len(type_counts) >= N_GROUPS:
        method = "subtypes assigned to groups to balance counts (largest first into smallest group)"
        type_groups = balanced_groups(type_counts, N_GROUPS)
        member = [np.isin(jo_types, names) for names in type_groups]
    else:
        method = f"even split sorted by bodyId, seed {SEED}"
        order = np.argsort(fb.bodies[jo])
        member = [np.isin(np.arange(len(jo)), part) for part in np.array_split(order, N_GROUPS)]
        type_groups = [sorted(set(jo_types[m])) for m in member]

    jo_groups, jo_summary = [], []
    for g, m in enumerate(member):
        idx = jo[m]
        jo_groups.append({
            "group": g,
            "types": sorted(type_groups[g]),
            "bodyIds": sorted(int(b) for b in fb.bodies[idx]),
        })
        jo_summary.append({
            "group": g,
            "count": int(m.sum()),
            "with_output_in_W": int(has_output[idx].sum()),
            "side": counts(side[m]),
            "annotated_function": counts(fb.subclass[idx]),
            "types": {t: type_counts[t] for t in sorted(type_groups[g])},
        })

    # ---- motor output: upstream's 6 groups, selected upstream's way ------
    motor_groups, motor_summary = {}, {}
    for sub in MOTOR_SUBCLASSES:
        idx = fb.where(subclass=sub)
        motor_groups[sub] = sorted(int(b) for b in fb.bodies[idx])
        motor_summary[sub] = {"count": len(idx), "superclass": counts(fb.superclass[idx])}

    jo_ok = all(s["count"] > 0 for s in jo_summary) and all(s["side"].get("L", 0) > 0 and s["side"].get("R", 0) > 0 for s in jo_summary)
    motor_ok = all(s["count"] > 0 for s in motor_summary.values())

    prov = provenance(SEED)
    (BUILD / "jo_groups.json").write_text(json.dumps({
        "_comment": "CHOSEN split of REAL Johnston's organ neurons into 6 input groups. See src/groups.py.",
        "method": method, "git_sha": prov["git_sha"], "groups": jo_groups,
    }, indent=1) + "\n")
    (BUILD / "motor_groups.json").write_text(json.dumps({
        "_comment": "Upstream's 6 motor groups: FlyBrain.where(subclass=...), as in fly_drums_sim.py.",
        "git_sha": prov["git_sha"], "groups": motor_groups,
    }, indent=1) + "\n")

    summary = {
        "step": "1.1",
        **prov,
        "seed_note": "grouping is deterministic; no random numbers were drawn",
        "jo": {
            "filter": f"FlyBrain.where(type_re={JO_TYPE_RE!r}) on build/graph.npz",
            "method": method,
            "total": len(jo),
            "n_subtypes": len(type_counts),
            "with_output_in_W": int(has_output[jo].sum()),
            "side": counts(side),
            "annotated_function": counts(fb.subclass[jo]),
            "type_counts": dict(sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "groups": jo_summary,
        },
        "motor": {
            "filter": "FlyBrain.where(subclass=<name>), same as upstream fly_drums_sim.py",
            "groups": motor_summary,
        },
        "pass": {
            "jo_six_nonempty_groups_with_both_sides": bool(jo_ok),
            "jo_total_at_least_50": bool(len(jo) >= MIN_JO_TOTAL),
            "motor_six_nonempty_groups": bool(motor_ok),
        },
    }
    (RESULTS / "groups_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for s in jo_summary:
        print(f"JO group {s['group']}: {s['count']} neurons, sides {s['side']}, {len(s['types'])} subtypes")
    for sub, s in motor_summary.items():
        print(f"motor {sub}: {s['count']} neurons, {s['superclass']}")
    print("pass:", summary["pass"])
    if not all(summary["pass"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
