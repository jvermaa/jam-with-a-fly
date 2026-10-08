"""Input-ratio check (Phase S, D1): how many input synapses does a neuron get in our graph?

No simulation is run. The model's 0.275 mV per synapse comes from Shiu et al.
2024, calibrated on the FlyWire female brain. If MaleCNS neurons carry a
different number of detected input synapses, the same per-synapse value gives a
different total drive. This reports the per-neuron input synapse count at four
levels of filtering, for the whole CNS and for the brain alone.

FlyWire's own numbers are NOT in this repository or its data folder, so no
ratio and no implied scale are computed here; none is guessed.

Real vs. chosen: synapse counts are REAL (MaleCNS v1.0 tables). The filters are
upstream's (build_graph.py). What counts as "brain" below is CHOSEN by us, by
annotated superclass.

Usage (from the repo root):  python -m src.input_ratio
"""

import json
import pathlib
import sys

import numpy as np
import pyarrow.feather as pf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_graph  # upstream, unmodified: MIN_SYN, MV_PER_SYNAPSE
from src.provenance import provenance

DATA = ROOT / "data"
GRAPH = ROOT / "build" / "graph.npz"

# CHOSEN grouping of annotated superclasses. "brain" = neurons whose inputs lie in the brain:
# central-brain and optic-lobe neurons plus descending neurons (dendrites in the brain).
BRAIN_PREFIXES = ("cb_", "ol_", "visual_")
BRAIN_EXTRA = ("descending_neuron", "efferent_descending")
OPTIC_PREFIXES = ("ol_", "visual_")
VNC_PREFIXES = ("vnc_",)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"neurons": len(x), "mean": round(float(x.mean()), 2), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "neurons_with_zero_input": int((x == 0).sum()), "total_synapses": int(x.sum())}


def region_masks(superclass):
    sc = np.asarray(superclass).astype(str)
    starts = lambda prefixes: np.array([s.startswith(prefixes) for s in sc])
    brain = starts(BRAIN_PREFIXES) | np.isin(sc, BRAIN_EXTRA)
    return {
        "whole_cns": np.ones(len(sc), dtype=bool),
        "brain_only": brain,
        "central_brain_without_optic_lobe": brain & ~starts(OPTIC_PREFIXES),
        "vnc": starts(VNC_PREFIXES),
    }


def input_counts(post, weight, index_of, n):
    """Synapses onto each of the n neurons; post bodies not in the index are ignored."""
    row = index_of[post]
    ok = row >= 0
    return np.bincount(row[ok], weights=weight[ok], minlength=n)


def main():
    z = np.load(GRAPH, allow_pickle=False)
    bodies, n = z["bodies"], len(z["bodies"])
    masks = region_masks(z["superclass"])

    tbl = pf.read_table(DATA / "connectome-weights.feather")
    pre, post, wt = (tbl.column(k).to_numpy() for k in ("body_pre", "body_post", "weight"))
    pairs_on_disk = len(wt)
    del tbl
    index_of = np.full(int(max(pre.max(), post.max())) + 1, -1, dtype=np.int64)
    index_of[bodies] = np.arange(n)

    traced_pre = index_of[pre] >= 0
    kept = traced_pre & (wt >= build_graph.MIN_SYN)
    levels = {
        "1_any_presynaptic_body": ("every synapse onto a traced neuron in the raw table, including from untraced fragments",
                                   input_counts(post, wt, index_of, n)),
        "2_traced_to_traced": ("presynaptic body is also a traced neuron; no threshold",
                               input_counts(post[traced_pre], wt[traced_pre], index_of, n)),
        "3_traced_pairs_with_3_or_more": (f"as 2, pairs with >= {build_graph.MIN_SYN} synapses (the builder's candidates)",
                                          input_counts(post[kept], wt[kept], index_of, n)),
    }
    # Level 4: what the simulation actually uses (sign-0 presynaptic neurons dropped). Row = postsynaptic.
    per_row = np.add.reduceat(np.abs(z["data"]).astype(np.float64), z["indptr"][:-1]) if len(z["data"]) else np.zeros(n)
    per_row[np.diff(z["indptr"]) == 0] = 0.0
    in_sim = np.rint(per_row / build_graph.MV_PER_SYNAPSE)
    levels["4_in_simulation"] = ("as 3, minus edges from neurons with sign 0 (monoamines, unclear, unknown): the weight matrix", in_sim)
    exc = np.zeros(n)
    data, indptr = z["data"].astype(np.float64), z["indptr"]
    row_of = np.repeat(np.arange(n), np.diff(indptr))
    exc = np.bincount(row_of[data > 0], weights=data[data > 0], minlength=n) / build_graph.MV_PER_SYNAPSE
    inh = np.bincount(row_of[data < 0], weights=-data[data < 0], minlength=n) / build_graph.MV_PER_SYNAPSE

    result = {
        "phase": "S",
        "step": "D1",
        **provenance(None),
        "seed_note": "no random numbers used",
        "dataset": "MaleCNS v1.0 (connectome-weights, minconf 0.5)",
        "neurons": n,
        "pairs_in_raw_table": pairs_on_disk,
        "region_definition": {"brain_only": f"superclass starting with {BRAIN_PREFIXES} or in {BRAIN_EXTRA}",
                              "central_brain_without_optic_lobe": f"brain_only minus superclass starting with {OPTIC_PREFIXES}",
                              "vnc": f"superclass starting with {VNC_PREFIXES}", "note": "CHOSEN grouping"},
        "input_synapses_per_neuron": {
            name: {"what": what, **{region: stats(counts[m]) for region, m in masks.items()}}
            for name, (what, counts) in levels.items()
        },
        "in_simulation_excitatory_vs_inhibitory": {
            region: {"excitatory_mean": round(float(exc[m].mean()), 2), "inhibitory_mean": round(float(inh[m].mean()), 2),
                     "excitatory_median": round(float(np.median(exc[m])), 1), "inhibitory_median": round(float(np.median(inh[m])), 1)}
            for region, m in masks.items()
        },
        "flywire_783": {
            "available_locally": False,
            "searched": ["data/ (three MaleCNS feather files only)", "reference_data/", "docs/", "repository text files"],
            "ratio_malecns_over_flywire": None,
            "implied_scale_flywire_over_malecns": None,
            "note": "No FlyWire table or published FlyWire statistic is stored locally; no number is assumed.",
        },
    }
    out = ROOT / "results" / "input_ratio.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    for name, level in result["input_synapses_per_neuron"].items():
        print(name, {r: (level[r]["neurons"], level[r]["mean"], level[r]["median"]) for r in masks})
    print("exc vs inh:", result["in_simulation_excitatory_vs_inhibitory"])
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
