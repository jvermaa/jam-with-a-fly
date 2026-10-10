"""Sign audit (Phase S, C1): what sign does the simulation actually apply to each edge?

No simulation is run. Two views of the same edges:

  applied:   the weight matrix the simulator reads (build/graph.npz, written by
             upstream build_graph.py). Every edge in it, grouped by the
             presynaptic neuron's transmitter and by the sign of the weight.
  dropped:   edges build_graph.py considered (>= 3 synapses, both ends traced
             neurons) but left out of the matrix because the presynaptic
             transmitter has sign 0. Recomputed here from the raw tables with
             the same filters; the count is checked against the builder's own.

The transmitter the builder uses is `consensus_nt`. The raw table also has a
per-body `predicted_nt`; the same edges are tabulated by that column too, so a
disagreement between the two columns is visible.

Real vs. chosen: transmitter labels are predictions shipped with MaleCNS
(REAL data, with their own error rate). Mapping a transmitter to a sign, and
0.275 mV per synapse, are CHOSEN by upstream (after Shiu et al. 2024).

Nothing is changed by this script. Usage (from the repo root):  python -m src.sign_audit
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_graph  # upstream, unmodified: SIGN table, MIN_SYN, MV_PER_SYNAPSE
from src.provenance import provenance

DATA = ROOT / "data"
GRAPH = ROOT / "build" / "graph.npz"

# The convention of the model this simulator follows (Shiu et al. 2024, as described in flysim.py
# and CLAUDE.md): acetylcholine excitatory, GABA and glutamate inhibitory.
REFERENCE_CONVENTION = {"acetylcholine": "excitatory", "gaba": "inhibitory", "glutamate": "inhibitory"}


def tabulate(labels, weight_mv, applied_sign):
    """{label: {sign: {"edges", "sum_abs_mv"}}} with sign in excitatory / inhibitory / dropped."""
    out = {}
    names = np.array(["inhibitory", "dropped", "excitatory"])[np.sign(applied_sign).astype(int) + 1]
    df = pd.DataFrame({"label": labels, "sign": names, "abs_mv": np.abs(weight_mv)})
    for (label, sign), g in df.groupby(["label", "sign"]):
        out.setdefault(str(label), {})[str(sign)] = {"edges": len(g), "sum_abs_mv": round(float(g["abs_mv"].sum()), 1)}
    return dict(sorted(out.items()))


def main():
    z = np.load(GRAPH, allow_pickle=False)
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).tocoo()
    bodies, nt_graph = z["bodies"], z["nt"].astype(str)
    n = len(bodies)

    # ---- every candidate edge, with the builder's own filters --------------------------------
    tbl = pf.read_table(DATA / "connectome-weights.feather")
    tbl = tbl.filter(pc.greater_equal(tbl.column("weight"), build_graph.MIN_SYN))
    pre, post = tbl.column("body_pre").to_numpy(), tbl.column("body_post").to_numpy()
    syn = tbl.column("weight").to_numpy().astype(np.float64)
    del tbl
    valid = np.zeros(int(max(pre.max(), post.max(), bodies.max())) + 1, dtype=bool)
    valid[bodies] = True
    ok = valid[pre] & valid[post]
    pre, post, syn = pre[ok], post[ok], syn[ok]
    idx = np.full(len(valid), -1, dtype=np.int64)
    idx[bodies] = np.arange(n)
    pre_i = idx[pre]

    nt_tbl = pd.read_feather(DATA / "body-neurotransmitters.feather").dropna(subset=["body"]).drop_duplicates(subset=["body"])
    nt_tbl = nt_tbl.set_index("body")
    consensus_raw = nt_tbl["consensus_nt"].reindex(bodies)
    predicted = nt_tbl["predicted_nt"].reindex(bodies).fillna("NaN (no prediction)").str.lower().to_numpy().astype(str)
    consensus = consensus_raw.fillna("NaN (no entry)").str.lower().to_numpy().astype(str)

    sign_of_body = np.array([build_graph.SIGN.get(s, 0.0) for s in nt_graph])
    applied = sign_of_body[pre_i]
    mv = syn * build_graph.MV_PER_SYNAPSE  # magnitude the edge has, or would have had if not dropped

    by_consensus = tabulate(consensus[pre_i], mv, applied)
    by_graph_label = tabulate(nt_graph[pre_i], mv, applied)
    by_predicted = tabulate(predicted[pre_i], mv, applied)

    # ---- cross-check against the matrix the simulator actually reads -------------------------
    w_sign = np.sign(W.data)
    in_matrix = tabulate(nt_graph[W.col], W.data, w_sign)
    recomputed_kept = int((applied != 0).sum())
    consistency = {
        "edges_in_matrix": int(W.nnz),
        "edges_recomputed_as_kept": recomputed_kept,
        "match": recomputed_kept == int(W.nnz),
        "sum_abs_mv_in_matrix": round(float(np.abs(W.data).sum()), 1),
        "sum_abs_mv_recomputed_kept": round(float(mv[applied != 0].sum()), 1),
        "every_matrix_edge_sign_equals_presynaptic_sign": bool(np.array_equal(w_sign, sign_of_body[W.col])),
    }

    mismatches = []
    for nt_name, expected in REFERENCE_CONVENTION.items():
        got = [s for s in in_matrix.get(nt_name, {}) if s != expected]
        if got or expected not in in_matrix.get(nt_name, {}):
            mismatches.append({"transmitter": nt_name, "expected": expected, "found": sorted(in_matrix.get(nt_name, {}))})

    neurons = pd.Series(nt_graph).value_counts().to_dict()
    result = {
        "phase": "S",
        "step": "C1",
        **provenance(None),
        "seed_note": "audit only; no simulation, no random numbers",
        "graph": str(GRAPH.relative_to(ROOT)),
        "transmitter_column_used_by_builder": "consensus_nt (missing -> 'unknown')",
        "sign_table_applied": {k: v for k, v in build_graph.SIGN.items()},
        "sign_for_any_other_label": 0.0,
        "mv_per_synapse": build_graph.MV_PER_SYNAPSE,
        "min_synapses_per_edge": build_graph.MIN_SYN,
        "neurons_per_transmitter_label": {str(k): int(v) for k, v in neurons.items()},
        "neurons_with_no_consensus_entry": int(consensus_raw.isna().sum()),
        "candidate_edges": len(syn),
        "edges_in_simulator_matrix_by_presynaptic_transmitter": in_matrix,
        "all_candidate_edges_by_consensus_nt": by_consensus,
        "all_candidate_edges_by_label_stored_in_graph": by_graph_label,
        "all_candidate_edges_by_predicted_nt": by_predicted,
        "consistency": consistency,
        "answers": {
            "glutamate": sorted(in_matrix.get("glutamate", {})),
            "unknown_or_missing": "label 'unknown' (includes bodies with no consensus entry) and 'unclear' get sign 0: "
                                  "every edge from such a neuron is dropped from the matrix, so the neuron can be "
                                  "driven but excites or inhibits nothing",
            "monoamines": "dopamine, serotonin, octopamine get sign 0: edges dropped",
            "histamine": "sign -1 (inhibitory); not part of the reference convention's three classes",
        },
        "reference_convention": REFERENCE_CONVENTION,
        "mismatches_vs_reference_convention": mismatches,
    }
    out = ROOT / "results" / "sign_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("by consensus_nt (all candidate edges):")
    for k, v in by_consensus.items():
        print(f"  {k:18} {v}")
    print("by predicted_nt (all candidate edges):")
    for k, v in by_predicted.items():
        print(f"  {k:18} {v}")
    print("consistency:", consistency)
    print("mismatches:", mismatches)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
