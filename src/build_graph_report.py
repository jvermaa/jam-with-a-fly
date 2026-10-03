"""Run the upstream build_graph.py untouched and record what it produced.

PLAN.md Step 0.4. build_graph.py is THEIRS and is not edited: this wrapper only
calls its main(), then reads build/graph.npz back and writes the counts to
results/build_graph.json so that every reported number comes from a saved run.

Real vs. chosen: the neurons, the synapse counts and the neurotransmitter
labels are REAL (MaleCNS v1.0). The >=3-synapse cutoff, the 0.275 mV per
synapse scale and the sign-per-neurotransmitter rule are upstream modelling
choices (Shiu et al. 2024), not ours and not measured per synapse.

Usage (from the repo root):  python -m src.build_graph_report
"""

import contextlib
import datetime
import io
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_graph  # upstream, unmodified

EXPECTED_NEURONS = 165_122  # PLAN.md Step 0.4 pass check; never loosen.

# build_graph.py reads these short names; the download has the long ones.
DATA_FILES = {
    "connectome-weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "body-annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
}


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def link_data_files():
    """Point the short names build_graph.py expects at the downloaded files."""
    for short, full in DATA_FILES.items():
        target = build_graph.DATA / full
        if not target.exists():
            raise SystemExit(f"missing {target}; download it first (PLAN.md Step 0.4)")
        link = build_graph.DATA / short
        if not link.exists():
            link.symlink_to(full)


def main():
    link_data_files()

    log = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        build_graph.main()
    runtime_s = time.time() - t0

    g = np.load(build_graph.BUILD / "graph.npz", allow_pickle=False)
    data = g["data"]
    n = int(g["shape"][0])
    # W holds synapses * 0.275 mV * sign, so |W| / 0.275 recovers synapse counts.
    synapses = int(np.rint(np.abs(data).astype(np.float64) / build_graph.MV_PER_SYNAPSE).sum())

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())

    result = {
        "step": "0.4",
        "seed": None,
        "seed_note": "build_graph.py is deterministic and uses no random numbers",
        "git_sha": git_sha,
        "git_dirty": dirty,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "runtime_s": round(runtime_s, 2),
        "neurons": n,
        "expected_neurons": EXPECTED_NEURONS,
        "pass": n == EXPECTED_NEURONS,
        "edges_nonzero": int(data.size),
        "edges_excitatory": int((data > 0).sum()),
        "edges_inhibitory": int((data < 0).sum()),
        "synapses_in_W": synapses,
        "synapses_note": "synapses on the signed edges kept in W (pairs with >=3 synapses, "
        "both ends traced neurons, modulatory/unknown-transmitter edges dropped)",
        "graph_npz_bytes": (build_graph.BUILD / "graph.npz").stat().st_size,
        "data_files": {
            full: (build_graph.DATA / full).stat().st_size for full in DATA_FILES.values()
        },
        # Repo-relative paths only: results are public, absolute paths are not.
        "build_graph_stdout": log.getvalue().replace(str(ROOT) + "/", "").splitlines(),
    }

    out = ROOT / "results" / "build_graph.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"neurons = {n:,} (expected {EXPECTED_NEURONS:,}) -> {'PASS' if result['pass'] else 'FAIL'}")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
