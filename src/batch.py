"""Best-of-N batch for the untrained fly (PLAN.md Step 1.6).

Runs calls 01-05 x seeds 0-1 (10 runs), each against the silent baseline of the
same seed, decodes and scores every run, and ranks them by: number of voices
used, then echo F1, then whether hit density is within 10-40%.

Picking the best of 10 is a selection we make; anything shown from this batch
must be described as "best of 10 runs".

Usage (from the repo root):  python -m src.batch
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flysim import FlyBrain  # upstream, unmodified
from src import decode, run_fly
from src.provenance import provenance

CALLS = [f"call_{i:02d}" for i in range(1, 6)]
SEEDS = [0, 1]
DENSITY_RANGE = (0.10, 0.40)
RESULTS = ROOT / "results"


def main():
    fb = FlyBrain(p=run_fly.SimParams())
    runs = []
    for seed in SEEDS:
        silent_arrays, silent_summary = run_fly.run(fb, None, seed)
        silent_npz = RESULTS / f"silent_seed{seed}.npz"
        np.savez_compressed(silent_npz, **silent_arrays)
        print(f"silent seed {seed}: {silent_summary['runtime_s']} s", flush=True)

        for call in CALLS:
            arrays, summary = run_fly.run(fb, ROOT / "calls" / f"{call}.mid", seed)
            run_npz = RESULTS / f"run_{call}_seed{seed}.npz"
            np.savez_compressed(run_npz, **arrays)
            r = decode.decode_run(run_npz, silent_npz)
            identical = bool(np.array_equal(arrays["motor_step_counts"], silent_arrays["motor_step_counts"]))
            runs.append({
                "run_id": r["run_id"],
                "call": call,
                "seed": seed,
                "voices_used": r["voices_used"],
                "f1": r["f1"],
                "baseline_f1": r["baseline_f1"],
                "n_answer_hits": r["n_answer_hits"],
                "hit_density": r["hit_density"],
                "density_in_range": DENSITY_RANGE[0] <= r["hit_density"] <= DENSITY_RANGE[1],
                "lag_steps": r["lag_steps"],
                "not_silent_not_saturated": r["not_silent_not_saturated"],
                "motor_identical_to_silent": identical,
                "runtime_s": summary["runtime_s"],
                "answer_mid": r["answer_mid"],
                "score_json": f"results/score_{r['run_id']}.json",
            })
            print(f"{r['run_id']}: voices {r['voices_used']}, F1 {r['f1']:.3f} (random {r['baseline_f1']:.3f}), "
                  f"{r['n_answer_hits']} hits, lag {r['lag_steps']}, {summary['runtime_s']} s", flush=True)

    ranked = sorted(runs, key=lambda r: (-r["voices_used"], -r["f1"], not r["density_in_range"], r["run_id"]))
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    result = {
        "step": "1.6",
        **provenance(SEEDS),
        "ranking": "voices_used (more first), then f1 (higher first), then hit density within 10-40%",
        "n_runs": len(ranked),
        "mean_f1": float(np.mean([r["f1"] for r in ranked])),
        "mean_baseline_f1": float(np.mean([r["baseline_f1"] for r in ranked])),
        "any_usable": any(r["not_silent_not_saturated"] for r in ranked),
        "runs": ranked,
    }
    (RESULTS / "batch_untrained.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote results/batch_untrained.json; mean F1 {result['mean_f1']:.3f} vs random {result['mean_baseline_f1']:.3f}")
    if not result["any_usable"]:
        raise SystemExit("STOP: every answer is silent or saturated")


if __name__ == "__main__":
    main()
