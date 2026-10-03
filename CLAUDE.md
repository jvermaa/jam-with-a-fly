# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A simulated male fruit fly (MaleCNS v1.0 connectome, ~165k neurons) hears a 2-bar drum beat through hearing neurons and drums an answer back. Forked from `sykeriin/fly-drums`. `PLAN.md` is the v1 roadmap (step-by-step, with human checkpoints); read it before starting planned work. `docs/ORIGINAL_README.md` is the upstream README.

## Commands

```bash
pip install -r requirements.txt
# fetch the connectome (~1.1 GB, CC-BY, no login; links in PLAN.md) into data/
python build_graph.py            # data/*.feather -> build/graph.npz (signed sparse W)
python mb_sides.py               # -> build/mb_sides.json (MBON dopamine side table)
python fly_drums_sim.py          # recorded CPU run -> fly_drums_export.json (viewed by real-brain.html)
python fly_drums_live_server.py  # live GPU server on http://localhost:4670 (needs torch + CUDA; torch is commented out in requirements.txt)
```

CI (`.github/workflows/ci.yml`) runs ruff on `src/` and `tools/` only (the legacy sim files are excluded), checks that every local image path in `README.md` exists, and runs `pytest` only if a `tests/` dir exists. There are currently no tests. `data/`, `build/graph.npz`, `checkpoints/` and `*.feather` are gitignored; never commit connectome data or large derived datasets.

## Architecture

Data flow: `build_graph.py` → `build/graph.npz` → `flysim.FlyBrain` (CPU) / `flysim_gpu.FlyBrainGPU` (batched GPU) → `fly_drums_sim.py` (recorded) or `fly_drums_live_server.py` (streamed over WebSocket) → `real-brain.html` (three.js viewer, uses `models/drum_kit.glb`).

- **`build_graph.py`** — builds the CSR weight matrix (row = post, col = pre, mV per presynaptic spike = synapses × 0.275 × sign). Sign from neurotransmitter (ACh +, GABA/glutamate/histamine −, monoamines 0 and dropped). Keeps only pairs with ≥3 synapses and only "Traced" non-glia bodies.
- **`flysim.py`** — ground-truth LIF simulator (Shiu et al. 2024 approach). Wiring is fixed; the only free parameters are per-cell-type `gains`. Runs can resume from a returned `_state`.
- **`flysim_gpu.py`** — same step order and same return/`_state` format as `flysim.py`, with B independent runs as columns of one sparse-dense product. Random kicks deliberately come from host numpy Generators (not torch) so states are interchangeable with the CPU sim and runs can be checked spike-for-spike. Keep the two simulators behaviourally identical.
- **`mushroom.py`** — the only place weights change: dopamine-gated depression-only KC→MBON rule with a floor and drift back to baseline. Learned gains are stored with `sides_sha` so gains learned under a different `mb_sides.json` are never applied. `mb_gains.npz` is stale and unread.
- **`mb_sides.py`** — assigns each MBON type to PAM or PPL1 by *input* synapse counts from the raw synapse table (the weight matrix can't answer this because dopamine synapses are dropped).
- **`fly_drums_sim.py`** — drives the connectome toward a drum performance: reads real leg/wing/haltere/neck motor neurons (6 motor groups = 6 voices), uses `reference_data/groove_reference.json` (Groove MIDI) as the rhythm-match reward, and trains by evaluating a population of mutated KC→MBON gain vectors from the same start state. dt is widened to 2.0 ms for speed. Writes `fly_drums_export.json`.
- **`fly_drums_live_server.py`** — FastAPI/uvicorn; imports constants and `build_target_grid`/`fitness` from `fly_drums_sim.py`.

## Rules from PLAN.md (guardrails)

1. **Never edit** `flysim.py`, `flysim_gpu.py`, `mushroom.py`, `mb_sides.py`; wrap them instead.
2. One plan step at a time; commit and `git tag step-XX` after each; update `STATUS.md`; log unspecified design choices in `DECISIONS.md`.
3. Every reported number must come from an actual run saved in `results/` with `seed`, `git_sha`, `timestamp`. Use fixed seeds. Never hand-write numbers.
4. If a pass check fails twice, stop and report at a checkpoint; never loosen a pass check.
5. **Never alter the fly's hits** to sound better (quantizing to the grid and choosing samples is allowed; adding/removing/moving notes is not).
6. Scientific honesty is non-negotiable: keep the "real vs. chosen" distinction (connectome/dynamics/learning rule are real; the drumming goal, drive onto vnc_intrinsic, and reward are chosen) explicit in docs and code comments.
