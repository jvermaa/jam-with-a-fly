# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A simulated male fruit fly (MaleCNS v1.0 connectome, ~165k neurons) hears a 2-bar drum beat through hearing neurons and drums an answer back. Forked from `sykeriin/fly-drums`. `PLAN.md` is the v1 roadmap (step-by-step, with human checkpoints); read it before starting planned work. `docs/ORIGINAL_README.md` is the upstream README.

## Commands

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # use .venv/bin/python below
# fetch the connectome (~1.1 GB, CC-BY, no login; links in PLAN.md) into data/
python build_graph.py            # data/*.feather -> build/graph.npz (signed sparse W)
python mb_sides.py               # -> build/mb_sides.json (MBON dopamine side table)
python fly_drums_sim.py          # recorded CPU run -> fly_drums_export.json (viewed by real-brain.html)
python fly_drums_live_server.py  # live GPU server on http://localhost:4670 (needs torch + CUDA; torch is commented out in requirements.txt)
```

CI (`.github/workflows/ci.yml`) runs ruff on `src/` and `tools/` only (the legacy sim files are excluded), checks that every local image path in `README.md` exists, and runs `pytest` only if a `tests/` dir exists. Tests live in `tests/` (run with `.venv/bin/python -m pytest`). `data/`, `build/graph.npz`, `checkpoints/` and `*.feather` are gitignored; never commit connectome data or large derived datasets.

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

Copied from PLAN.md section 0. PLAN.md is the authority if the two ever differ.

1. **Never edit** `flysim.py`, `flysim_gpu.py`, `mushroom.py`, `mb_sides.py`. Wrap them; don't modify them.
2. One step at a time. Do not start step N+1 until step N's pass check is met and committed.
3. After each step: `git commit` + `git tag step-XX`.
4. Every number you report must come from an actual run, saved in `results/`. Never write numbers by hand.
5. Fixed seeds everywhere. Every results file includes `seed`, `git_sha`, `timestamp`.
6. If a pass check fails twice: **stop and report at a checkpoint**. Never loosen a pass check to make it pass.
7. **Never alter the fly's hits** to sound better. Allowed: quantizing to the grid, choosing drum samples. Not allowed: adding/removing/moving notes.
8. Update `STATUS.md` at the end of every step (what's done, what's next, blockers).
9. Log any design choice not specified in PLAN.md in `DECISIONS.md` and flag it at the next checkpoint.

Scientific honesty is non-negotiable: keep the "real vs. chosen" distinction (connectome/dynamics/learning rule are real; the drumming goal, drive onto vnc_intrinsic, and reward are chosen) explicit in docs and code comments.

### Working preferences (from the human)

- **Always delegate test writing to a subagent.** Whoever is doing a plan step (the main session or any agent) must not write tests itself: spawn a subagent with the module path, the behaviour to cover and the step's pass check, and have it write and run the tests. The delegating agent only reads the pass/fail result. Reason: one task per agent; the agent doing the step should not spend its tokens on testing.
- **Work goes through a branch, not straight onto `main`.** Commit and tag on a concisely but descriptively named branch and push that; `main` only moves by merge.

- **Keep personal details out of the public repo.** `DECISIONS.md` is a local working log: it is gitignored and must never be committed or pushed. Nothing committed (code, docs, `results/`) may contain absolute paths, usernames, machine names, emails or real names; use repo-relative paths.

### Human checkpoint protocol

When a step is marked `🧑 CHECKPOINT`, stop, print exactly this block, then wait:

```
════════ 🧑 HUMAN CHECKPOINT [step-XX] ════════
WHAT I DID:    <1–3 bullets>
REAL RESULTS:  <numbers copied from results/*.json, file paths>
YOUR TASK:     <exact actions for the human, numbered, with file paths>
I NEED BACK:   <exactly what to reply with, e.g. "approve" / a file / a choice>
═══════════════════════════════════════════════
```

- Never continue past a checkpoint without the human's reply.
- Never guess a human decision; ask.
- Keep questions short (<100 chars each).

### Stop conditions (checkpoint immediately)

- Neuron count ≠ 165,122 after build.
- JO neurons total < 50.
- Motor output identical with and without call input.
- Answer is silent or saturated across all 10 Phase 1 runs.
- Any pass check fails twice.
- Any temptation to edit protected files or hand-edit hits.
