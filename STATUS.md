# STATUS

Updated at the end of every step (PLAN.md guardrail 8).

## Done
- **Step 0.1 — environment** (tag `step-0.1`). `.venv` on Python 3.12.8; `requirements.txt` extended with `mido`, `pretty_midi`, `matplotlib`, `pytest`. Pass check `import mido, pretty_midi, pandas, pyarrow` succeeded.
- **Step 0.2 — scaffolding** (tag `step-0.2`). Folder structure from PLAN.md section 2, `CLAUDE.md` guardrails, `STATUS.md`, `DECISIONS.md`, `src/constants.py`, `config/voices.json`.

- **Step 0.3 — upstream demo untouched** (tag `step-0.3`). 🧑 confirmed on `http://localhost:8000/real-brain.html`: neuron cloud renders and colours, the fly plays the kit, audio plays. No upstream file changed.

## In progress
- **Step 0.4 — connectome download + `build_graph.py`.** Pass check: 165,122 neurons.

## Next
- Phase 1, Step 1.1 — find hearing (JO) + motor groups.

## Blockers
- None. (Step 0.4 has not been started; it waits on the Step 0.3 reply.)

## Post links
- Post #1: —
- Post #2: —
- Post #3: —
