# STATUS

Updated at the end of every step (PLAN.md guardrail 8).

## Done
- **Step 0.1 — environment** (tag `step-0.1`). `.venv` on Python 3.12.8; `requirements.txt` extended with `mido`, `pretty_midi`, `matplotlib`, `pytest`. Pass check `import mido, pretty_midi, pandas, pyarrow` succeeded.
- **Step 0.2 — scaffolding** (tag `step-0.2`). Folder structure from PLAN.md section 2, `CLAUDE.md` guardrails, `STATUS.md`, `DECISIONS.md`, `src/constants.py`, `config/voices.json`.

- **Step 0.3 — upstream demo untouched** (tag `step-0.3`). 🧑 confirmed on `http://localhost:8000/real-brain.html`: neuron cloud renders and colours, the fly plays the kit, audio plays. No upstream file changed.

- **Step 0.4 — connectome + graph** (tag `step-0.4`). Three MaleCNS v1.0 files downloaded to `data/` (MD5s match the server's). `build_graph.py` run unchanged via `src/build_graph_report.py`; pass check met, numbers in `results/build_graph.json`.

**Phase 0 complete.**

## In progress
- **Step 1.1 — hearing + motor groups.** Automated pass checks met (`results/groups_summary.json`, `build/jo_groups.json`, `build/motor_groups.json`, `results/annotation_columns.txt`). Waiting on 🧑 checkpoint: confirm/edit the voice table, and decide the two honesty questions (which JO neurons count as "hearing"; whether the foreleg group keeps its descending neurons).

## Next
- Step 1.2 — 🧑 exports `calls/call_01.mid` … `call_05.mid` from Ableton; 🤖 validates them.

## Blockers
- None. (Step 0.4 has not been started; it waits on the Step 0.3 reply.)

## Post links
- Post #1: —
- Post #2: —
- Post #3: —
