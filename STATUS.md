# STATUS

Updated at the end of every step (PLAN.md guardrail 8).

## Done
- **Step 0.1 — environment** (tag `step-0.1`). `.venv` on Python 3.12.8; `requirements.txt` extended with `mido`, `pretty_midi`, `matplotlib`, `pytest`. Pass check `import mido, pretty_midi, pandas, pyarrow` succeeded.
- **Step 0.2 — scaffolding** (tag `step-0.2`). Folder structure from PLAN.md section 2, `CLAUDE.md` guardrails, `STATUS.md`, `DECISIONS.md`, `src/constants.py`, `config/voices.json`.

- **Step 0.3 — upstream demo untouched** (tag `step-0.3`). 🧑 confirmed on `http://localhost:8000/real-brain.html`: neuron cloud renders and colours, the fly plays the kit, audio plays. No upstream file changed.

- **Step 0.4 — connectome + graph** (tag `step-0.4`). Three MaleCNS v1.0 files downloaded to `data/` (MD5s match the server's). `build_graph.py` run unchanged via `src/build_graph_report.py`; pass check met, numbers in `results/build_graph.json`.

**Phase 0 complete.**

- **Step 1.1 — hearing + motor groups** (tag `step-1.1`). 🧑 confirmed the voice table and three choices: all JO subtypes are used (wording: "antennal sound and wind sensors", not "hearing neurons"); JO neurons with no outgoing connection are dropped before balancing; motor groups are motor neurons only (differs from upstream, which keeps descending neurons). Numbers in `results/groups_summary.json`, evidence in `results/reach_check.json`.

## In progress
- **Step 1.2 — first calls.** Waiting on 🧑: export `calls/call_01.mid` … `calls/call_05.mid` from Ableton (120 BPM, 4/4, 2 bars, only notes 36 38 42 46 45 49, all 6 voices used at least once across the set).

## Next
- 🤖 validates every call → `results/calls_check.json`, then Step 1.3 (encoder; tests delegated to a subagent).

## Blockers
- None. (Step 0.4 has not been started; it waits on the Step 0.3 reply.)

## Post links
- Post #1: —
- Post #2: —
- Post #3: —
