# STATUS

Updated at the end of every step (PLAN.md guardrail 8).

## Done
- **Step 0.1 — environment** (tag `step-0.1`). `.venv` on Python 3.12.8; `requirements.txt` extended with `mido`, `pretty_midi`, `matplotlib`, `pytest`. Pass check `import mido, pretty_midi, pandas, pyarrow` succeeded.
- **Step 0.2 — scaffolding** (tag `step-0.2`). Folder structure from PLAN.md section 2, `CLAUDE.md` guardrails, `STATUS.md`, `DECISIONS.md`, `src/constants.py`, `config/voices.json`.

- **Step 0.3 — upstream demo untouched** (tag `step-0.3`). 🧑 confirmed on `http://localhost:8000/real-brain.html`: neuron cloud renders and colours, the fly plays the kit, audio plays. No upstream file changed.

- **Step 0.4 — connectome + graph** (tag `step-0.4`). Three MaleCNS v1.0 files downloaded to `data/` (MD5s match the server's). `build_graph.py` run unchanged via `src/build_graph_report.py`; pass check met, numbers in `results/build_graph.json`.

**Phase 0 complete.**

- **Step 1.1 — hearing + motor groups** (tag `step-1.1`). 🧑 confirmed the voice table and three choices: all JO subtypes are used (wording: "antennal sound and wind sensors", not "hearing neurons"); JO neurons with no outgoing connection are dropped before balancing; motor groups are motor neurons only (differs from upstream, which keeps descending neurons). Numbers in `results/groups_summary.json`, evidence in `results/reach_check.json`.

- **Step 1.2 — first calls** (tag `step-1.2`). 🧑 exported `calls/call_01.mid` … `call_05.mid` from Ableton; all five pass (tempo 120, length 4.0 s, allowed notes only, all 6 voices used across the set). Report in `results/calls_check.json`.

- **Step 1.3 — encoder.** `src/encode.py` + `config/encoder.json`; tests in `tests/test_encode.py` (written by a subagent) green; raster `results/call_01_input.png` shows a burst at every hit. Tagging stopped at 🧑's request.

- **Step 1.4 — run the fly.** `src/run_fly.py` plays a call into the JO groups through the unmodified simulator and runs a silent baseline with identical random numbers. Pass check met for `call_01`, seed 0: motor spike counts differ from silent (`results/run_meta.json`). Spike recordings (`results/*.npz`) are gitignored (about 30 MB each).

## In progress
- **Step 1.5 — decoder + score.**

## Next
- Step 1.6 best-of-10 batch → 🧑 checkpoint to pick a run.

## Blockers
- None. (Step 0.4 has not been started; it waits on the Step 0.3 reply.)

## Post links
- Post #1: —
- Post #2: —
- Post #3: —
