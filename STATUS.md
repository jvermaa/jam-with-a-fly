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

- **Step 1.5 — decoder + score.** `src/decode.py`, `src/score.py`; tests (subagent-written) green. `call_01` seed 0: answer written to `answers/untrained_call_01_seed0.mid`, score in `results/score_untrained_call_01_seed0.json` (F1 and random baseline both reported; answer is neither silent nor saturated).

- **Step 1.6 — best-of-10 batch.** Ranked in `results/batch_untrained.json`. 🧑 said to use any run; the top-ranked `untrained_call_01_seed0` was taken.

- **Step 1.7 — export to upstream's viewer.** `fly_drums_export.json` first held `untrained_call_01_seed0` (original kept as `fly_drums_export.original.json`; schema in `docs/legacy_export_schema.md`). 🧑 confirmed: no console errors from our data, viewer pattern matches the `.mid` in Ableton at 120 BPM. One approved viewer change: playback tempo read from `jam.bpm`.

- **Extra calls (🧑 request).** `calls/call_06`–`08.mid` are script-generated transcriptions of well-known beats (`src/make_calls.py`, `calls/call_spec_famous.txt`); 6 more untrained runs in `results/batch_untrained_famous.json`. 🧑 chose `untrained_call_07_seed0` (Be My Baby pattern); `fly_drums_export.json` now holds it. Any claim is "best of 16 runs".

- **Phase 1 code complete** on branch `phase-1-untrained-demo` (to be merged by 🧑). Step 1.8 (record + post) is deferred by 🧑, who is in no rush to post.

## Phase S — "We Will Rock You" sprint (branch `phase-s-wwry-sprint`)
One-song training sprint requested by 🧑, outside PLAN.md's step order. Five gates (0–4), stop and report at each. Stacked on the Phase 2 branch; step 2.1 stays parked.

- **Gate 0 — facts + setup: done, approved by 🧑** (with added Gate 1a, and a trained-readout arm B at Gate 3).
  - Per-call tempo: `src/timing.py` + `config/calls.json` (default 120, `wwry` 82). Calls 01–08 unchanged (regression test on `call_06` reproduces its committed score).
  - Target `calls/wwry.mid` built from `calls/call_spec_wwry.txt` (82 BPM, 2 bars = 5.854 s); passes `results/wwry_calls_check.json`.
  - Pattern equals bars 1–7 of the local reference arrangement with clap 39 → snare 38 (`results/wwry_reference.json`; reference file is gitignored). `NOTICE.md` updated.
  - One profiled run + what training can change: `results/wwry_profile.json`.
  - Tests (subagent-written): `tests/test_timing.py`, `tests/test_wwry_reference.py`; 79 passed.
- **Gate 1a — saturation check: FAILED, stopped for 🧑.** `results/wwry_saturation.json`.
  - APL is GABA and inhibitory in the graph (both cells, every outgoing edge negative); no sign fix needed. The graph is built from `consensus_nt`, the column the malecns-nt-audit README recommends; that README does not mention APL.
  - Tonic drive at 100/50/25/10 % of upstream's 45 Hz: every Kenyon cell fires at the model ceiling at all four levels, call or silent. Motor rates barely follow the drive. Coupling passes at no level.
  - New: `src/coupling.py` (metric), `src/sim_pool.py` (parallel runs), `src/saturation.py`; `run_fly` takes `drive_hz`.
- **Gate 1a step B — zero / 1 Hz drive: done.** `results/wwry_wake_check.json`. No drive and no sound: 0 spikes. The call alone ignites the same runaway state. Saturation is not KC-only (non-KC central brain, descending and motor neurons also have 12–17 % of cells above 200 Hz).
- **Gate 1a step A — KC→KC scaled by 0 / 0.25 / 0.5: FAILED, stopped for 🧑.** `results/wwry_kc_gate.json`. KCs stay 67–93 % active per sixteenth step at every scale and drive; the rest of the network is unchanged; coupling confidence intervals (seeds 0–3) all include zero. New: `src/kc_recurrence.py` (wrapper), `src/wake_check.py`, `src/kc_gate.py`.
- Gate 1b (encoder grid), Gates 2–4: not started; blocked on Gate 1a.

## In progress
- **Step 2.1 — lock export schema v1** on branch `phase-2-viewer-upgrade`. Draft in `docs/export_schema_v1.md`. Waiting on 🧑 checkpoint: approve the schema and decide how to place the antennal (JO) neurons, which have no recorded position.

## Next
- `src/export_v1.py` + JSON-schema validation, then Step 2.2 (extend the viewer in `viewer/index.html`).

## Deferred
- Step 1.8 — Post #1 (🧑). Clip A `calls/call_07.mid`; Clip B the viewer + `answers/untrained_call_07_seed0.mid`; "best of 16 runs".

## Blockers
- Phase S: Gate 1a pass check failed twice (tonic-drive grid, then KC→KC scaling). The whole network runs away once anything ignites it. Waiting on 🧑.

## Post links
- Post #1: —
- Post #2: —
- Post #3: —
