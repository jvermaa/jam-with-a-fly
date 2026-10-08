# PLAN.md — Fly Drum Jam (v1)

A simulated fruit fly (MaleCNS v1.0 connectome) hears a 2-bar drum beat through its hearing neurons and drums an answer back (call-and-response, echo mode). Built as a fork of `sykeriin/fly-drums`.

**Goal of v1:** a 3-post video series. Visual quality is priority #1; scientific honesty is non-negotiable.

---

## 0. How we work (READ FIRST, Claude)

This is a **50/50 partnership** between Claude Code (🤖) and Jatin (🧑).

### Roles
- 🤖 **Claude Code:** writes code, runs scripts, produces outputs, checks pass criteria, reports real numbers.
- 🧑 **Jatin:** anything in Ableton, anything visual/taste-based, anything needing hardware access, all approvals at checkpoints, all posting.

### Human checkpoint protocol
When a step is marked `🧑 CHECKPOINT`, Claude must **stop and print exactly this block**, then wait:

```
════════ 🧑 HUMAN CHECKPOINT [step-XX] ════════
WHAT I DID:    <1–3 bullets>
REAL RESULTS:  <numbers copied from results/*.json, file paths>
YOUR TASK:     <exact actions for Jatin, numbered, with file paths>
I NEED BACK:   <exactly what to reply with, e.g. "approve" / a file / a choice>
═══════════════════════════════════════════════
```

- Never continue past a checkpoint without Jatin's reply.
- Never guess a human decision; ask.
- Keep questions short (<100 chars each).

### Guardrails (copy into CLAUDE.md in Step 0.2)
1. **Never edit** `flysim.py`, `flysim_gpu.py`, `mushroom.py`, `mb_sides.py`. Wrap them; don't modify them.
2. One step at a time. Do not start step N+1 until step N's ✅ pass check is met and committed.
3. After each step: `git commit` + `git tag step-XX`.
4. Every number you report must come from an actual run, saved in `results/`. Never write numbers by hand.
5. Fixed seeds everywhere. Every results file includes `seed`, `git_sha`, `timestamp`.
6. If a pass check fails twice: **stop and report at a checkpoint**. Never loosen a pass check to make it pass.
7. **Never alter the fly's hits** to sound better. Allowed: quantizing to the grid, choosing drum samples. Not allowed: adding/removing/moving notes.
8. Update `STATUS.md` at the end of every step (what's done, what's next, blockers).
9. Log any design choice not specified here in `DECISIONS.md` and flag it at the next checkpoint.

---

## 1. Locked decisions

| Area | Decision |
|---|---|
| Base | Fork `sykeriin/fly-drums` (MIT). Credit it + upstream `fruitflydev/flycoinrh` |
| Connectome | MaleCNS v1.0 (CC-BY), 3 flat-connectome feather files |
| Input | 2-bar beats programmed in Ableton, exported as `.mid` |
| Tempo | Fixed 120 BPM, 4/4 |
| Mode | Echo via play-along: fly drums while hearing the call; its output is replayed as the answer |
| Voices | 6 (matches repo's 6 motor groups) |
| Encoding | Each drum voice → its own hearing-neuron (Johnston's organ) group |
| Viewer | Extend their `real-brain.html` (three.js) |
| Jev/CLM | Not in v1 |
| Training | Last phase. Benchmark Mac CPU vs GTX 1650, use faster |
| Video | Include early fails |
| Deadline | v1 series done within 7 days |

### Timing constants (use these everywhere — `src/constants.py`)
- `BPM = 120`
- `BAR_SEC = 2.0`, `CALL_BARS = 2`, `CALL_SEC = 4.0`
- `STEPS_PER_BAR = 16` (16th notes), `STEP_SEC = 0.125`, `CALL_STEPS = 32`
- `SIM_TAIL_SEC = 0.5` (extra sim time after call for late responses)
- `DT_MS = 2.0` (repo's speed tradeoff; validated default is 0.2 — note in credits)
- `HIT_TOLERANCE_STEPS = 1`
- `MAX_LAG_STEPS = 2`

### Voice table (default — 🧑 confirms in Step 1.1)

| idx | Drum voice | GM MIDI note | Motor group (output) | Hearing group (input) |
|---|---|---|---|---|
| 0 | Kick | 36 | `hl` hindleg | JO group 0 |
| 1 | Snare | 38 | `ml` midleg | JO group 1 |
| 2 | Closed hat | 42 | `fl` foreleg | JO group 2 |
| 3 | Open hat | 46 | `hm` haltere | JO group 3 |
| 4 | Tom | 45 | `nm` neck | JO group 4 |
| 5 | Crash | 49 | `wm` wing | JO group 5 |

Echo requires input voice *i* ↔ output voice *i*. This table is the single source of truth: `config/voices.json`.

---

## 2. File structure

```
fly-jam/                          # fork of sykeriin/fly-drums (name: 🧑 decides)
├── PLAN.md                       # this file
├── CLAUDE.md                     # guardrails (Step 0.2)
├── STATUS.md                     # updated every step
├── DECISIONS.md                  # every unplanned choice, flagged to 🧑
├── NOTICE.md                     # keep theirs + add our credits
├── README.md                     # rewritten in Phase 4
│
├── flysim.py  flysim_gpu.py      # THEIRS — DO NOT EDIT
├── mushroom.py  mb_sides.py      # THEIRS — DO NOT EDIT
├── build_graph.py                # THEIRS — run, don't edit
├── fly_drums_sim.py              # THEIRS — copied to src/train.py in Phase 3
├── real-brain.html               # THEIRS — copied to viewer/ in Phase 2
├── fly_drums_export.json         # THEIRS — back up before touching
│
├── config/
│   ├── voices.json               # voice table above
│   └── encoder.json              # burst rate/duration defaults
├── data/                         # raw feather files (gitignored)
├── build/
│   ├── graph.npz                 # from build_graph.py (gitignored)
│   ├── jo_groups.json            # 6 hearing-neuron groups
│   └── motor_groups.json         # 6 motor groups (bodyIds)
├── calls/                        # 🧑 Ableton exports: call_01.mid ...
│   └── heldout/                  # 🧑 3 calls never used in training
├── answers/                      # fly output .mid files
├── renders/                      # 🧑 WAV renders from Ableton
├── results/                      # all metrics JSON + plots
├── checkpoints/                  # training checkpoints (gitignored if large)
├── exports/                      # viewer timeline JSONs (schema v1)
├── src/
│   ├── constants.py
│   ├── groups.py                 # find JO + motor groups
│   ├── encode.py                 # MIDI → input spike trains
│   ├── run_fly.py                # stimulate + simulate + record
│   ├── decode.py                 # motor spikes → hits → .mid
│   ├── score.py                  # echo F1 + random baseline
│   ├── export_legacy.py          # → their viewer's JSON format (today)
│   ├── export_v1.py              # → our schema v1 (Phase 2)
│   ├── benchmark.py
│   └── train.py                  # Phase 3
├── viewer/
│   ├── index.html                # extended viewer (Phase 2)
│   └── assets/
├── docs/
│   ├── legacy_export_schema.md   # documented from their JSON
│   └── export_schema_v1.md       # ours, locked in Phase 2
└── tests/
    ├── test_encode.py
    ├── test_decode.py
    └── test_score.py
```

---

## 3. Phases (execution order)

**Order:** Phase 0 → 1 (post today) → 2 (viewer, post day 3) → 3 (training) → 4 (payoff post, day 7).

---

## PHASE 0 — Setup (today, ~1–2h)

### Step 0.1 — Fork & environment
- 🧑 Fork `sykeriin/fly-drums` on GitHub. Pick repo name. Clone locally on the Mac.
- 🤖 Create venv (Python 3.11 or 3.12), `pip install -r requirements.txt`, add `mido`, `pretty_midi`, `pandas`, `pyarrow`, `matplotlib`, `pytest`.
- 🤖 Add `.gitignore` entries: `data/`, `build/graph.npz`, `checkpoints/`, `.venv/`.
- **Outputs:** working venv, `requirements.txt` updated.
- ✅ **Pass:** `python -c "import mido, pretty_midi, pandas, pyarrow"` succeeds.
- Note: repo README uses `py` (Windows). On Mac/Linux use `python3`.

### Step 0.2 — Scaffolding
- 🤖 Create the folder structure above, `CLAUDE.md` (guardrails from section 0), `STATUS.md`, `DECISIONS.md`, `src/constants.py`, `config/voices.json`.
- ✅ **Pass:** `tree -L 2` matches section 2 (minus files not yet generated).

### Step 0.3 — Run their demo untouched
- 🤖 `python3 -m http.server 8000`
- 🧑 **CHECKPOINT:** open `http://localhost:8000/real-brain.html`.
- ✅ **Pass (🧑 confirms):** neuron cloud renders, fly plays kit, audio plays.

### Step 0.4 — Download connectome + build graph
- 🤖 Download into `data/` (no login):
  ```
  BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
  body-annotations-male-cns-v1.0-minconf-0.5.feather      (~14 MB)
  body-neurotransmitters-male-cns-v1.0.feather            (~42 MB)
  connectome-weights-male-cns-v1.0-minconf-0.5.feather    (~1.1 GB)
  ```
- 🤖 Run `python3 build_graph.py`.
- **Outputs:** `build/graph.npz`, `results/build_graph.json` (neuron count, synapse count, runtime).
- ✅ **Pass:** reports **165,122 neurons** (the repo's post-filter count). Any other number → stop, checkpoint.

---

## PHASE 1 — Untrained demo (today, ~3–5h) → 📱 POST #1

### Step 1.1 — Find hearing + motor groups
- 🤖 Load `body-annotations`. **First print all column names and 20 sample rows** to `results/annotation_columns.txt`. Do not assume column names.
- 🤖 Find Johnston's organ (JO) neurons: filter type/class columns for `JO`. Print every matching type name + count.
- 🤖 Split into 6 groups:
  - If ≥6 distinct JO subtypes: assign subtypes to groups to balance counts.
  - Else: even split sorted by bodyId with seed 0.
  - Include both left and right antennae in every group.
- 🤖 Load the repo's 6 motor groups (`fl ml hl wm hm nm`) the same way the repo does; save bodyIds.
- **Outputs:** `build/jo_groups.json`, `build/motor_groups.json`, `results/groups_summary.json` (counts per group, type names used).
- ✅ **Pass:** 6 non-empty JO groups, total JO neurons in the hundreds. **If total < 50 → wrong filter, stop.** 6 non-empty motor groups.
- 🧑 **CHECKPOINT:** show group counts + JO type names. Ask 🧑 to confirm/edit the voice table (section 1).

### Step 1.2 — 🧑 Make the first call
- 🧑 In Ableton: 120 BPM, 4/4, 2 bars, using only the 6 GM notes in the voice table. Export `calls/call_01.mid`.
- 🧑 Make 4 more variations (`call_02`–`call_05`) — different grooves, all 6 voices used at least once across the set. (Needed for best-of selection in Step 1.6.)
- 🤖 Validate every call file.
- ✅ **Pass:** each file: tempo 120, length exactly 4.0 s (±1 tick), only allowed notes. Print per-file report to `results/calls_check.json`.

### Step 1.3 — Encoder (MIDI → ears)
- 🤖 `src/encode.py`: for each hit at time *t* on voice *i*, inject a Poisson spike burst into JO group *i*:
  - duration 30 ms, rate = 150 Hz × (velocity / 127). Defaults in `config/encoder.json` (tunable, log changes in DECISIONS.md).
- 🤖 `tests/test_encode.py`: burst onsets land within 1 ms of MIDI hit times; correct group targeted.
- **Outputs:** `results/call_01_input.png` (raster: 6 rows = JO groups, x = time, with MIDI hits overlaid).
- ✅ **Pass:** tests green; raster bursts visibly align with every hit.

### Step 1.4 — Run the fly (play-along)
- 🤖 `src/run_fly.py`: simulate `CALL_SEC + SIM_TAIL_SEC` with the encoder input, keeping the repo's tonic drive onto the walking CPG (needed to wake the circuit — note in DECISIONS.md). Record spikes for JO + motor groups (and all-neuron spike times for the viewer).
- 🤖 Also run a **silent baseline** (same tonic drive, no call input) with the same seed.
- **Outputs:** `results/run_call_01_seed0.npz`, `results/silent_seed0.npz`, `results/run_meta.json` (runtime, machine, seed).
- ✅ **Pass:** motor spike counts during the call differ from the silent baseline (print both). If identical → input isn't reaching motor groups, stop.

### Step 1.5 — Decoder (motor spikes → drum hits) + score
- 🤖 `src/decode.py`:
  - Bin motor spikes into 32 steps of 125 ms.
  - Threshold per group = silent-baseline mean + 2·std of that group's step counts (from silent run, NOT tuned to the call).
  - Lag search: shift output by 0–`MAX_LAG_STEPS`, pick best fit, log lag.
  - Velocity = scaled spike count above threshold (clip 40–127).
  - Write `answers/untrained_call_01_seed0.mid` (120 BPM, 2 bars, voice table notes).
- 🤖 `src/score.py`: echo F1 per voice + overall, ±1 step tolerance. Random baseline: random hits at the answer's per-voice density, mean over 100 runs (seeded).
- 🤖 Tests for decode + score (synthetic perfect echo → F1 = 1.0; empty answer → F1 = 0).
- **Outputs:** `.mid` answer, `results/score_untrained_call_01_seed0.json`.
- ✅ **Pass:** `.mid` has >0 and <32×6 hits (not silent, not saturated); tests green; F1 and baseline both reported.

### Step 1.6 — Best-of-N batch
- 🤖 Run calls 01–05 × seeds 0–1 (= 10 runs). Score all.
- **Outputs:** `results/batch_untrained.json` ranked by: number of voices used, F1, hit density 10–40%.
- 🧑 **CHECKPOINT:** Claude lists top 3 runs with numbers + `.mid` paths. 🧑 drags them into Ableton, listens, picks one.
- ✅ **Pass:** 🧑 replies with chosen run ID.

### Step 1.7 — Export to THEIR viewer (no viewer code changes)
- 🤖 Back up `fly_drums_export.json` → `fly_drums_export.original.json`.
- 🤖 Read their JSON + how `real-brain.html` consumes it. Write `docs/legacy_export_schema.md` **before** writing any converter.
- 🤖 `src/export_legacy.py`: convert chosen run into that schema → `fly_drums_export.json`.
- ✅ **Pass:** viewer loads with no console errors; neurons flash during run; kit hits match the chosen `.mid` (Claude prints hit list for 🧑 to spot-check).
- 🛑 **Fallback (if >1h fighting schema):** `src/render_fallback.py` — matplotlib animation of neurons at real positions flashing on spikes → `renders/fallback.mp4`. Checkpoint before switching.

### Step 1.8 — 🧑 Record POST #1
- 🧑 Visual polish in viewer (only CSS/camera — no logic): dark bg, slow orbit, larger neuron glow. 🤖 can help with these edits on request.
- 🧑 Clip A: call playing in Ableton. Clip B: screen-record viewer playing the fly's answer. Route the answer `.mid` through a good Ableton kit for audio.
- 🧑 Caption angle: "I played a beat into a simulated fruit fly's brain. It answered (badly). Day 1." Credit Janelia FlyEM, Cambridge, Google Research, fly-drums. Say "best of 10 runs." Say "simulation on the real wiring diagram."
- ✅ **Pass:** posted. 🧑 drops the link in STATUS.md.

---

## PHASE 2 — Viewer upgrade (days 2–3) → 📱 POST #2

### Step 2.1 — Lock export schema v1
- 🤖 Draft `docs/export_schema_v1.md` covering everything Phase 3 will produce, so training output drops in with zero viewer changes:
  ```json
  {
    "schema_version": "1.0",
    "meta": { "seed": 0, "git_sha": "", "timestamp": "", "bpm": 120,
              "call_id": "call_01", "generation": null, "is_heldout": false,
              "score": { "f1": 0.0, "baseline_f1": 0.0 }, "lag_steps": 0 },
    "timeline": { "call_start": 0.0, "call_end": 4.0,
                  "answer_start": 4.0, "answer_end": 8.0 },
    "call_hits":   [ { "t": 0.0, "voice": 0, "vel": 100 } ],
    "answer_hits": [ { "t": 4.0, "voice": 0, "vel": 90 } ],
    "neurons": { "positions": "ref to positions file", "group_of": {} },
    "spikes": { "format": "binned_10ms", "jo": [], "motor": [], "all_sampled": [] },
    "audio": { "wav": "renders/jam_call_01.wav", "offset_sec": 0.0 }
  }
  ```
  - Answer = play-along output shifted by `CALL_SEC` (that's the replay).
  - `all_sampled`: downsample spikes if file > 20 MB.
- 🧑 **CHECKPOINT:** approve schema.
- ✅ **Pass:** 🧑 approves; `src/export_v1.py` writes a valid file for the chosen run; JSON-schema validation passes; timestamps match MIDI within 1 ms.

### Step 2.2 — Extend viewer
- 🤖 Copy `real-brain.html` → `viewer/index.html` (keep original untouched). Add:
  - Call/answer timeline bar with "YOU" / "FLY" labels and playhead.
  - JO neurons glow (color A) during the call; motor neurons glow (color B) during the answer.
  - Kit plays only answer hits; call hits shown on timeline.
  - Audio from `audio.wav` (not browser synth), timeline driven by audio clock.
- ✅ **Pass:** loads export v1 with no console errors; play/pause/seek work.

### Step 2.3 — 🧑 Render synced audio
- 🧑 In Ableton: place call MIDI at bar 1, answer `.mid` at bar 3, render one WAV → `renders/jam_<call_id>.wav`. Start exactly at 0.0 s.
- 🤖 Verify WAV duration ≈ 8.0 s (+ tail) and detect first transient ≈ first call hit.
- ✅ **Pass:** offset error < 10 ms.

### Step 2.4 — Sync + polish
- 🧑 **CHECKPOINT:** record 3 full plays. Check that the last hit's flash lines up with its sound by eye/ear.
- 🧑 Taste pass: colors, camera path, fonts, captions. 🤖 implements.
- ✅ **Pass:** 🧑 confirms no visible drift; 🧑 approves look.

### Step 2.5 — 🧑 POST #2
- "Now you can watch it hear me." Show JO glow on your beat → motor glow on its answer.
- ✅ **Pass:** posted, link in STATUS.md.

---

## PHASE 3 — Training (days 4–6)

### Step 3.1 — Benchmark
- 🧑 Boot the Linux laptop (GTX 1650), clone repo, install CUDA torch. 🤖 provides exact commands.
- 🤖 `src/benchmark.py`: simulate 1 call (4.5 s sim) on Mac CPU and on 1650.
- **Outputs:** `results/benchmark.json`: sec per call, peak RAM, peak VRAM, machine.
- ✅ **Pass:** both numbers reported. 1650 eligible only if peak VRAM < 3.8 GB.
- 🧑 **CHECKPOINT:** pick training machine.

### Step 3.2 — 🧑 Training dataset
- 🧑 Program 10 calls total (reuse 01–05 + 5 new). Put 3 in `calls/heldout/` — 🧑 chooses which, Claude never sees them during training.
- ✅ **Pass:** `results/calls_check.json` valid for all 10; 7 train / 3 held-out.

### Step 3.3 — Swap in echo reward
- 🤖 Copy `fly_drums_sim.py` → `src/train.py`. Replace the Groove-dataset reward with mean echo F1 over a minibatch of training calls. Keep their population/mutation loop + dopamine-gated depression rule unchanged.
- 🤖 Checkpoint every generation to `checkpoints/gen_XXX`. Log to `results/train_log.json` (reward per candidate, best per gen, time per gen).
- ✅ **Pass:** 1-generation dry run completes; log written; estimated full-run time printed.
- 🧑 **CHECKPOINT:** approve generation count based on time estimate (target: fits overnight).

### Step 3.4 — Train
- 🤖 Launch full run (resumable).
- **Outputs:** `results/train_curve.png`, checkpoints, final log.
- ✅ **Pass:** best reward trends upward (last 5 gens mean > first 5 gens mean). Keep gen 0 and a mid gen — that's "fail" footage.

### Step 3.5 — Held-out test (honesty gate)
- 🤖 Score trained fly on the 3 held-out calls (seeds 0–2).
- **Outputs:** `results/heldout_score.json`: F1 untrained vs trained vs random baseline.
- ✅ **Pass:** trained held-out F1 ≥ 2× random baseline.
- ❌ If it's only good on training calls → it memorized. **Do not claim "it learned to listen."** Checkpoint with 🧑 to decide: more data, more gens, or reframe the post.

### Step 3.6 — Export training story
- 🤖 Export v1 JSONs for: gen 0, mid gen, final gen — all on the same held-out call.
- ✅ **Pass:** all load in viewer with zero viewer code changes (proves schema lock).

---

## PHASE 4 — Payoff post (day 7) → 📱 POST #3

### Step 4.1 — 🧑 Render final video (<60 s)
- Hook (brain + "I trained a fly to jam with me") → gen 0 fail → training curve → final gen answering a **held-out** beat.
- 🧑 renders WAVs for each in Ableton (same as Step 2.3).

### Step 4.2 — 🤖 README + credits
- Rewrite README: what's real vs chosen (mirror the original repo's honesty section), results table from `results/`, how to run.
- NOTICE.md: MaleCNS (Janelia FlyEM, Cambridge Connectomics Group, Google Research, CC-BY), fly-drums, flycoinrh, Shiu et al. 2024 sim approach, Hige/Cohn learning rule, timestep tradeoff (2.0 ms vs validated 0.2 ms).
- ✅ **Pass:** every claim in the caption maps to a file in `results/`. 🤖 prints that mapping for 🧑.

### Step 4.3 — 🧑 Post
- ✅ **Pass:** posted, link in STATUS.md.

---

## Stretch (only after Post #3)
- 4-bar calls.
- Neurotransmitter sign check vs `neuropunks/malecns-nt-audit`.
- v2: accompaniment mode with Jev/CLM as conductor.
- Orchestra: multiple flies, different stems/readouts, flies hearing each other.

---

## Stop conditions (Claude must checkpoint immediately)
- Neuron count ≠ 165,122 after build.
- JO neurons total < 50.
- Motor output identical with and without call input.
- Answer is silent or saturated across all 10 Phase 1 runs.
- Any pass check fails twice.
- Any temptation to edit protected files or hand-edit hits.
---

## Phase S addendum — WWRY sprint (added 2026-10-08, 🧑's instructions)

A one-song sprint run outside the step order above. Gates and results are tracked in `STATUS.md`.

### Seed policy
| Use | Seeds |
|---|---|
| Training (both Gate 3 arms) | 100, 101 |
| Gate screening (Gate 1a onward: saturation, timestep, coupling, encoder grid) | 200–203 |
| Gate 4 final evaluation and demo | 300–303 |

Seeds 0–3 were used for screening in Gate 1a (tonic-drive grid, zero-drive check, KC→KC scaling) and are therefore not used for the final evaluation. Earlier Gate 0/1a results on seeds 0–3 stay as recorded.
