# Export schema v1 (DRAFT, awaiting approval)

One JSON file per run, read by `viewer/index.html`. Written by
`src/export_v1.py`. Designed so that Phase 3 (trained runs, generations,
held-out calls) produces the same file shape and the viewer needs no change.

Status: **draft**. Once approved it is locked: fields may be added later, but
none renamed, removed or re-meant.

## What one file represents

One run of the fly on one call:

- The simulation covers 0 to `CALL_SEC + SIM_TAIL_SEC` (4.5 s). During it the
  fly hears the call and drums **at the same time** (play-along).
- The video shows this as call (0–4 s) then answer (4–8 s). The answer is the
  drumming recorded during the call, **replayed** 4 s later. Nothing is
  simulated between 4.5 s and 8 s.

Every time in the file is in **presentation seconds** (0–8), so the viewer
never has to know about the replay. The file states the replay explicitly
(`timeline.answer_is_replay`), and any caption must not say the fly listened
first and answered afterwards.

## Top level

```json
{
  "schema_version": "1.0",
  "meta": { },
  "voices": [ ],
  "timeline": { },
  "call_hits": [ ],
  "answer_hits": [ ],
  "neurons": { },
  "spikes": { },
  "audio": { }
}
```

## `meta`

| Field | Type | Meaning |
|---|---|---|
| `run_id` | string | e.g. `untrained_call_07_seed0`, `gen012_heldout_02_seed1` |
| `label` | string | `untrained` or `trained` |
| `seed` | int | simulation and encoder seed |
| `git_sha` | string | commit that produced the run |
| `timestamp` | string | ISO 8601, UTC |
| `bpm` | number | 120 |
| `dt_ms` | number | simulation timestep (2.0; the validated default is 0.2) |
| `call_id` | string | e.g. `call_07` |
| `is_heldout` | bool | call was never used in training |
| `generation` | int or null | training generation; null when untrained |
| `checkpoint` | string or null | path of the gains used; null when untrained |
| `score` | object | `{ "f1", "baseline_f1", "per_voice_f1": [6] }` |
| `lag_steps` | int | global lag removed from the answer (0–2 sixteenths) |
| `selection` | string or null | how the run was picked, e.g. "chosen from 16 untrained runs" |
| `tonic_drive_hz` | number | artificial wake-up drive on `vnc_intrinsic` (chosen, from upstream) |
| `sources` | object | repo-relative paths: `score_json`, `answer_mid`, `call_mid` |

## `voices`

Six entries, index = voice. Makes the file self-describing.

```json
{ "idx": 0, "name": "Kick", "midi_note": 36,
  "motor_group": "hl", "motor_neurons": 130,
  "jo_group": 0, "jo_neurons": 93 }
```

## `timeline`

```json
{ "call_start": 0.0, "call_end": 4.0,
  "answer_start": 4.0, "answer_end": 8.0,
  "step_sec": 0.125, "answer_is_replay": true }
```

## `call_hits` and `answer_hits`

```json
{ "t": 0.0, "step": 0, "voice": 0, "vel": 110 }
```

- `t` in presentation seconds. Call hits: `step * 0.125`. Answer hits:
  `4.0 + step * 0.125`, after the lag has been removed.
- `step` 0–31, `voice` 0–5, `vel` MIDI velocity 1–127.
- `answer_hits` are the decoded hits exactly as in the answer `.mid`.

## `neurons`

```json
{ "positions": "exports/neurons_v1.json",
  "group_of": { "jo": [[...], ...6], "motor": [[...], ...6] } }
```

- `positions` points to one shared file (same for every run):
  `{ "bodyId": [...], "xyz": [[x, y, z], ...], "superclass": [...], "superclass_names": [...] }`.
  One entry per neuron that has a recorded position.
- `group_of.jo[i]` / `group_of.motor[i]`: indices into that file for voice `i`.

**Open question (needs a decision): the antennal (JO) neurons have no recorded
position.** Their cell bodies sit in the antenna, outside the imaged volume, so
the annotation file gives them no coordinates. See the checkpoint options.

## `spikes`

```json
{ "format": "binned_10ms", "bin_sec": 0.01, "n_bins": 800,
  "jo":    [[...800], ...6],
  "motor": [[...800], ...6],
  "all_sampled": { "neurons": [...], "sample_fraction": 0.05,
                   "active": [[...], ...800] } }
```

- `jo[i][b]`: spikes of voice `i`'s antennal group in presentation bin `b`.
  Non-zero only during the call (bins 0–399).
- `motor[i][b]`: spikes of voice `i`'s motor group, placed in the **answer**
  window (bins 400–799): the play-along motor activity shifted by 4 s minus
  the lag, so it lines up with `answer_hits`. Zero during the call window.
- `all_sampled`: a fixed random sample (seeded) of positioned neurons, for the
  background "brain is active" shimmer. `active[b]` lists which sampled neurons
  (indices into `neurons`) spiked in bin `b`. Call window = real simulation
  time; answer window = the same activity replayed. The sample fraction is
  lowered until the file is under 20 MB.

A run has about 19 million spikes, so every spike of every neuron cannot be
shipped; counts per group plus a sample is what the viewer gets.

## `audio`

```json
{ "wav": "renders/jam_call_07.wav", "offset_sec": 0.0 }
```

`wav` may be null until the render exists (Step 2.3). The viewer's clock is the
audio clock; `offset_sec` is added to audio time to get presentation time.

## Validation

`src/export_v1.py` validates every file it writes against a JSON Schema kept in
`docs/export_schema_v1.schema.json`, and checks that hit times match the
`.mid` files within 1 ms.

## Real vs. chosen in this file

- Real: neuron identities, positions, which neurons spiked and when (from the
  simulation on the real wiring).
- Chosen: the call, the encoding onto antennal groups, the tonic drive, reading
  motor groups as drums, thresholds and lag, the echo score, and presenting
  play-along as call-then-answer.
