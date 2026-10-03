# Legacy export schema (`fly_drums_export.json`)

What upstream's `real-brain.html` reads in recorded mode, documented from the
original file (kept as `fly_drums_export.original.json`) and from the viewer's
`useRecorded()` code. Written before `src/export_legacy.py`, which targets it.

## Fields the viewer reads

| Field | Type | How the viewer uses it |
|---|---|---|
| `meta.neurons` | int | "neurons" readout and status line |
| `meta.edges` | int | "edges" readout |
| `meta.generations` | int | generation counter |
| `meta.bars` | int | number of bars; playback loops over them forever |
| `meta.wall_clock_s` | number | status line "…s to simulate" |
| `meta.channel_subclass` | {channel: text} | legend: "`<channel>` = `<text>` motor neurons" |
| `hits[]` | list | the performance (see below) |
| `bar_fitness[]` | list of numbers | last value shown as "fitness" (0 if empty) |
| `reward_history[]` | list of numbers | training curve; nothing is drawn with fewer than 2 values |
| `mushroom_stats.depressed`, `.synapses` | int | "depressed / synapses" readout |
| `neuron_cloud.xyz` | list of [x, y, z] | one point per neuron (soma location) |
| `neuron_cloud.superclass_names`, `.superclass_code` | list, list of int | base colour of each point; first 6 names go in the legend |
| `neuron_cloud.channel_names`, `.channel_code` | list, list of int | `channel_code[i] = k+1` puts point `i` in the bright overlay of `channel_names[k]`; 0 = none |

`target_grid`, `meta.dt_ms` and `meta.drive_hz` exist in the original file but
the viewer never reads them.

### `hits[]`

```json
{"bar": 0, "section": "intro", "step": 0, "ch": "cymbal", "vel": 0.7}
```

- `bar`: 0-based bar index, `step`: 0–15 (16th notes), `vel`: 0–1.
- `section`: free text shown in the "section" readout.
- `ch`: must be one of the six names hard-coded in the viewer:
  `kick`, `snare`, `hihat`, `tom`, `cymbal`, `chord`. Each has a fixed kit
  piece, colour and synthesised sound. `chord` is a synth pad, not a drum.
  `hihat` plays its open sound when `vel > 0.75`, closed otherwise.

## What the viewer does and does not show

- **Neurons do not flash on real spikes.** On each hit the viewer pulses the
  overlay of that channel's neurons. The export carries no spike data, and the
  viewer has nowhere to put any. What flashes is "the motor group that produced
  this hit", not "the neurons that fired".
- **Tempo is hard-coded to 100 BPM in recorded mode** (`scheduleBar(…, 100, …)`).
  Our material is 120 BPM, so the viewer plays the answer 1.2× slower than the
  `.mid`. It cannot be changed from the data.
- **Six fixed channels.** There is no open-hat piece.

## How our run maps onto it (`src/export_legacy.py`)

| Our voice | Motor group | Viewer channel | Note |
|---|---|---|---|
| Kick | `hl` | `kick` | |
| Snare | `ml` | `snare` | |
| Closed hat | `fl` | `hihat` | `vel` capped at 0.75 so it stays the closed sound |
| Open hat | `hm` | `hihat` | `vel` forced above 0.75 so it plays the open sound |
| Tom | `nm` | `tom` | |
| Crash | `wm` | `cymbal` | |
| — | — | `chord` | unused |

- Hits are the decoded answer exactly as in the `.mid`: same steps, same voices.
  Nothing is added, removed or moved. `vel` = MIDI velocity / 127, with the
  hi-hat adjustment above (a display/sound choice, not a change to the hit).
- `bar` = step // 16, `step` = step % 16, `section` = "answer".
- `neuron_cloud` is built the way upstream builds it (soma locations of traced
  neurons), but `channel_code` marks **our** motor groups (motor neurons only),
  under the viewer channel from the table.
- Untrained run, so `meta.generations` = 0, `reward_history` = [],
  `bar_fitness` = [], `mushroom_stats.depressed` = 0. `mushroom_stats.synapses`
  is the real KC→MBON synapse count.
- An extra top-level `jam` object carries provenance (`run_id`, `seed`,
  `git_sha`, `timestamp`, `bpm`, `f1`, `baseline_f1`, `lag_steps`). The viewer
  ignores it.
