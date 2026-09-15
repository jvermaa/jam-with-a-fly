# fly-drums

A fruit fly's real brain, playing a real drum kit.

Not a metaphor and not a toy neural net "inspired by" a brain: this loads the
actual FlyEM/Janelia **male *Drosophila* CNS connectome** — 165,122 traced
neurons, 10,228,000 measured, signed synapses — into a leaky
integrate-and-fire simulation, reads six real, annotated motor-neuron
populations (foreleg, midleg, hindleg, wing, haltere, neck) as drum voices,
and trains them with the same mechanism a fly actually learns odors with:
dopamine-gated depression at the Kenyon-cell → MBON synapse. It's rewarded by
how closely its output matches real human drumming — six electronic-kit
performances from Google Magenta's Groove MIDI Dataset, blended with 23 real
acoustic-kit performances from MDBDrums when you have a local checkout of it
(see "Real drumming data" below).

**[See it running](real-brain.html)** — a three.js page: every traced neuron
plotted at its real measured position on the left, a fly performing on a kit
on the right, synthesized in the browser with the Web Audio API.

## Two ways to hear it

| | requires | what it does |
|---|---|---|
| **Recorded** | nothing — just open `real-brain.html` over http | replays `fly_drums_export.json`, a run trained and rendered once offline |
| **Live** | an NVIDIA GPU, `py fly_drums_live_server.py` | trains and performs in real time, streamed to the page over a WebSocket the moment each bar is computed |

`real-brain.html` tries the live WebSocket first and falls back to the
recorded run automatically if nothing answers — there's also a manual
"use recorded / try live" toggle on the page.

## Running it

```bash
pip install -r requirements.txt

# the connectome itself — 1.1 GB, CC-BY, no account or key needed
# (see "The connectome" below for the exact URLs), into data/
py build_graph.py            # -> build/graph.npz, 165,122 neurons

# recorded: trains + renders a performance once, writes fly_drums_export.json
py fly_drums_sim.py
py -m http.server 8000       # then open http://localhost:8000/real-brain.html

# live (needs a CUDA GPU + torch): trains and performs continuously
py fly_drums_live_server.py  # open http://localhost:4670
```

`build/mb_sides.json` is committed (it's small and derived once from the
connectome), so you don't need to run `mb_sides.py` yourself unless you
want to regenerate it.

## The connectome

CC-BY, from a public bucket, no account and no key:

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
  body-annotations-male-cns-v1.0-minconf-0.5.feather      14 MB
  body-neurotransmitters-male-cns-v1.0.feather            42 MB
  connectome-weights-male-cns-v1.0-minconf-0.5.feather   1.1 GB
```

Put those three files in `data/`, then `py build_graph.py`.

## Real drumming data

The reward target in `fly_drums_sim.py` is quantized real drumming, not
invented rhythm:

- **Groove MIDI Dataset** (Google Magenta, CC BY 4.0) — six electronic-kit
  performances, `reference_data/groove_reference.json`, committed here since
  its licence permits that.
- **[MDBDrums](https://github.com/CarlSouthall/MDBDrums)** (Southall et al.
  2017, CC BY-NC-SA 4.0) — 23 real acoustic-kit performances. Its licence is
  share-alike and non-commercial, so **nothing derived from it is committed
  to this repo**. Clone it yourself as a sibling folder —
  `git clone https://github.com/CarlSouthall/MDBDrums ../MDBDrums` from this
  repo's parent directory — and `fly_drums_sim.py` finds it automatically
  (or point `MDB_DRUMS_DIR` at wherever you put it). Training works fine
  without it; the reward is just Groove MIDI alone in that case.

## What's real, what's chosen — stated plainly

- **Real, unmodified:** the connectome, the LIF dynamics (`flysim.py`,
  `flysim_gpu.py`), the six motor populations (real annotated subclasses
  `fl` / `ml` / `hl` / `wm` / `hm` / `nm`), and the learning rule —
  dopamine-gated depression at the Kenyon-cell → MBON synapse, the one place
  this project lets a weight move (`mushroom.py`).
- **Chosen:** nothing in a real fly asks it to drum. A tonic drive onto the
  ventral-nerve-cord's own walking central-pattern-generator population is
  what wakes the circuit up at all. The reward — how closely six real motor
  readouts match six real human grooves — is invented. So is the population/
  selection/mutation training loop in `fly_drums_sim.py`: each generation
  evaluates several candidate gain vectors from the same starting brain
  state and keeps whichever actually scores best, then applies the real
  depression rule to the winner.
- **A real tradeoff:** the timestep is widened from the validated 0.2ms to
  2.0ms so a bar simulates in about a minute on CPU instead of about eight,
  or in real time on a GPU. That's not the connectome's validated default.

## Files

| file | what it is |
|---|---|
| `flysim.py`, `flysim_gpu.py` | the LIF simulator (CPU / CUDA), carried over from flycoinrh |
| `mushroom.py`, `mb_sides.py` | the real dopamine-gated learning circuit |
| `build_graph.py` | turns the raw connectome download into `build/graph.npz` |
| `fly_drums_sim.py` | offline trainer + exporter — writes `fly_drums_export.json` |
| `fly_drums_live_server.py` | FastAPI/WebSocket server for the live GPU path |
| `real-brain.html` | the viewer — neuron point cloud, fly, kit, audio |
| `models/drum_kit.glb` | the 3D drum kit model |

See `NOTICE.md` for licensing and attribution — the connectome and the
Groove MIDI reference data are not ours and stay under their own licences.

## Credits

Connectome: HHMI Janelia FlyEM, the Cambridge Connectomics Group, and Google
Research, CC-BY. Simulation approach after Shiu et al. 2024 (*Nature*).
Learning rule after Hige et al. 2015 / Cohn et al. 2015. Groove reference:
Google Magenta's Groove MIDI Dataset (CC BY 4.0). `flysim.py`,
`flysim_gpu.py`, `mushroom.py`, `mb_sides.py`, and `build_graph.py` are from
[fruitflydev/flycoinrh](https://github.com/fruitflydev/flycoinrh). Not
affiliated with Janelia, Google, Magenta, or fruitflydev.
