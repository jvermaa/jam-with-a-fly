# Notice

The code in this repository is MIT licensed — see `LICENSE`. That licence
covers the code and nothing else.

`flysim.py`, `flysim_gpu.py`, `mushroom.py`, `mb_sides.py`, and
`build_graph.py` are carried over, unmodified, from
[fruitflydev/flycoinrh](https://github.com/fruitflydev/flycoinrh) — the real
leaky-integrate-and-fire simulation, the dopamine-gated learning rule, and
the connectome graph loader. `fly_drums_sim.py`, `fly_drums_live_server.py`,
and `real-brain.html` are new, written for this project.

**The connectome is not ours to license.** The male *Drosophila* CNS dataset
is © HHMI Janelia FlyEM, the Cambridge Connectomics Group and Google
Research, released under CC-BY 4.0, and it stays under CC-BY wherever it
goes. If you fork this, keep that attribution — it's the whole reason any of
this is real.

Groove reference data (part of the reward target in `fly_drums_sim.py`) is
quantized from six tracks in Google Magenta's
[Groove MIDI Dataset](https://magenta.withgoogle.com/datasets/groove)
(CC BY 4.0) and committed in `reference_data/`.

The rest of that reward target, when available, comes from
[MDBDrums](https://github.com/CarlSouthall/MDBDrums) (C. Southall, C. Wu,
A. Lerch, J. Hockman, *MDB Drums — An Annotated Subset of MedleyDB for
Automatic Drum Transcription*, ISMIR 2017), which is **CC BY-NC-SA 4.0** —
non-commercial and share-alike. Because of that, nothing derived from it is
committed here: `fly_drums_sim.py` reads it live from your own local
checkout (see README) and folds it into the reward target at training time,
never redistributing it.

The bundled `models/drum_kit.glb` drum kit model — check its own licence
before reusing it outside this project if it did not originate here.

Simulation approach after Shiu et al. 2024 (*Nature*) and the LIF model in
`flysim.py`. Learning rule after Hige et al. 2015 / Cohn et al. 2015,
implemented in `mushroom.py`. Not affiliated with Janelia, Google, Magenta,
or fruitflydev.

Additional credits: MaleCNS v1.0 (HHMI Janelia FlyEM, Cambridge Connectomics Group, Google Research, CC-BY 4.0); sykeriin/fly-drums (MIT); fruitflydev/flycoinrh; Bagel Fat One font (SIL OFL) used in assets/.

`calls/wwry.mid` (the Phase S target call) is built by `src/make_calls.py` from
our own text grid in `calls/call_spec_wwry.txt`. The pattern was verified
against the "We Will Rock You — Queen" drum score (Midi Drum Scores) and the
MIDI arrangement at onlinesequencer.net/1499428. Both are reference only and
are not redistributed: the arrangement is read from a local, gitignored
`reference/` folder by `src/wwry_reference.py`, and only the resulting hit
grid is recorded in `results/wwry_reference.json`.
