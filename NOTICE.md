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

Groove reference data (`fly_drums_export.json`'s reward target) is quantized
from six tracks in Google Magenta's
[Groove MIDI Dataset](https://magenta.withgoogle.com/datasets/groove)
(CC BY 4.0).

The bundled `models/drum_kit.glb` drum kit model — check its own licence
before reusing it outside this project if it did not originate here.

Simulation approach after Shiu et al. 2024 (*Nature*) and the LIF model in
`flysim.py`. Learning rule after Hige et al. 2015 / Cohn et al. 2015,
implemented in `mushroom.py`. Not affiliated with Janelia, Google, Magenta,
or fruitflydev.
