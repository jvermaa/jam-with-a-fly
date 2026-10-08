"""Run many simulations in parallel worker processes (Phase S).

Each worker holds its own FlyBrain (upstream flysim.py, imported and not edited)
and runs one job at a time through src/run_fly.py. A job returns compact arrays
(per-step motor counts, optional per-neuron bins), never the full spike list,
so results are cheap to send back.

One simulation uses one core (results/wwry_profile.json), so N workers on N
performance cores run N simulations in about the time of one.

Real vs. chosen: nothing here changes the model. Gains, when a job carries
them, are written into the KC->MBON weights by upstream's own
mushroom.MushroomBody.apply(), exactly as upstream's training does.
"""

import multiprocessing
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constants as c
from src.timing import Timing

UNUSED_STORE = ROOT / "checkpoints" / "unused_mb_store.npz"  # never written; keeps build/ out of it
PERFORMANCE_CORES = 8

_fb = None
_mb = None
_kc = None
_motor_neurons = None


def _init():
    global _fb, _kc
    os.environ["OMP_NUM_THREADS"] = "1"
    from flysim import FlyBrain  # upstream, unmodified
    from src import run_fly

    _fb = FlyBrain(p=run_fly.SimParams())
    _kc = _fb.where(type_re=r"^KC")


def _mushroom():
    global _mb
    if _mb is None:
        from mushroom import MushroomBody  # upstream, unmodified

        _mb = MushroomBody(_fb, store=UNUSED_STORE)
    return _mb


def _step_bins(spike_step, timing, n_bins):
    t = spike_step * (c.DT_MS / 1000.0)
    return np.minimum(np.floor(t / timing.step_sec + 1e-9).astype(np.int64), n_bins)


def run_job(job):
    """One simulation. job keys (all optional except seed):
      call            path of the call file, or None for the silent run
      seed, bpm, drive_hz, encoder_config, jo_bodies   passed to run_fly.run
      gain            KC->MBON gain vector to apply for this run (None = untrained, all 1)
      want            names of extra outputs: "motor_neurons" (per motor neuron x 16th-step
                      spike counts), "kc_windows" (which KCs fired in each 16th step)
      tag             anything; returned unchanged
    """
    from src import run_fly

    if _fb is None:
        _init()
    fb = _fb
    timing = Timing(float(job.get("bpm", c.BPM)))
    want = set(job.get("want", ()))

    gain = job.get("gain")
    if gain is not None or _mb is not None:
        mb = _mushroom()
        mb.gain = np.ones(len(mb.pos), dtype=np.float32) if gain is None else np.asarray(gain, dtype=np.float32)
        mb.apply()

    kwargs = {}
    if job.get("drive_hz") is not None:
        kwargs["drive_hz"] = float(job["drive_hz"])
    jo_bodies = job.get("jo_bodies")
    if jo_bodies is not None:
        jo_bodies = [np.asarray(g, dtype=np.int64) for g in jo_bodies]
    arrays, summary = run_fly.run(fb, job.get("call"), job["seed"], timing,
                                  encoder_config=job.get("encoder_config"), jo_bodies=jo_bodies, **kwargs)

    step, neuron = arrays["spike_step"], arrays["spike_neuron"]
    n_steps = summary["steps"]
    n_bins = int(np.floor(n_steps * c.DT_MS / 1000.0 / timing.step_sec + 1e-9))
    sim_sec = n_steps * c.DT_MS / 1000.0

    is_kc = np.zeros(fb.n, dtype=bool)
    is_kc[_kc] = True
    kc_sel = is_kc[neuron]
    kc_neuron, kc_bin = neuron[kc_sel], _step_bins(step[kc_sel], timing, n_bins)
    pairs = np.unique(kc_bin.astype(np.int64) * fb.n + kc_neuron)
    per_bin = np.bincount(pairs // fb.n, minlength=n_bins + 1)[:n_bins]
    kc_stats = {
        "n_kc": len(_kc),
        "kc_spikes": int(kc_sel.sum()),
        "kc_fired_at_least_once": len(np.unique(kc_neuron)),
        "kc_pct_fired_at_least_once": round(100.0 * len(np.unique(kc_neuron)) / len(_kc), 2),
        "kc_mean_rate_hz": round(float(kc_sel.sum()) / len(_kc) / sim_sec, 2),
        "kc_pct_active_per_16th_mean": round(100.0 * float(per_bin.mean()) / len(_kc), 2),
        "kc_pct_active_per_16th_max": round(100.0 * float(per_bin.max()) / len(_kc), 2),
    }

    voices, _, motor = run_fly.load_groups(fb, jo_bodies)
    motor_rate = {v["name"]: round(float(arrays["motor_step_counts"][i].sum()) / len(motor[i]) / sim_sec, 2)
                  for i, v in enumerate(voices)}

    out = {
        "tag": job.get("tag"),
        "summary": summary,
        "kc": kc_stats,
        "motor_rate_hz": motor_rate,
        "motor_step_counts": arrays["motor_step_counts"],
        "jo_step_counts": arrays["jo_step_counts"],
        "hit_t": arrays["hit_t"],
        "hit_voice": arrays["hit_voice"],
    }
    if "motor_neurons" in want:
        global _motor_neurons
        if _motor_neurons is None:
            _motor_neurons = np.unique(np.concatenate(motor))
        pos = np.full(fb.n, -1, dtype=np.int64)
        pos[_motor_neurons] = np.arange(len(_motor_neurons))
        sel = pos[neuron] >= 0
        b = _step_bins(step[sel], timing, n_bins)
        keep = b < n_bins
        mat = np.zeros((len(_motor_neurons), n_bins), dtype=np.int32)
        np.add.at(mat, (pos[neuron[sel]][keep], b[keep]), 1)
        out["motor_neuron_bins"] = mat
        out["motor_neuron_bodies"] = fb.bodies[_motor_neurons]
    if "kc_windows" in want:
        out["kc_windows"] = [kc_neuron[kc_bin == b].astype(np.int32) for b in range(n_bins)]
        out["kc_windows"] = [np.unique(w) for w in out["kc_windows"]]
    return out


def run_jobs(jobs, workers=PERFORMANCE_CORES):
    """Run jobs in parallel; results come back in job order."""
    workers = max(1, min(workers, len(jobs)))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, maxtasksperchild=None) as pool:
        return pool.map(run_job, jobs, chunksize=1)
