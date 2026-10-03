"""
Live version of fly_drums_sim.py: the real 165k-neuron connectome, stepped on
this machine's GPU (flysim_gpu.FlyBrainGPU) in actual real time, streamed to
the browser over a WebSocket bar by bar as it's computed.

Benchmarked on this GPU (RTX 3050, dt widened to 2.0ms as in fly_drums_sim.py):
one 150ms window of simulated time takes ~140ms of wall-clock time — i.e. this
runs faster than real time, unlike the CPU path (flysim.FlyBrain), which is
what fly_drums_sim.py's recorded version exists for.

Run: py fly_drums_live_server.py, then open http://localhost:4670
"""
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from flysim import Params
from flysim_gpu import FlyBrainGPU
from mushroom import MushroomBody
from fly_drums_sim import (
    CHANNEL_SUBCLASS, STEPS_PER_BAR, WINDOW_MS, DRIVE_HZ,
    build_target_grid, fitness, GROOVE_JSON,
)

ROOT = Path(__file__).parent
TRAIN_GENERATIONS = 6
SECTIONS = ["intro", "intro", "groove", "groove", "groove", "fill", "variation", "outro"]


class LiveParams(Params):
    dt = 2.0  # see fly_drums_sim.py docstring — the real-time/fidelity tradeoff


app = FastAPI()
state = {}
_conn_id = {"current": 0}  # only the most recently connected client drives the (single, shared) brain


def setup():
    print("loading real connectome onto the GPU ...")
    fb = FlyBrainGPU(p=LiveParams(), device="cuda")
    mb = MushroomBody(fb)
    populations = {ch: fb.where(subclass=sub) for ch, sub in CHANNEL_SUBCLASS.items()}
    kc = fb.where(type_re=r"^KC")
    vnc_int = fb.where(superclass="vnc_intrinsic")
    drive = {tuple(vnc_int.tolist()): DRIVE_HZ}
    target = build_target_grid()
    state.update(fb=fb, mb=mb, populations=populations, kc=kc, drive=drive,
                 target=target, sim_state=None)
    print(f"  {fb.n:,} neurons on {fb.device}, mushroom body: {mb.stats()}")


@app.get("/")
async def index():
    return FileResponse(ROOT / "real-brain.html")


@app.get("/fly_drums_export.json")
async def export_file():
    # so the page's own "use recorded" toggle works even while this live
    # server is the one answering — it just reads the same file back
    return FileResponse(ROOT / "fly_drums_export.json")


_cloud_cache = None


@app.get("/neuron_cloud.json")
async def neuron_cloud():
    global _cloud_cache
    if _cloud_cache is None:
        data = json.loads((ROOT / "fly_drums_export.json").read_text())
        _cloud_cache = data["neuron_cloud"]
    return JSONResponse(_cloud_cache)


def run_window(seed):
    fb, mb, populations, kc = state["fb"], state["mb"], state["populations"], state["kc"]
    steps = int(round(WINDOW_MS / fb.p.dt))
    out = fb.run(state["drive"], steps=steps, record=populations, seed=seed, state=state["sim_state"])
    state["sim_state"] = out["_state"]
    mb.observe(out["_fired"][np.isin(out["_fired"], kc)])
    return {ch: (float(np.mean(out[ch])) if len(out[ch]) else 0.0) for ch in populations}


def run_bar(seed_base):
    raw = {ch: [] for ch in CHANNEL_SUBCLASS}
    for s in range(STEPS_PER_BAR):
        rates = run_window(seed_base * 1000 + s)
        for ch, v in rates.items():
            raw[ch].append(v)
    norm = {}
    for ch, vals in raw.items():
        arr = np.array(vals)
        lo, hi = arr.min(), arr.max()
        norm[ch] = ((arr - lo) / (hi - lo) if hi > lo else arr * 0).tolist()
    return norm


ALLOWED_ORIGINS = {"http://localhost:4670", "http://127.0.0.1:4670", "http://[::1]:4670"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    # cross-site WebSocket hijacking guard: this stream has no auth, so any
    # page in the browser could otherwise open it and read the live brain
    # data. A browser always sends Origin on a WebSocket handshake; a
    # same-machine, non-browser client (curl, a test) sends none and is let
    # through, since it isn't the cross-site case this is guarding against.
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    _conn_id["current"] += 1
    my_id = _conn_id["current"]  # a newer tab connecting supersedes this one — see checks below
    fb, mb = state["fb"], state["mb"]
    loop = asyncio.get_event_loop()

    def mine():
        return _conn_id["current"] == my_id

    try:
        await websocket.send_json({
            "type": "meta", "neurons": int(fb.n), "edges": int(fb.W.nnz),
            "channel_subclass": CHANNEL_SUBCLASS, "dt_ms": fb.p.dt, "drive_hz": DRIVE_HZ,
            "device": str(fb.device),
        })

        for gen in range(TRAIN_GENERATIONS):
            if not mine():
                return
            norm = await loop.run_in_executor(None, run_bar, gen)
            if not mine():
                return
            fit = fitness(norm, state["target"])
            mb.dopamine(+1, amount=fit / 100.0)
            mb.apply()
            await websocket.send_json({
                "type": "training", "gen": gen + 1, "total": TRAIN_GENERATIONS,
                "fitness": fit, "depressed": mb.stats()["depressed"], "synapses": mb.stats()["synapses"],
            })

        bar_i = 0
        while True:
            if not mine():
                return
            section = SECTIONS[bar_i % len(SECTIONS)]
            t0 = loop.time()
            norm = await loop.run_in_executor(None, run_bar, 1000 + bar_i)
            compute_s = loop.time() - t0
            if not mine():
                return
            density = {"intro": 0.6, "outro": 0.5}.get(section, 1.0)
            hits = []
            for step in range(STEPS_PER_BAR):
                for ch in CHANNEL_SUBCLASS:
                    p = norm[ch][step] * density
                    if p > 0.55:
                        hits.append({"step": step, "ch": ch, "vel": round(min(1.0, 0.4 + p * 0.7), 3)})
            await websocket.send_json({
                "type": "bar", "bar": bar_i, "section": section, "bpm": 100,
                "hits": hits, "compute_s": round(compute_s, 3),
            })
            bar_i += 1
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    setup()
    print("open http://localhost:4670")
    uvicorn.run(app, host="0.0.0.0", port=4670)
