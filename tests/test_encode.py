"""Step 1.3 pass check: burst onsets land within 1 ms of the MIDI hit times and
the correct JO group is targeted. Uses made-up bodyIds, so no connectome data
or build/ files are needed."""

import mido
import numpy as np
import pytest

from src import encode

SEED = 0
VOICES = encode.load_voices()
CONFIG = encode.load_encoder_config()
DURATION = CONFIG["burst_duration_ms"] / 1000.0
# Six disjoint, sorted groups of synthetic bodyIds (group g = 1000*(g+1) + 0..39).
JO_GROUPS = [np.arange(1000 * (g + 1), 1000 * (g + 1) + 40, dtype=np.int64) for g in range(6)]


def run(hits, seed=SEED):
    return encode.encode(hits, JO_GROUPS, config=CONFIG, seed=seed, voices=VOICES)


def one_hit_per_voice(vel=100):
    # 0.5 s apart, so bursts (30 ms) never overlap.
    return [{"t": 0.5 * i, "voice": i, "vel": vel} for i in range(6)]


def test_burst_onsets_within_1ms_of_hits():
    hits = one_hit_per_voice()
    out = run(hits)
    assert len(out["bursts"]) == len(hits)
    for h, b in zip(hits, out["bursts"]):
        assert abs(b["onset"] - h["t"]) < 1e-3
        assert b["offset"] == pytest.approx(b["onset"] + DURATION)
        # The spikes themselves must not start before the hit either.
        sel = out["spike_group"] == b["group"]
        assert sel.any()
        assert out["spike_t"][sel].min() >= h["t"]
        assert out["spike_t"][sel].min() - h["t"] < DURATION


@pytest.mark.parametrize("voice", range(6))
def test_hit_targets_only_its_jo_group(voice):
    out = run([{"t": 0.25, "voice": voice, "vel": 127}])
    group = VOICES[voice]["jo_group"]
    assert out["bursts"][0]["group"] == group
    assert len(out["spike_t"]) > 0
    assert set(out["spike_group"].tolist()) == {group}
    assert np.isin(out["spike_body"], JO_GROUPS[group]).all()


def test_group_follows_voice_table_not_voice_index():
    # Reversed mapping: voice i -> JO group 5 - i.
    voices = [{**v, "jo_group": 5 - v["idx"]} for v in VOICES]
    out = encode.encode([{"t": 0.0, "voice": 1, "vel": 127}], JO_GROUPS, config=CONFIG, seed=SEED, voices=voices)
    assert set(out["spike_group"].tolist()) == {4}
    assert np.isin(out["spike_body"], JO_GROUPS[4]).all()


def test_spikes_fall_inside_their_burst_window():
    out = run(one_hit_per_voice())
    t, group = out["spike_t"], out["spike_group"]
    assert (np.diff(t) >= 0).all()
    inside = np.zeros(len(t), dtype=bool)
    for b in out["bursts"]:
        inside |= (group == b["group"]) & (t >= b["onset"]) & (t <= b["onset"] + DURATION)
    assert inside.all()
    for g in range(6):
        assert np.isin(out["spike_body"][group == g], JO_GROUPS[g]).all()


def test_rate_scales_with_velocity():
    def mean_spikes_per_neuron(vel):
        hits = [{"t": 0.5 * i, "voice": 0, "vel": vel} for i in range(200)]
        out = run(hits)
        assert out["bursts"][0]["rate_hz"] == pytest.approx(CONFIG["max_rate_hz"] * vel / CONFIG["velocity_full_scale"])
        return len(out["spike_t"]) / (len(hits) * len(JO_GROUPS[0])), out["bursts"][0]["rate_hz"] * DURATION

    loud, loud_expected = mean_spikes_per_neuron(127)
    soft, soft_expected = mean_spikes_per_neuron(40)
    # 8000 Poisson draws each: the mean is within a few percent of rate * duration.
    assert loud == pytest.approx(loud_expected, rel=0.05)
    assert soft == pytest.approx(soft_expected, rel=0.05)
    assert loud > soft


def test_same_seed_same_spikes_different_seed_different_spikes():
    hits = one_hit_per_voice()
    a, b, other = run(hits, seed=7), run(hits, seed=7), run(hits, seed=8)
    for key in ("spike_t", "spike_body", "spike_group"):
        assert np.array_equal(a[key], b[key])
    assert a["bursts"] == b["bursts"]
    assert not (len(a["spike_t"]) == len(other["spike_t"]) and np.array_equal(a["spike_t"], other["spike_t"]))


def test_no_hits_gives_empty_arrays():
    out = run([])
    assert out["bursts"] == []
    for key in ("spike_t", "spike_body", "spike_group"):
        assert len(out[key]) == 0
    assert out["spike_body"].dtype == np.int64


def test_read_hits_from_midi_file(tmp_path):
    # (tick, note, velocity) at 480 ticks per beat; 120 BPM -> 480 ticks = 0.5 s.
    # Note 60 is not in the voice table and must be ignored.
    notes = [(0, 36, 100), (0, 42, 80), (240, 60, 90), (480, 38, 127), (720, 42, 40), (960, 49, 1)]
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    events = sorted([(tick, "note_on", n, v) for tick, n, v in notes] + [(tick + 60, "note_off", n, 0) for tick, n, _ in notes])
    last = 0
    for tick, kind, note, vel in events:
        track.append(mido.Message(kind, channel=9, note=note, velocity=vel, time=tick - last))
        last = tick
    path = tmp_path / "call.mid"
    mid.save(path)

    hits = encode.read_hits(path, voices=VOICES)
    assert [(h["voice"], h["vel"]) for h in hits] == [(0, 100), (2, 80), (1, 127), (2, 40), (5, 1)]
    assert [h["t"] for h in hits] == pytest.approx([0.0, 0.0, 0.5, 0.75, 1.0])

    # End to end: the bursts start within 1 ms of those MIDI times.
    out = run(hits)
    assert [b["onset"] for b in out["bursts"]] == pytest.approx([0.0, 0.0, 0.5, 0.75, 1.0], abs=1e-3)
