"""Tests for the Schroeder/Freeverb-style reverb module."""

import pytest
import torch

from llm_music.reverb import apply_reverb

SAMPLE_RATE = 44100


def test_apply_reverb_output_shape_matches_input() -> None:
    waveform = torch.randn(SAMPLE_RATE)
    output = apply_reverb(waveform, SAMPLE_RATE, room_size=0.5, damping=0.5, decay=0.5, wet_dry=0.5)
    assert output.shape == waveform.shape


def test_apply_reverb_is_finite() -> None:
    waveform = torch.randn(SAMPLE_RATE * 2) * 0.5
    output = apply_reverb(waveform, SAMPLE_RATE, room_size=0.9, damping=0.1, decay=0.95, wet_dry=0.8)
    assert torch.isfinite(output).all()


def test_apply_reverb_wet_dry_zero_is_dry_signal() -> None:
    waveform = torch.randn(SAMPLE_RATE)
    output = apply_reverb(waveform, SAMPLE_RATE, room_size=0.5, damping=0.5, decay=0.5, wet_dry=0.0)
    assert torch.equal(output, waveform)


def test_apply_reverb_wet_dry_one_is_fully_wet() -> None:
    waveform = torch.randn(SAMPLE_RATE)
    output = apply_reverb(waveform, SAMPLE_RATE, room_size=0.5, damping=0.5, decay=0.5, wet_dry=1.0)
    assert not torch.equal(output, waveform)


def test_apply_reverb_impulse_response_decays_over_time() -> None:
    n = SAMPLE_RATE * 2
    impulse = torch.zeros(n)
    impulse[0] = 1.0
    output = apply_reverb(impulse, SAMPLE_RATE, room_size=0.8, damping=0.3, decay=0.9, wet_dry=1.0)

    window = SAMPLE_RATE // 10
    early_rms = output[:window].pow(2).mean().sqrt()
    late_rms = output[-window:].pow(2).mean().sqrt()
    assert late_rms < early_rms


def test_apply_reverb_higher_decay_sustains_longer() -> None:
    n = SAMPLE_RATE * 2
    impulse = torch.zeros(n)
    impulse[0] = 1.0

    low_decay = apply_reverb(impulse, SAMPLE_RATE, room_size=0.8, damping=0.3, decay=0.2, wet_dry=1.0)
    high_decay = apply_reverb(impulse, SAMPLE_RATE, room_size=0.8, damping=0.3, decay=0.95, wet_dry=1.0)

    window = SAMPLE_RATE // 10
    tail_slice = slice(SAMPLE_RATE, SAMPLE_RATE + window)  # the 1.0-1.1s window
    assert high_decay[tail_slice].pow(2).mean() > low_decay[tail_slice].pow(2).mean()


def test_apply_reverb_rejects_non_1d_waveform() -> None:
    with pytest.raises(ValueError):
        apply_reverb(torch.randn(2, 1000), SAMPLE_RATE, 0.5, 0.5, 0.5, 0.5)


@pytest.mark.parametrize("param", ["room_size", "damping", "decay", "wet_dry"])
def test_apply_reverb_rejects_out_of_range_params(param: str) -> None:
    kwargs = {"room_size": 0.5, "damping": 0.5, "decay": 0.5, "wet_dry": 0.5}
    kwargs[param] = 1.5
    with pytest.raises(ValueError):
        apply_reverb(torch.randn(1000), SAMPLE_RATE, **kwargs)
