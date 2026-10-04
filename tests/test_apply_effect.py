"""Tests for dispatching a predicted effect dict to real DSP."""

import pytest
import torch

from llm_music.apply_effect import apply_predicted_effect

SAMPLE_RATE = 44100


def test_apply_predicted_effect_lowpass_matches_eq_module() -> None:
    from llm_music.eq import apply_biquad, lowpass

    waveform = torch.randn(SAMPLE_RATE)
    params = {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7071}

    result = apply_predicted_effect(waveform, SAMPLE_RATE, params)

    b, a = lowpass(1000.0, 0.7071, SAMPLE_RATE)
    expected = apply_biquad(waveform, b, a)
    assert torch.equal(result, expected)


def test_apply_predicted_effect_reverb_matches_reverb_module() -> None:
    from llm_music.reverb import apply_reverb

    waveform = torch.randn(SAMPLE_RATE)
    params = {"effect": "reverb", "room_size": 0.6, "damping": 0.4, "decay": 0.7, "wet_dry": 0.5}

    result = apply_predicted_effect(waveform, SAMPLE_RATE, params)
    expected = apply_reverb(waveform, SAMPLE_RATE, 0.6, 0.4, 0.7, 0.5)
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "effect,params",
    [
        ("highpass", {"freq_hz": 500.0, "q": 1.0}),
        ("peaking", {"freq_hz": 1000.0, "gain_db": 3.0, "q": 1.0}),
        ("low_shelf", {"freq_hz": 200.0, "gain_db": -4.0}),
        ("high_shelf", {"freq_hz": 6000.0, "gain_db": 5.0}),
    ],
)
def test_apply_predicted_effect_runs_each_eq_type(effect, params) -> None:
    waveform = torch.randn(SAMPLE_RATE)
    full_params = {"effect": effect, **params}
    result = apply_predicted_effect(waveform, SAMPLE_RATE, full_params)
    assert result.shape == waveform.shape
    assert torch.isfinite(result).all()


def test_apply_predicted_effect_rejects_invalid_params() -> None:
    waveform = torch.randn(1000)
    with pytest.raises(ValueError):
        apply_predicted_effect(waveform, SAMPLE_RATE, {"effect": "lowpass", "freq_hz": 1000.0})


def test_apply_predicted_effect_rejects_unknown_effect() -> None:
    waveform = torch.randn(1000)
    with pytest.raises(ValueError):
        apply_predicted_effect(waveform, SAMPLE_RATE, {"effect": "flanger", "depth": 0.5})
