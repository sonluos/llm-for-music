"""Tests for parametric EQ (biquad) filter design and application."""

import math

import pytest
import torch

from llm_music.eq import apply_biquad, high_shelf, highpass, low_shelf, lowpass, peaking
from llm_music.spectrum import compute_fft, magnitude_to_db

SAMPLE_RATE = 44100


def _impulse_response(b: torch.Tensor, a: torch.Tensor, length: int = 8192) -> torch.Tensor:
    impulse = torch.zeros(length)
    impulse[0] = 1.0
    return apply_biquad(impulse, b, a)


def _magnitude_at(freqs: torch.Tensor, magnitude: torch.Tensor, target_freq: float) -> float:
    idx = (freqs - target_freq).abs().argmin()
    return magnitude[idx].item()


def test_lowpass_attenuates_high_frequency() -> None:
    b, a = lowpass(freq=1000.0, q=0.7071, sample_rate=SAMPLE_RATE)
    freqs, magnitude = compute_fft(_impulse_response(b, a), SAMPLE_RATE)
    assert _magnitude_at(freqs, magnitude, 10000.0) < _magnitude_at(freqs, magnitude, 100.0)


def test_highpass_attenuates_low_frequency() -> None:
    b, a = highpass(freq=1000.0, q=0.7071, sample_rate=SAMPLE_RATE)
    freqs, magnitude = compute_fft(_impulse_response(b, a), SAMPLE_RATE)
    assert _magnitude_at(freqs, magnitude, 100.0) < _magnitude_at(freqs, magnitude, 10000.0)


def test_peaking_boosts_center_frequency() -> None:
    gain_db = 12.0
    b, a = peaking(freq=1000.0, gain_db=gain_db, q=1.0, sample_rate=SAMPLE_RATE)
    freqs, magnitude = compute_fft(_impulse_response(b, a), SAMPLE_RATE)
    db = magnitude_to_db(magnitude)

    center_db = db[(freqs - 1000.0).abs().argmin()].item()
    far_db = db[(freqs - 100.0).abs().argmin()].item()

    assert abs(center_db - gain_db) < 1.0
    assert abs(far_db) < 1.0
    assert center_db > far_db


def test_peaking_cut_reduces_center_frequency() -> None:
    gain_db = -12.0
    b, a = peaking(freq=1000.0, gain_db=gain_db, q=1.0, sample_rate=SAMPLE_RATE)
    freqs, magnitude = compute_fft(_impulse_response(b, a), SAMPLE_RATE)
    db = magnitude_to_db(magnitude)

    center_db = db[(freqs - 1000.0).abs().argmin()].item()
    assert abs(center_db - gain_db) < 1.0


def test_low_shelf_boosts_low_frequencies() -> None:
    gain_db = 12.0
    b, a = low_shelf(freq=200.0, gain_db=gain_db, sample_rate=SAMPLE_RATE)
    freqs, magnitude = compute_fft(_impulse_response(b, a), SAMPLE_RATE)
    db = magnitude_to_db(magnitude)

    low_db = db[(freqs - 20.0).abs().argmin()].item()
    high_db = db[(freqs - 10000.0).abs().argmin()].item()

    assert abs(low_db - gain_db) < 1.0
    assert abs(high_db) < 1.0


def test_high_shelf_boosts_high_frequencies() -> None:
    gain_db = 12.0
    b, a = high_shelf(freq=5000.0, gain_db=gain_db, sample_rate=SAMPLE_RATE)
    freqs, magnitude = compute_fft(_impulse_response(b, a), SAMPLE_RATE)
    db = magnitude_to_db(magnitude)

    low_db = db[(freqs - 20.0).abs().argmin()].item()
    high_db = db[(freqs - 20000.0).abs().argmin()].item()

    assert abs(low_db) < 1.0
    assert abs(high_db - gain_db) < 1.0


def test_apply_biquad_output_length_matches_input() -> None:
    b, a = lowpass(freq=1000.0, q=0.7071, sample_rate=SAMPLE_RATE)
    signal = torch.randn(2000)
    output = apply_biquad(signal, b, a)
    assert output.shape == signal.shape


def test_apply_biquad_matches_signal_dtype_and_device() -> None:
    b, a = lowpass(freq=1000.0, q=0.7071, sample_rate=SAMPLE_RATE)
    signal = torch.randn(2000, dtype=torch.float64)
    output = apply_biquad(signal, b, a)
    assert output.dtype == torch.float64


@pytest.mark.parametrize("q", [0.0, -1.0])
def test_lowpass_rejects_non_positive_q(q: float) -> None:
    with pytest.raises(ValueError):
        lowpass(freq=1000.0, q=q, sample_rate=SAMPLE_RATE)


@pytest.mark.parametrize("shelf_slope", [0.0, -1.0])
def test_low_shelf_rejects_non_positive_shelf_slope(shelf_slope: float) -> None:
    with pytest.raises(ValueError):
        low_shelf(freq=200.0, gain_db=12.0, sample_rate=SAMPLE_RATE, shelf_slope=shelf_slope)


def test_high_shelf_rejects_shelf_slope_with_negative_discriminant() -> None:
    with pytest.raises(ValueError):
        high_shelf(freq=5000.0, gain_db=24.0, sample_rate=SAMPLE_RATE, shelf_slope=100.0)
