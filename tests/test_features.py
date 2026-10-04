"""Tests for RMS, spectral centroid, spectral rolloff, and band energy features."""

import math

import pytest
import torch

from llm_music.features import band_energy, compute_rms, spectral_centroid, spectral_rolloff
from llm_music.signals import generate_multitone, generate_sine

SAMPLE_RATE = 44100


def test_compute_rms_of_sine_matches_theory() -> None:
    amplitude = 0.8
    waveform = generate_sine(1000.0, 1.0, SAMPLE_RATE, amplitude=amplitude)
    expected = amplitude / math.sqrt(2)
    assert compute_rms(waveform).item() == pytest.approx(expected, abs=1e-3)


def test_compute_rms_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_rms(torch.empty(0))


def test_compute_rms_batched_matches_per_row() -> None:
    batch = torch.randn(3, 1000)
    batched_rms = compute_rms(batch)
    for i in range(3):
        assert batched_rms[i].item() == pytest.approx(compute_rms(batch[i]).item())


def test_spectral_centroid_of_pure_tone_matches_frequency() -> None:
    freq = 1000.0
    waveform = generate_sine(freq, 2.0, SAMPLE_RATE, amplitude=0.8)
    assert spectral_centroid(waveform, SAMPLE_RATE) == pytest.approx(freq, abs=5.0)


def test_spectral_centroid_silence_is_zero() -> None:
    waveform = torch.zeros(1000)
    assert spectral_centroid(waveform, SAMPLE_RATE) == 0.0


def test_spectral_rolloff_pure_tone_matches_frequency() -> None:
    freq = 1000.0
    waveform = generate_sine(freq, 2.0, SAMPLE_RATE, amplitude=0.8)
    assert spectral_rolloff(waveform, SAMPLE_RATE, rolloff_percent=0.85) == pytest.approx(
        freq, abs=5.0
    )


def test_spectral_rolloff_increases_with_percent() -> None:
    waveform = generate_multitone((440.0, 1000.0, 4000.0), 2.0, SAMPLE_RATE, amplitude=0.8)
    low = spectral_rolloff(waveform, SAMPLE_RATE, rolloff_percent=0.3)
    mid = spectral_rolloff(waveform, SAMPLE_RATE, rolloff_percent=0.6)
    high = spectral_rolloff(waveform, SAMPLE_RATE, rolloff_percent=0.95)
    assert low < mid < high


def test_spectral_rolloff_silence_is_zero() -> None:
    waveform = torch.zeros(1000)
    assert spectral_rolloff(waveform, SAMPLE_RATE) == 0.0


@pytest.mark.parametrize("percent", [0.0, 1.5, -0.1])
def test_spectral_rolloff_rejects_invalid_percent(percent: float) -> None:
    with pytest.raises(ValueError):
        spectral_rolloff(torch.randn(1000), SAMPLE_RATE, rolloff_percent=percent)


def test_band_energy_concentrates_in_correct_band() -> None:
    waveform = generate_sine(2000.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    bands = [(0.0, 1000.0), (1000.0, 3000.0), (3000.0, SAMPLE_RATE / 2)]
    energies = band_energy(waveform, SAMPLE_RATE, bands)

    assert energies.shape == (3,)
    assert energies[1] > energies[0] * 100
    assert energies[1] > energies[2] * 100


def test_band_energy_sums_to_total_energy() -> None:
    waveform = generate_multitone((440.0, 1000.0, 4000.0), 1.0, SAMPLE_RATE, amplitude=0.8)
    nyquist = SAMPLE_RATE / 2
    bands = [(0.0, nyquist / 2), (nyquist / 2, nyquist + 1.0)]
    energies = band_energy(waveform, SAMPLE_RATE, bands)

    from llm_music.spectrum import compute_fft

    _, magnitude = compute_fft(waveform, SAMPLE_RATE)
    total_energy = (magnitude**2).sum()
    assert energies.sum().item() == pytest.approx(total_energy.item(), rel=1e-5)


@pytest.mark.parametrize("bands", [[(-10.0, 100.0)], [(100.0, 50.0)], [(100.0, 100.0)]])
def test_band_energy_rejects_invalid_bands(bands) -> None:
    with pytest.raises(ValueError):
        band_energy(torch.randn(1000), SAMPLE_RATE, bands)


def test_band_energy_rejects_empty_bands() -> None:
    with pytest.raises(ValueError):
        band_energy(torch.randn(1000), SAMPLE_RATE, [])


@pytest.mark.parametrize("sample_rate", [0, -44100])
def test_spectral_centroid_rejects_non_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        spectral_centroid(torch.randn(1000), sample_rate)


@pytest.mark.parametrize("sample_rate", [0, -44100])
def test_spectral_rolloff_rejects_non_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        spectral_rolloff(torch.randn(1000), sample_rate)


@pytest.mark.parametrize("sample_rate", [0, -44100])
def test_band_energy_rejects_non_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        band_energy(torch.randn(1000), sample_rate, [(0.0, 100.0)])
