"""Tests for the DSP foundation: signal generation, FFT, and audio I/O."""

import pytest
import torch

from llm_music.audio_io import load_audio, save_audio
from llm_music.signals import generate_multitone, generate_sine
from llm_music.spectrum import compute_fft


def test_sine_length() -> None:
    sample_rate = 8000
    duration = 0.5
    waveform = generate_sine(440.0, duration, sample_rate)
    assert waveform.shape[-1] == int(duration * sample_rate)


def test_fft_peak_440hz() -> None:
    sample_rate = 8000
    duration = 1.0
    waveform = generate_sine(440.0, duration, sample_rate)
    freqs, magnitude = compute_fft(waveform, sample_rate)
    peak_freq = freqs[torch.argmax(magnitude)].item()
    assert abs(peak_freq - 440.0) < 1.0


def test_fft_peaks_for_demo_multitone() -> None:
    sample_rate = 44100
    waveform = generate_multitone((440.0, 1000.0), 2.0, sample_rate, amplitude=0.8)
    freqs, magnitude = compute_fft(waveform, sample_rate)
    peak_freqs = freqs[torch.topk(magnitude, 2).indices]

    assert torch.any(torch.abs(peak_freqs - 440.0) < 1.0)
    assert torch.any(torch.abs(peak_freqs - 1000.0) < 1.0)


@pytest.mark.parametrize("sample_rate", [0, -44100])
def test_fft_rejects_non_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        compute_fft(torch.randn(1000), sample_rate)


def test_save_load_consistency(tmp_path) -> None:
    sample_rate = 8000
    waveform = generate_sine(440.0, 0.5, sample_rate)
    path = tmp_path / "test_tone.wav"

    save_audio(path, waveform, sample_rate)
    loaded, loaded_sr = load_audio(path)

    assert loaded_sr == sample_rate
    assert loaded.shape[-1] == waveform.shape[-1]
    assert torch.allclose(loaded.squeeze(0), waveform, atol=1e-3)
