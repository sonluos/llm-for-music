"""Tests for the short-time Fourier transform (STFT) / spectrogram."""

import pytest
import torch

from llm_music.preprocess import segment
from llm_music.signals import generate_sine
from llm_music.stft import compute_stft

SAMPLE_RATE = 44100


def test_stft_output_shapes() -> None:
    waveform = torch.randn(SAMPLE_RATE)
    n_fft = 1024
    hop_length = 256
    freqs, times, magnitude = compute_stft(waveform, SAMPLE_RATE, n_fft, hop_length)

    expected_num_frames = segment(waveform, n_fft, hop_length).shape[0]
    assert freqs.shape == (n_fft // 2 + 1,)
    assert times.shape == (expected_num_frames,)
    assert magnitude.shape == (n_fft // 2 + 1, expected_num_frames)


def test_stft_rejects_non_1d_waveform() -> None:
    with pytest.raises(ValueError):
        compute_stft(torch.randn(2, 1000), SAMPLE_RATE, n_fft=256)


def test_stft_rejects_integer_waveform() -> None:
    with pytest.raises(ValueError):
        compute_stft(torch.randint(-100, 100, (1000,)), SAMPLE_RATE, n_fft=256)


def test_stft_rejects_non_positive_sample_rate() -> None:
    with pytest.raises(ValueError):
        compute_stft(torch.randn(1000), sample_rate=0, n_fft=256)


def test_stft_rejects_invalid_n_fft() -> None:
    with pytest.raises(ValueError):
        compute_stft(torch.randn(1000), SAMPLE_RATE, n_fft=0)


def test_stft_tracks_stationary_tone_frequency_in_every_frame() -> None:
    freq = 1000.0
    waveform = generate_sine(freq, 1.0, SAMPLE_RATE)
    freqs, _, magnitude = compute_stft(waveform, SAMPLE_RATE, n_fft=1024, hop_length=256)

    peak_freqs = freqs[magnitude.argmax(dim=0)]
    assert torch.all((peak_freqs - freq).abs() < 50.0)


def test_stft_tracks_frequency_change_over_time() -> None:
    n_fft = 2048
    hop_length = 512
    waveform = torch.cat(
        [generate_sine(440.0, 1.0, SAMPLE_RATE), generate_sine(2000.0, 1.0, SAMPLE_RATE)]
    )
    freqs, times, magnitude = compute_stft(waveform, SAMPLE_RATE, n_fft, hop_length)

    early_frame = (times - 0.5).abs().argmin()
    late_frame = (times - 1.5).abs().argmin()

    early_peak_freq = freqs[magnitude[:, early_frame].argmax()].item()
    late_peak_freq = freqs[magnitude[:, late_frame].argmax()].item()

    assert abs(early_peak_freq - 440.0) < 50.0
    assert abs(late_peak_freq - 2000.0) < 50.0
