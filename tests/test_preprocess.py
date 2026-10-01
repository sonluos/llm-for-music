"""Tests for audio preprocessing: resampling, mono, normalization, segmentation."""

import pytest
import torch

from llm_music.audio_io import save_audio
from llm_music.preprocess import (
    normalize_peak,
    normalize_rms,
    preprocess_batch,
    resample,
    segment,
    to_mono,
)
from llm_music.signals import generate_sine


def test_resample_changes_length() -> None:
    waveform = torch.randn(1000)
    resampled = resample(waveform, orig_sr=1000, target_sr=2000)
    assert resampled.shape[-1] == pytest.approx(2000, abs=5)


def test_resample_noop_when_same_rate() -> None:
    waveform = torch.randn(500)
    resampled = resample(waveform, orig_sr=44100, target_sr=44100)
    assert torch.equal(resampled, waveform)


def test_to_mono_averages_channels() -> None:
    stereo = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    mono = to_mono(stereo)
    assert mono.shape == (3,)
    assert torch.allclose(mono, torch.tensor([2.0, 3.0, 4.0]))


def test_to_mono_passthrough_for_1d() -> None:
    waveform = torch.randn(10)
    assert torch.equal(to_mono(waveform), waveform)


def test_normalize_peak_scales_to_target() -> None:
    waveform = torch.tensor([-0.5, 0.2, 0.4])
    normalized = normalize_peak(waveform, target_peak=1.0)
    assert normalized.abs().max().item() == pytest.approx(1.0)


def test_normalize_rms_scales_to_target() -> None:
    waveform = torch.randn(10000)
    target_rms = 0.1
    normalized = normalize_rms(waveform, target_rms=target_rms)
    actual_rms = torch.sqrt(torch.mean(normalized**2)).item()
    assert actual_rms == pytest.approx(target_rms, rel=1e-4)


def test_segment_exact_multiple_length() -> None:
    waveform = torch.arange(12, dtype=torch.float32)
    segments = segment(waveform, segment_length=4)
    assert segments.shape == (3, 4)
    assert torch.equal(segments[1], torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_segment_pads_final_segment() -> None:
    waveform = torch.arange(10, dtype=torch.float32)
    segments = segment(waveform, segment_length=4)
    assert segments.shape == (3, 4)
    assert torch.equal(segments[2], torch.tensor([8.0, 9.0, 0.0, 0.0]))


def test_segment_with_overlap() -> None:
    waveform = torch.arange(10, dtype=torch.float32)
    segments = segment(waveform, segment_length=4, hop_length=2)
    assert torch.equal(segments[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(segments[1], torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_normalize_peak_rejects_negative_target() -> None:
    with pytest.raises(ValueError):
        normalize_peak(torch.randn(10), target_peak=-1.0)


def test_normalize_rms_rejects_negative_target() -> None:
    with pytest.raises(ValueError):
        normalize_rms(torch.randn(10), target_rms=-0.1)


def test_normalize_peak_rejects_empty_waveform() -> None:
    with pytest.raises(ValueError):
        normalize_peak(torch.empty(0))


def test_preprocess_batch_rejects_empty_paths() -> None:
    with pytest.raises(ValueError):
        preprocess_batch([], sample_rate=44100, segment_length=1000)


def test_preprocess_batch_concatenates_segments(tmp_path) -> None:
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    save_audio(path_a, generate_sine(300.0, 1.0, 22050), 22050)
    save_audio(path_b, generate_sine(600.0, 0.5, 44100), 44100)

    target_sr = 44100
    segment_length = target_sr // 4
    batch = preprocess_batch([path_a, path_b], sample_rate=target_sr, segment_length=segment_length)

    assert batch.shape[-1] == segment_length
    assert batch.shape[0] >= 1
    assert batch.abs().max().item() == pytest.approx(1.0, abs=1e-3)
