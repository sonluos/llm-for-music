"""Audio preprocessing: resampling, mono downmix, normalization, segmentation,
and a batch pipeline tying them together.
"""

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torchaudio

from llm_music.audio_io import load_audio


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a waveform to a new sample rate.

    Args:
        waveform: Tensor of shape (..., samples).
        orig_sr: Current sample rate in Hz.
        target_sr: Desired sample rate in Hz.

    Returns:
        Tensor of shape (..., new_samples). Returned unchanged if orig_sr == target_sr.
    """
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Downmix a multi-channel waveform to mono by averaging channels.

    Args:
        waveform: Tensor of shape (channels, samples) or (samples,).

    Returns:
        1D tensor of shape (samples,).
    """
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() == 2:
        return waveform.mean(dim=0)
    raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(waveform.shape)}")


def normalize_peak(waveform: torch.Tensor, target_peak: float = 1.0) -> torch.Tensor:
    """Scale a waveform so its peak absolute amplitude equals target_peak.

    Args:
        waveform: Tensor of any shape.
        target_peak: Desired peak absolute amplitude.

    Returns:
        Scaled tensor, same shape as input. Returned unchanged if silent (all zeros).
    """
    if target_peak < 0:
        raise ValueError(f"target_peak must be >= 0, got {target_peak}")
    if waveform.numel() == 0:
        raise ValueError("waveform must not be empty")
    peak = waveform.abs().max()
    if peak == 0:
        return waveform
    return waveform * (target_peak / peak)


def normalize_rms(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Scale a waveform so its root-mean-square level equals target_rms.

    Args:
        waveform: Tensor of any shape.
        target_rms: Desired RMS level.

    Returns:
        Scaled tensor, same shape as input. Returned unchanged if silent (all zeros).
    """
    if target_rms < 0:
        raise ValueError(f"target_rms must be >= 0, got {target_rms}")
    if waveform.numel() == 0:
        raise ValueError("waveform must not be empty")
    rms = torch.sqrt(torch.mean(waveform**2))
    if rms == 0:
        return waveform
    return waveform * (target_rms / rms)


def segment(
    waveform: torch.Tensor, segment_length: int, hop_length: Optional[int] = None
) -> torch.Tensor:
    """Split a 1D waveform into fixed-length segments, zero-padding the final one if needed.

    Args:
        waveform: 1D tensor of shape (samples,).
        segment_length: Number of samples per segment.
        hop_length: Stride between segment starts. Defaults to segment_length (no overlap).

    Returns:
        2D tensor of shape (num_segments, segment_length).
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(waveform.shape)}")
    if segment_length < 1:
        raise ValueError(f"segment_length must be >= 1, got {segment_length}")
    hop_length = hop_length if hop_length is not None else segment_length
    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1, got {hop_length}")

    num_samples = waveform.shape[-1]
    if num_samples <= segment_length:
        num_segments = 1
    else:
        num_segments = math.ceil((num_samples - segment_length) / hop_length) + 1

    padded_length = (num_segments - 1) * hop_length + segment_length
    padded = F.pad(waveform, (0, padded_length - num_samples))
    return padded.unfold(0, segment_length, hop_length)


def preprocess_batch(
    paths: Sequence[Union[str, Path]],
    sample_rate: int,
    segment_length: int,
    hop_length: Optional[int] = None,
    target_peak: Optional[float] = 1.0,
) -> torch.Tensor:
    """Load, resample, mono-downmix, peak-normalize, and segment a batch of audio files.

    Args:
        paths: Audio file paths to process.
        sample_rate: Sample rate every file is resampled to.
        segment_length: Number of samples per segment.
        hop_length: Stride between segment starts. Defaults to segment_length.
        target_peak: If given, peak-normalize each file before segmenting.

    Returns:
        2D tensor of shape (total_segments, segment_length), with segments from
        all files concatenated in input order.
    """
    if not paths:
        raise ValueError("paths must not be empty")

    all_segments = []
    for path in paths:
        waveform, _ = load_audio(path, target_sr=sample_rate, mono=True)
        waveform = to_mono(waveform)
        if target_peak is not None:
            waveform = normalize_peak(waveform, target_peak)
        all_segments.append(segment(waveform, segment_length, hop_length))
    return torch.cat(all_segments, dim=0)
