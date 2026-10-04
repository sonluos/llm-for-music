"""Audio feature extraction: RMS, spectral centroid, spectral rolloff, band energy.

Each function takes a single waveform (or, for compute_rms, a batch of equal-
length waveforms) and returns a scalar description of it. Building a feature
time series (e.g. one value per STFT frame) is just calling these in a loop
over segments/frames, the same way llm_music.pipeline loops apply_eq_chain
over segments.
"""

from typing import Sequence, Tuple

import torch

from llm_music.spectrum import compute_fft


def compute_rms(waveform: torch.Tensor) -> torch.Tensor:
    """Compute the root-mean-square level of a waveform.

    Args:
        waveform: Tensor of shape (samples,) or (batch, samples).

    Returns:
        RMS along the last dimension: a 0-d tensor for 1D input, or a tensor
        of shape (batch,) for 2D input.
    """
    if waveform.numel() == 0:
        raise ValueError("waveform must not be empty")
    return torch.sqrt(torch.mean(waveform**2, dim=-1))


def spectral_centroid(waveform: torch.Tensor, sample_rate: int) -> float:
    """Compute the spectral centroid: the magnitude-weighted mean frequency, in Hz.

    Higher values indicate a "brighter" sound with more high-frequency content.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.

    Returns:
        The spectral centroid in Hz, or 0.0 for a silent (all-zero) waveform.
    """
    freqs, magnitude = compute_fft(waveform, sample_rate)
    total_magnitude = magnitude.sum()
    if total_magnitude == 0:
        return 0.0
    return ((freqs * magnitude).sum() / total_magnitude).item()


def spectral_rolloff(
    waveform: torch.Tensor, sample_rate: int, rolloff_percent: float = 0.85
) -> float:
    """Compute the spectral rolloff: the frequency below which most energy lies.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        rolloff_percent: Fraction of total spectral energy that must lie below
            the returned frequency (e.g. 0.85 for the standard 85% rolloff).

    Returns:
        The rolloff frequency in Hz, or 0.0 for a silent (all-zero) waveform.
    """
    if not 0 < rolloff_percent <= 1:
        raise ValueError(f"rolloff_percent must be in (0, 1], got {rolloff_percent}")

    freqs, magnitude = compute_fft(waveform, sample_rate)
    energy = magnitude**2
    cumulative_energy = torch.cumsum(energy, dim=0)
    total_energy = cumulative_energy[-1]
    if total_energy == 0:
        return 0.0

    idx = torch.searchsorted(cumulative_energy, rolloff_percent * total_energy)
    idx = min(idx.item(), freqs.shape[-1] - 1)
    return freqs[idx].item()


def band_energy(
    waveform: torch.Tensor, sample_rate: int, bands: Sequence[Tuple[float, float]]
) -> torch.Tensor:
    """Compute the spectral energy within each of several frequency bands.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        bands: Sequence of (low_hz, high_hz) pairs, each low < high. A bin at
            frequency f belongs to a band when low <= f < high.

    Returns:
        1D tensor of shape (len(bands),): summed squared-magnitude energy in
        each band, in the same order as `bands`.
    """
    if not bands:
        raise ValueError("bands must not be empty")
    for low, high in bands:
        if low < 0 or high <= low:
            raise ValueError(f"invalid band ({low}, {high}): require 0 <= low < high")

    freqs, magnitude = compute_fft(waveform, sample_rate)
    energy = magnitude**2
    return torch.stack(
        [energy[(freqs >= low) & (freqs < high)].sum() for low, high in bands]
    )
