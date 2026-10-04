"""Frequency-domain analysis utilities."""

from typing import Tuple

import torch


def compute_fft(waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the single-sided FFT magnitude spectrum of a waveform.

    Args:
        waveform: 1D tensor of shape (samples,), or 2D tensor of shape
            (channels, samples) which is averaged across channels.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (freqs, magnitude), both 1D tensors of length samples // 2 + 1.
    """
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.dim() != 1:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(waveform.shape)}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    num_samples = waveform.shape[-1]
    spectrum = torch.fft.rfft(waveform)
    freqs = torch.fft.rfftfreq(num_samples, d=1.0 / sample_rate, device=waveform.device)
    magnitude = spectrum.abs()
    return freqs, magnitude


def magnitude_to_db(magnitude: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Convert a linear magnitude spectrum to decibels.

    Args:
        magnitude: Linear magnitude tensor.
        eps: Floor value to avoid log(0).

    Returns:
        Tensor of the same shape, in dB.
    """
    return 20.0 * torch.log10(magnitude.clamp_min(eps))
