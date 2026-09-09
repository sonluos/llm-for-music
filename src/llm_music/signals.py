"""Synthetic test signal generators."""

import math
from typing import Sequence

import torch


def generate_sine(
    freq: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Generate a single sine wave.

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Peak amplitude.

    Returns:
        1D tensor of shape (num_samples,).
    """
    num_samples = int(duration * sample_rate)
    # Phase (2*pi*freq*t) grows large for long durations/high frequencies; computing
    # it in float64 avoids the phase-precision loss float32 would introduce there.
    t = torch.arange(num_samples, dtype=torch.float64) / sample_rate
    signal = amplitude * torch.sin(2 * math.pi * freq * t)
    return signal.to(torch.float32)


def generate_multitone(
    freqs: Sequence[float],
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Generate a normalized sum of sine waves at the given frequencies.

    Args:
        freqs: Frequencies in Hz.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Peak amplitude of the combined signal after normalization.

    Returns:
        1D tensor of shape (num_samples,).
    """
    if not freqs:
        raise ValueError("freqs must contain at least one frequency")

    signal = sum(generate_sine(f, duration, sample_rate, amplitude=1.0) for f in freqs)
    peak = signal.abs().max()
    if peak > 0:
        signal = signal * (amplitude / peak)
    return signal
