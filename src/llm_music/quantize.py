"""Bit-depth quantization and signal quality comparison (analog-vs-digital tradeoffs).

Sample-rate quality tradeoffs are covered by llm_music.preprocess.resample; this
module adds the other half of A/D conversion (bit-depth quantization) plus a
shared SNR metric used to compare degraded signals against the original.
"""

import math

import torch


def quantize_bit_depth(waveform: torch.Tensor, bits: int) -> torch.Tensor:
    """Simulate uniform quantization to a reduced bit depth.

    The waveform is clamped to [-1, 1] (the usual normalized audio range) and
    rounded to the nearest of exactly 2**bits evenly spaced levels, matching
    standard signed N-bit PCM (integer codes from -2**(bits-1) to
    2**(bits-1) - 1), modeling how an ADC/DAC with fewer bits per sample would
    represent the signal.

    Args:
        waveform: Tensor of any shape, assumed to be in [-1, 1].
        bits: Target bit depth (e.g. 16, 8, 4).

    Returns:
        Quantized tensor, same shape as input.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    scale = 2 ** (bits - 1)
    clamped = waveform.clamp(-1.0, 1.0)
    codes = torch.round(clamped * scale).clamp(-scale, scale - 1)
    return codes / scale


def signal_to_noise_ratio(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Compute the signal-to-noise ratio of a degraded signal against a reference, in dB.

    Args:
        reference: Clean/original tensor.
        test: Degraded tensor, same shape as reference (e.g. quantized or
            resampled-and-restored).

    Returns:
        SNR in dB, as 10*log10(signal_power / noise_power). Returns float('inf')
        if test exactly matches reference, or float('-inf') if reference is
        silent (all zeros) but test is not.
    """
    if reference.shape != test.shape:
        raise ValueError(
            f"reference and test must have the same shape, got {tuple(reference.shape)} "
            f"and {tuple(test.shape)}"
        )
    noise_power = torch.mean((reference - test) ** 2).item()
    if noise_power == 0:
        return float("inf")
    signal_power = torch.mean(reference**2).item()
    if signal_power == 0:
        return float("-inf")
    return 10 * math.log10(signal_power / noise_power)


def theoretical_sqnr_db(bits: int) -> float:
    """Theoretical signal-to-quantization-noise ratio for a full-scale sine wave, in dB.

    Uses the standard engineering approximation SQNR ≈ 6.02*bits + 1.76 dB.

    Args:
        bits: Bit depth.

    Returns:
        Expected SQNR in dB.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    return 6.02 * bits + 1.76
