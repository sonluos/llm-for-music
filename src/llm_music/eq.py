"""Parametric EQ: biquad (2nd-order IIR) filter design and application.

Coefficients follow the RBJ "Audio EQ Cookbook" formulas. Each design function
returns (b, a) difference-equation coefficients for use with apply_biquad.
"""

import math
from typing import Sequence, Tuple

import torch
import torchaudio.functional as AF


def _normalize(
    b0: float, b1: float, b2: float, a0: float, a1: float, a2: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    b = torch.tensor([b0, b1, b2], dtype=torch.float32) / a0
    a = torch.tensor([a0, a1, a2], dtype=torch.float32) / a0
    return b, a


def _validate_freq(freq: float, sample_rate: int) -> None:
    if not 0 < freq < sample_rate / 2:
        raise ValueError(f"freq must be in (0, {sample_rate / 2}), got {freq}")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _shelf_alpha(a_gain: float, shelf_slope: float, sin_w0: float) -> float:
    _validate_positive("shelf_slope", shelf_slope)
    discriminant = (a_gain + 1 / a_gain) * (1 / shelf_slope - 1) + 2
    if discriminant < 0:
        raise ValueError(
            f"gain_db/shelf_slope={shelf_slope} combination is invalid "
            f"(negative discriminant); use a smaller |gain_db| or shelf_slope closer to 1.0"
        )
    return sin_w0 / 2 * math.sqrt(discriminant)


def lowpass(freq: float, q: float, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design a 2nd-order low-pass biquad filter.

    Args:
        freq: Cutoff frequency in Hz.
        q: Quality factor (0.7071 gives a maximally flat Butterworth response).
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (b, a) coefficients, each a 1D tensor of shape (3,).
    """
    _validate_freq(freq, sample_rate)
    _validate_positive("q", q)
    w0 = 2 * math.pi * freq / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2 * q)

    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def highpass(freq: float, q: float, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design a 2nd-order high-pass biquad filter.

    Args:
        freq: Cutoff frequency in Hz.
        q: Quality factor (0.7071 gives a maximally flat Butterworth response).
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (b, a) coefficients, each a 1D tensor of shape (3,).
    """
    _validate_freq(freq, sample_rate)
    _validate_positive("q", q)
    w0 = 2 * math.pi * freq / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2 * q)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def peaking(
    freq: float, gain_db: float, q: float, sample_rate: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design a peaking (bell) EQ biquad filter.

    Args:
        freq: Center frequency in Hz.
        gain_db: Boost (positive) or cut (negative) at the center frequency, in dB.
        q: Quality factor; higher values give a narrower bell.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (b, a) coefficients, each a 1D tensor of shape (3,).
    """
    _validate_freq(freq, sample_rate)
    _validate_positive("q", q)
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * freq / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2 * q)

    b0 = 1 + alpha * a_gain
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a_gain
    a0 = 1 + alpha / a_gain
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a_gain
    return _normalize(b0, b1, b2, a0, a1, a2)


def low_shelf(
    freq: float, gain_db: float, sample_rate: int, shelf_slope: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design a low-shelf EQ biquad filter.

    Args:
        freq: Corner frequency in Hz.
        gain_db: Boost (positive) or cut (negative) below the corner, in dB.
        sample_rate: Sample rate in Hz.
        shelf_slope: Shelf slope S; 1.0 gives the steepest shelf without overshoot.

    Returns:
        Tuple of (b, a) coefficients, each a 1D tensor of shape (3,).
    """
    _validate_freq(freq, sample_rate)
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * freq / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = _shelf_alpha(a_gain, shelf_slope, sin_w0)
    sqrt_a_2alpha = 2 * math.sqrt(a_gain) * alpha

    b0 = a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 + sqrt_a_2alpha)
    b1 = 2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cos_w0)
    b2 = a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 - sqrt_a_2alpha)
    a0 = (a_gain + 1) + (a_gain - 1) * cos_w0 + sqrt_a_2alpha
    a1 = -2 * ((a_gain - 1) + (a_gain + 1) * cos_w0)
    a2 = (a_gain + 1) + (a_gain - 1) * cos_w0 - sqrt_a_2alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def high_shelf(
    freq: float, gain_db: float, sample_rate: int, shelf_slope: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Design a high-shelf EQ biquad filter.

    Args:
        freq: Corner frequency in Hz.
        gain_db: Boost (positive) or cut (negative) above the corner, in dB.
        sample_rate: Sample rate in Hz.
        shelf_slope: Shelf slope S; 1.0 gives the steepest shelf without overshoot.

    Returns:
        Tuple of (b, a) coefficients, each a 1D tensor of shape (3,).
    """
    _validate_freq(freq, sample_rate)
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * freq / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = _shelf_alpha(a_gain, shelf_slope, sin_w0)
    sqrt_a_2alpha = 2 * math.sqrt(a_gain) * alpha

    b0 = a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 + sqrt_a_2alpha)
    b1 = -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos_w0)
    b2 = a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 - sqrt_a_2alpha)
    a0 = (a_gain + 1) - (a_gain - 1) * cos_w0 + sqrt_a_2alpha
    a1 = 2 * ((a_gain - 1) - (a_gain + 1) * cos_w0)
    a2 = (a_gain + 1) - (a_gain - 1) * cos_w0 - sqrt_a_2alpha
    return _normalize(b0, b1, b2, a0, a1, a2)


def apply_biquad(signal: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Apply a biquad IIR filter to a signal via its difference-equation coefficients.

    Args:
        signal: 1D tensor of shape (samples,).
        b: Feedforward (numerator) coefficients, shape (3,).
        a: Feedback (denominator) coefficients, shape (3,), with a[0] == 1.

    Returns:
        1D tensor of shape (samples,), the filtered signal.
    """
    if signal.dim() != 1:
        raise ValueError(f"Expected 1D signal, got shape {tuple(signal.shape)}")
    b = b.to(dtype=signal.dtype, device=signal.device)
    a = a.to(dtype=signal.dtype, device=signal.device)
    return AF.lfilter(signal, a, b, clamp=False)


def apply_eq_chain(
    signal: torch.Tensor, filters: Sequence[Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Apply a sequence of biquad filters to a signal, each feeding into the next.

    Models a real parametric EQ, which is usually several bands (e.g. a
    high-pass plus a couple of peaking bands) applied in series rather than
    a single filter.

    Args:
        signal: 1D tensor of shape (samples,).
        filters: Sequence of (b, a) coefficient pairs, each as returned by one
            of this module's design functions (lowpass, highpass, peaking,
            low_shelf, high_shelf). Applied in list order. An empty sequence
            leaves the signal unchanged.

    Returns:
        1D tensor of shape (samples,), the signal after the full chain.
    """
    for b, a in filters:
        signal = apply_biquad(signal, b, a)
    return signal
