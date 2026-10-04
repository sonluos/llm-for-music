"""Reverb: a Schroeder/Freeverb-style parallel-comb + series-allpass reverberator.

Four independent, named controls (matching the project's EQ/Reverb parameter
schema in llm_music.baseline_model):
- room_size: scales each comb filter's delay length (a bigger room means a
  longer time between reflections).
- damping: a one-pole lowpass coefficient inside each comb filter's feedback
  loop — real rooms absorb high frequencies faster on each bounce, so higher
  damping darkens the reverb tail over time.
- decay: comb filter feedback gain (how slowly each reflection's energy dies
  out; closer to 1 means a longer, denser tail).
- wet_dry: final dry/wet mix (0 = unprocessed, 1 = fully reverberated).

Each comb/allpass stage is a feedback delay line, which is naturally defined
sample-by-sample. Both reduce algebraically to an equivalent closed-form IIR
difference equation (derived in each helper's docstring below), so the whole
chain runs through torchaudio.functional.lfilter rather than a slow Python
per-sample loop.
"""

from typing import Tuple

import torch
import torchaudio.functional as AF

# Comb/allpass delay times in milliseconds, at room_size = 1.0 (spread apart,
# per the classic Freeverb design, to avoid harmonically related resonances).
_COMB_DELAYS_MS: Tuple[float, ...] = (29.7, 37.1, 41.1, 43.7)
_ALLPASS_DELAYS_MS: Tuple[float, ...] = (5.0, 1.7)
_ALLPASS_GAIN = 0.5

# Decay in [0, 1] maps to a comb feedback gain in this range: high enough for
# an audible tail, capped below 1 so the filter always remains stable.
_MIN_FEEDBACK = 0.7
_MAX_FEEDBACK = 0.98


def _validate_unit_range(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _damped_comb_filter(
    signal: torch.Tensor, delay_samples: int, feedback: float, damping: float
) -> torch.Tensor:
    """Apply one damped comb filter via its equivalent closed-form IIR.

    A comb filter is usually described sample-by-sample as a delay line with
    feedback run through a one-pole lowpass (the "damping" element):
        lp[n] = (1 - damping) * y[n - delay] + damping * lp[n - 1]
        y[n]  = x[n] + feedback * lp[n]
    Substituting the first equation into the second and clearing the
    lowpass's denominator gives a single difference equation relating y only
    to x and its own past samples:
        y[n] - damping*y[n-1] - feedback*(1-damping)*y[n-delay]
            = x[n] - damping*x[n-1]
    which is exactly the (a, b) coefficient form torchaudio.functional.lfilter
    expects — letting it run the whole recursion internally instead of a
    Python loop.
    """
    a = torch.zeros(delay_samples + 1, dtype=signal.dtype, device=signal.device)
    a[0] = 1.0
    a[1] += -damping
    a[delay_samples] += -feedback * (1 - damping)

    b = torch.zeros(delay_samples + 1, dtype=signal.dtype, device=signal.device)
    b[0] = 1.0
    b[1] = -damping

    return AF.lfilter(signal, a, b, clamp=False)


def _allpass_filter(signal: torch.Tensor, delay_samples: int, gain: float) -> torch.Tensor:
    """Apply one Schroeder allpass filter via its closed-form IIR.

    Sample-by-sample: y[n] = -gain*x[n] + x[n-delay] + gain*y[n-delay]. This
    is already a finite-order difference equation (no extra damping element),
    so it maps directly to (a, b) coefficients for lfilter.
    """
    a = torch.zeros(delay_samples + 1, dtype=signal.dtype, device=signal.device)
    a[0] = 1.0
    a[delay_samples] += -gain

    b = torch.zeros(delay_samples + 1, dtype=signal.dtype, device=signal.device)
    b[0] = -gain
    b[delay_samples] += 1.0

    return AF.lfilter(signal, a, b, clamp=False)


def apply_reverb(
    waveform: torch.Tensor,
    sample_rate: int,
    room_size: float,
    damping: float,
    decay: float,
    wet_dry: float,
) -> torch.Tensor:
    """Apply a Schroeder/Freeverb-style reverb to a waveform.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        room_size: 0-1. Scales each comb filter's delay length.
        damping: 0-1. Feedback-path lowpass coefficient; higher darkens the
            tail over time.
        decay: 0-1. Comb filter feedback gain; higher means a longer tail.
        wet_dry: 0-1. Final mix (0 = dry only, 1 = wet/reverb only).

    Returns:
        1D tensor of shape (samples,), the reverberated signal.
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(waveform.shape)}")
    for name, value in (
        ("room_size", room_size),
        ("damping", damping),
        ("decay", decay),
        ("wet_dry", wet_dry),
    ):
        _validate_unit_range(name, value)

    feedback = _MIN_FEEDBACK + (_MAX_FEEDBACK - _MIN_FEEDBACK) * decay
    # room_size=0 still gives a short-but-nonzero delay (half the base time);
    # room_size=1 gives the full base delay.
    size_scale = 0.5 + 0.5 * room_size

    wet_signal = torch.zeros_like(waveform)
    for delay_ms in _COMB_DELAYS_MS:
        delay_samples = max(int(delay_ms * size_scale / 1000 * sample_rate), 2)
        wet_signal = wet_signal + _damped_comb_filter(waveform, delay_samples, feedback, damping)
    wet_signal = wet_signal / len(_COMB_DELAYS_MS)

    for delay_ms in _ALLPASS_DELAYS_MS:
        delay_samples = max(int(delay_ms / 1000 * sample_rate), 2)
        wet_signal = _allpass_filter(wet_signal, delay_samples, _ALLPASS_GAIN)

    return (1 - wet_dry) * waveform + wet_dry * wet_signal
