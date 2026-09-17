"""1D linear convolution and FIR filter analysis utilities."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from llm_music.spectrum import compute_fft


def convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Compute the full 1D linear convolution of a signal with a kernel.

    Args:
        signal: 1D tensor of shape (samples,).
        kernel: 1D tensor of shape (taps,).

    Returns:
        1D tensor of shape (samples + taps - 1,), the full convolution.
    """
    if signal.dim() != 1:
        raise ValueError(f"Expected 1D signal, got shape {tuple(signal.shape)}")
    if kernel.dim() != 1:
        raise ValueError(f"Expected 1D kernel, got shape {tuple(kernel.shape)}")

    taps = kernel.shape[-1]
    padded = F.pad(signal, (taps - 1, taps - 1))
    flipped_kernel = kernel.flip(0)

    output = F.conv1d(padded.view(1, 1, -1), flipped_kernel.view(1, 1, -1))
    return output.view(-1)


def moving_average_kernel(size: int) -> torch.Tensor:
    """Build a normalized moving-average (box) kernel.

    Args:
        size: Number of taps.

    Returns:
        1D tensor of shape (size,) whose values sum to 1.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return torch.ones(size, dtype=torch.float32) / size


def frequency_response(
    kernel: torch.Tensor,
    sample_rate: int,
    n_fft: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a FIR filter's frequency response from its kernel (coefficients).

    Unlike comparing a signal's spectrum before/after filtering, this analyzes
    the filter itself: the kernel is zero-padded and transformed directly, so
    the result does not depend on any particular input signal.

    Args:
        kernel: 1D tensor of filter coefficients, shape (taps,).
        sample_rate: Sample rate in Hz.
        n_fft: Zero-padded FFT length, for a smoother response curve. Defaults
            to max(len(kernel), 512).

    Returns:
        Tuple of (freqs, magnitude), the same convention as compute_fft.
    """
    if kernel.dim() != 1:
        raise ValueError(f"Expected 1D kernel, got shape {tuple(kernel.shape)}")

    taps = kernel.shape[-1]
    if n_fft is None:
        n_fft = max(taps, 512)
    elif n_fft < taps:
        raise ValueError(f"n_fft ({n_fft}) must be >= kernel length ({taps})")

    padded_kernel = F.pad(kernel, (0, n_fft - taps))
    return compute_fft(padded_kernel, sample_rate)
