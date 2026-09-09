"""1D linear convolution utilities."""

import torch
import torch.nn.functional as F


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
