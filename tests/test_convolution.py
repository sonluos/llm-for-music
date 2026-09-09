"""Tests for 1D convolution utilities."""

import torch

from llm_music.convolution import convolve, moving_average_kernel


def test_convolution_output_length() -> None:
    signal = torch.randn(20)
    kernel = torch.randn(5)
    output = convolve(signal, kernel)
    assert output.shape[-1] == signal.shape[-1] + kernel.shape[-1] - 1


def test_impulse_convolution_reproduces_kernel() -> None:
    impulse = torch.tensor([1.0])
    kernel = torch.tensor([0.2, 0.5, 0.3, 0.1])
    output = convolve(impulse, kernel)
    assert torch.allclose(output, kernel, atol=1e-6)


def test_moving_average_kernel_sums_to_one() -> None:
    kernel = moving_average_kernel(7)
    assert kernel.shape[-1] == 7
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
