"""Tests for bit-depth quantization and signal quality comparison."""

import pytest
import torch

from llm_music.quantize import quantize_bit_depth, signal_to_noise_ratio, theoretical_sqnr_db
from llm_music.signals import generate_sine


def test_quantize_bit_depth_limits_unique_levels() -> None:
    # Standard signed N-bit PCM: integer codes from -2**(bits-1) to
    # 2**(bits-1) - 1, giving exactly 2**bits distinct levels.
    waveform = torch.linspace(-1.0, 1.0, steps=10000)
    quantized = quantize_bit_depth(waveform, bits=3)
    assert torch.unique(quantized).numel() <= 2**3


def test_quantize_bit_depth_stays_in_range() -> None:
    waveform = torch.randn(1000) * 3  # includes values outside [-1, 1]
    quantized = quantize_bit_depth(waveform, bits=8)
    assert quantized.min() >= -1.0
    assert quantized.max() <= 1.0


def test_quantize_bit_depth_rejects_invalid_bits() -> None:
    with pytest.raises(ValueError):
        quantize_bit_depth(torch.randn(10), bits=0)


def test_snr_identical_signals_is_infinite() -> None:
    waveform = torch.randn(100)
    assert signal_to_noise_ratio(waveform, waveform) == float("inf")


def test_snr_decreases_with_more_noise() -> None:
    waveform = torch.randn(10000)
    small_noise = waveform + 0.01 * torch.randn(10000)
    large_noise = waveform + 0.5 * torch.randn(10000)
    assert signal_to_noise_ratio(waveform, small_noise) > signal_to_noise_ratio(waveform, large_noise)


def test_snr_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        signal_to_noise_ratio(torch.randn(100), torch.randn(50))


def test_snr_silent_reference_with_noise_is_negative_infinity() -> None:
    reference = torch.zeros(100)
    test = torch.randn(100) * 0.1
    assert signal_to_noise_ratio(reference, test) == float("-inf")


def test_snr_silent_reference_matching_silent_test_is_infinite() -> None:
    reference = torch.zeros(100)
    assert signal_to_noise_ratio(reference, reference.clone()) == float("inf")


def test_theoretical_sqnr_rejects_non_positive_bits() -> None:
    with pytest.raises(ValueError):
        theoretical_sqnr_db(0)


def test_quantization_snr_matches_theoretical_sqnr() -> None:
    # A near-full-scale, non-harmonic sine approximates the textbook SQNR derivation.
    waveform = generate_sine(437.0, 2.0, sample_rate=44100, amplitude=0.99)
    for bits in (8, 12, 16):
        quantized = quantize_bit_depth(waveform, bits)
        measured = signal_to_noise_ratio(waveform, quantized)
        assert measured == pytest.approx(theoretical_sqnr_db(bits), abs=1.0)
