"""Tests for normalizing audio features and formatting them as LLM input."""

import json

import pytest
import torch

from llm_music.llm_input import (
    audio_to_llm_input,
    extract_feature_vector,
    format_for_llm_input,
    normalize_features,
)
from llm_music.signals import generate_multitone, generate_sine

SAMPLE_RATE = 44100


def test_extract_feature_vector_has_expected_keys() -> None:
    waveform = generate_sine(1000.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    features = extract_feature_vector(waveform, SAMPLE_RATE)

    expected_keys = {
        "rms",
        "spectral_centroid_hz",
        "spectral_rolloff_hz",
        "low_energy_frac",
        "mid_energy_frac",
        "high_energy_frac",
    }
    assert set(features.keys()) == expected_keys


def test_extract_feature_vector_band_fractions_sum_to_one_when_fully_covered() -> None:
    # All three tones fall within the default bands' combined range (20-20000 Hz),
    # so the fractions (of *total* spectral energy) should sum to ~1 here.
    waveform = generate_multitone((440.0, 1000.0, 4000.0), 1.0, SAMPLE_RATE, amplitude=0.8)
    features = extract_feature_vector(waveform, SAMPLE_RATE)

    band_fraction_sum = (
        features["low_energy_frac"] + features["mid_energy_frac"] + features["high_energy_frac"]
    )
    assert band_fraction_sum == pytest.approx(1.0, abs=1e-5)


def test_extract_feature_vector_band_fractions_reflect_out_of_band_energy() -> None:
    # A 10 Hz tone falls entirely below the default bands' 20 Hz floor, so its
    # energy is NOT part of the selected bands: fractions should be ~0, not sum
    # to 1 via leakage alone (the bug this denominator fix addresses).
    waveform = generate_sine(10.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    features = extract_feature_vector(waveform, SAMPLE_RATE)

    band_fraction_sum = (
        features["low_energy_frac"] + features["mid_energy_frac"] + features["high_energy_frac"]
    )
    assert band_fraction_sum < 0.01


def test_extract_feature_vector_silence_gives_zero_band_fractions() -> None:
    waveform = torch.zeros(1000)
    features = extract_feature_vector(waveform, SAMPLE_RATE)
    assert features["low_energy_frac"] == 0.0
    assert features["mid_energy_frac"] == 0.0
    assert features["high_energy_frac"] == 0.0


def test_extract_feature_vector_rejects_mismatched_band_names() -> None:
    waveform = generate_sine(1000.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        extract_feature_vector(waveform, SAMPLE_RATE, bands=[(0.0, 100.0)], band_names=["a", "b"])


def test_extract_feature_vector_rejects_duplicate_band_names() -> None:
    waveform = generate_sine(1000.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        extract_feature_vector(
            waveform, SAMPLE_RATE, bands=[(0.0, 100.0), (100.0, 200.0)], band_names=["a", "a"]
        )


def test_extract_feature_vector_rejects_reserved_band_name() -> None:
    waveform = generate_sine(1000.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        extract_feature_vector(waveform, SAMPLE_RATE, bands=[(0.0, 100.0)], band_names=["rms"])


def test_normalize_features_scales_hz_fields_by_nyquist() -> None:
    raw = {"spectral_centroid_hz": SAMPLE_RATE / 4, "rms": 0.5, "low_energy_frac": 0.3}
    normalized = normalize_features(raw, SAMPLE_RATE)
    assert normalized["spectral_centroid_hz"] == pytest.approx(0.5)
    assert normalized["rms"] == 0.5
    assert normalized["low_energy_frac"] == 0.3


def test_normalize_features_clamps_out_of_range_values() -> None:
    raw = {"spectral_rolloff_hz": SAMPLE_RATE, "rms": 5.0}
    normalized = normalize_features(raw, SAMPLE_RATE)
    assert normalized["spectral_rolloff_hz"] == 1.0
    assert normalized["rms"] == 1.0


def test_format_for_llm_input_is_valid_json() -> None:
    normalized = {"rms": 0.40135, "spectral_centroid_hz": 0.10054}
    text = format_for_llm_input(normalized, precision=3)
    parsed = json.loads(text)
    assert parsed == {"rms": 0.401, "spectral_centroid_hz": 0.101}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_format_for_llm_input_rejects_non_finite_values(bad_value: float) -> None:
    with pytest.raises(ValueError):
        format_for_llm_input({"rms": bad_value})


@pytest.mark.parametrize("sample_rate", [0, -44100])
def test_normalize_features_rejects_non_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError):
        normalize_features({"rms": 0.5}, sample_rate)


def test_audio_to_llm_input_end_to_end() -> None:
    waveform = generate_multitone((440.0, 1000.0), 1.0, SAMPLE_RATE, amplitude=0.8)
    text = audio_to_llm_input(waveform, SAMPLE_RATE)
    parsed = json.loads(text)

    assert set(parsed.keys()) == {
        "rms",
        "spectral_centroid_hz",
        "spectral_rolloff_hz",
        "low_energy_frac",
        "mid_energy_frac",
        "high_energy_frac",
    }
    for value in parsed.values():
        assert 0.0 <= value <= 1.0
