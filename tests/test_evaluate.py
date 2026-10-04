"""Tests for the baseline-vs-proposed model evaluation metrics."""

import pytest
import torch

from llm_music.evaluate import (
    acoustic_feature_change,
    evaluate_predictions,
    parameter_error,
    repetition_stability,
)
from llm_music.signals import generate_sine

SAMPLE_RATE = 44100


# --- parameter_error --------------------------------------------------------


def test_parameter_error_computes_normalized_mean_absolute_error() -> None:
    predicted = {"effect": "low_shelf", "freq_hz": 200.0, "gain_db": 3.0}
    target = {"effect": "low_shelf", "freq_hz": 150.0, "gain_db": 5.0}

    expected = (abs(200.0 - 150.0) / 19980.0 + abs(3.0 - 5.0) / 48.0) / 2
    assert parameter_error(predicted, target) == pytest.approx(expected)


def test_parameter_error_zero_for_exact_match() -> None:
    params = {"effect": "high_shelf", "freq_hz": 6000.0, "gain_db": 5.0}
    assert parameter_error(dict(params), dict(params)) == pytest.approx(0.0)


def test_parameter_error_none_for_mismatched_effect() -> None:
    predicted = {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7}
    target = {"effect": "highpass", "freq_hz": 1000.0, "q": 0.7}
    assert parameter_error(predicted, target) is None


def test_parameter_error_none_for_invalid_predicted() -> None:
    predicted = {"effect": "lowpass", "freq_hz": 1000.0}  # missing "q"
    target = {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7}
    assert parameter_error(predicted, target) is None


# --- evaluate_predictions -----------------------------------------------------


def test_evaluate_predictions_aggregates_ratios_and_error() -> None:
    targets = [
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
    ]
    predictions = [
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},  # valid, matches, zero error
        {"effect": "highpass", "freq_hz": 1000.0, "q": 0.7},  # valid, effect mismatch
        {"effect": "lowpass", "freq_hz": 1000.0},  # invalid (missing q)
    ]

    result = evaluate_predictions(predictions, targets)

    assert result["valid_output_ratio"] == pytest.approx(2 / 3)
    # effect_match_ratio compares the "effect" string regardless of schema
    # validity: predictions[0] and predictions[2] both say "lowpass" (matching
    # their targets), even though predictions[2] is schema-invalid overall.
    assert result["effect_match_ratio"] == pytest.approx(2 / 3)
    assert result["mean_parameter_error"] == pytest.approx(0.0)
    assert result["num_comparable"] == 1


def test_evaluate_predictions_mean_error_none_when_nothing_comparable() -> None:
    targets = [{"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7}]
    predictions = [{"effect": "highpass", "freq_hz": 1000.0, "q": 0.7}]
    result = evaluate_predictions(predictions, targets)
    assert result["mean_parameter_error"] is None
    assert result["num_comparable"] == 0


def test_evaluate_predictions_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions([{"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7}], [])


def test_evaluate_predictions_rejects_empty() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions([], [])


# --- repetition_stability -----------------------------------------------------


def test_repetition_stability_full_agreement_has_zero_spread() -> None:
    predictions = [{"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7}] * 3
    result = repetition_stability(predictions)
    assert result["effect_agreement_ratio"] == pytest.approx(1.0)
    assert result["mean_param_spread"] == pytest.approx(0.0)


def test_repetition_stability_partial_agreement() -> None:
    predictions = [
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
        {"effect": "lowpass", "freq_hz": 1100.0, "q": 0.6},
        {"effect": "highpass", "freq_hz": 1000.0, "q": 0.7},
    ]
    result = repetition_stability(predictions)
    assert result["effect_agreement_ratio"] == pytest.approx(2 / 3)
    assert result["mean_param_spread"] is not None
    assert result["mean_param_spread"] > 0.0


def test_repetition_stability_none_spread_with_fewer_than_two_matches() -> None:
    predictions = [
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
        {"effect": "highpass", "freq_hz": 1000.0, "q": 0.7},
        {"effect": "peaking", "freq_hz": 1000.0, "gain_db": 1.0, "q": 0.7},
    ]
    result = repetition_stability(predictions)
    assert result["mean_param_spread"] is None


def test_repetition_stability_rejects_empty() -> None:
    with pytest.raises(ValueError):
        repetition_stability([])


def test_repetition_stability_all_invalid_is_zero_agreement_not_one() -> None:
    # Repeated unparseable/empty outputs must NOT count as "agreeing" with each
    # other (they'd otherwise all share effect=None and look perfectly stable,
    # rewarding repeated failure instead of penalizing it).
    predictions = [{}, {}, {}]
    result = repetition_stability(predictions)
    assert result["effect_agreement_ratio"] == pytest.approx(0.0)
    assert result["mean_param_spread"] is None


def test_repetition_stability_invalid_outputs_lower_agreement() -> None:
    # Two real, agreeing predictions plus one unparseable (empty) output: the
    # empty one must count against agreement, not toward it.
    predictions = [
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
        {"effect": "lowpass", "freq_hz": 1000.0, "q": 0.7},
        {},
    ]
    result = repetition_stability(predictions)
    assert result["effect_agreement_ratio"] == pytest.approx(2 / 3)


# --- acoustic_feature_change --------------------------------------------------


def test_acoustic_feature_change_reports_before_after_delta() -> None:
    quiet = generate_sine(1000.0, 1.0, SAMPLE_RATE, amplitude=0.2)
    loud = generate_sine(1000.0, 1.0, SAMPLE_RATE, amplitude=0.8)

    result = acoustic_feature_change(quiet, loud, SAMPLE_RATE)

    assert result["rms_before"] < result["rms_after"]
    assert result["rms_delta"] == pytest.approx(result["rms_after"] - result["rms_before"])
    assert result["spectral_centroid_hz_before"] == pytest.approx(1000.0, abs=5.0)
    assert result["spectral_centroid_hz_after"] == pytest.approx(1000.0, abs=5.0)


def test_acoustic_feature_change_zero_delta_for_identical_signal() -> None:
    waveform = generate_sine(1000.0, 1.0, SAMPLE_RATE, amplitude=0.5)
    result = acoustic_feature_change(waveform, waveform, SAMPLE_RATE)
    assert result["rms_delta"] == pytest.approx(0.0)
    assert result["spectral_centroid_hz_delta"] == pytest.approx(0.0)
    assert result["spectral_rolloff_hz_delta"] == pytest.approx(0.0)
