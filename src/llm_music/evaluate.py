"""Evaluation metrics for the baseline vs. proposed model comparison.

Implements the four metrics the project proposal names for this comparison:
parameter error, valid output ratio, repetition stability, and acoustic
feature change. Pure metric functions only — no model calls here; callers
(e.g. scripts/demo_audio.py) generate predictions with
llm_music.baseline_model / llm_music.proposed_model and pass the results in.
"""

from collections import Counter
from typing import Any, Dict, Optional, Sequence

import torch

from llm_music.baseline_model import EFFECT_SCHEMAS, FIELD_RANGES, validate_effect_params
from llm_music.features import compute_rms, spectral_centroid, spectral_rolloff


def parameter_error(predicted: Any, target: Dict[str, Any]) -> Optional[float]:
    """Compute a normalized mean absolute error between predicted and target params.

    Each shared numeric field's absolute error is divided by that field's
    FIELD_RANGES span (so freq_hz and gain_db, with very different absolute
    scales, contribute comparably) before averaging.

    Args:
        predicted: A model's output, as from predict_effect_params[_with_audio].
        target: The ground-truth target_params dict (e.g. from an
            llm_music.dataset example); assumed well-formed.

    Returns:
        The mean normalized per-field error, or None if `predicted` isn't a
        schema-valid dict for the same "effect" as `target` (making a
        per-field comparison meaningless).
    """
    if not validate_effect_params(predicted):
        return None
    if predicted.get("effect") != target.get("effect"):
        return None

    effect = predicted["effect"]
    errors = []
    for key in EFFECT_SCHEMAS[effect]:
        low, high = FIELD_RANGES[key]
        errors.append(abs(predicted[key] - target[key]) / (high - low))
    return sum(errors) / len(errors)


def evaluate_predictions(
    predictions: Sequence[Any], targets: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate evaluation metrics over a batch of (predicted, target) pairs.

    Args:
        predictions: One model output per example (as from
            predict_effect_params[_with_audio]); any of these may be
            malformed or schema-invalid.
        targets: The matching ground-truth target_params dicts, same length
            and order as `predictions`.

    Returns:
        Dict with:
            "valid_output_ratio": fraction of predictions that are
                schema-valid (llm_music.baseline_model.validate_effect_params).
            "effect_match_ratio": fraction whose "effect" matches the
                target's "effect", regardless of schema validity.
            "mean_parameter_error": mean of parameter_error() over examples
                where it isn't None (valid output with a matching effect
                type); None if there are no such examples.
            "num_comparable": how many examples contributed to
                mean_parameter_error.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have the same length, got {len(predictions)} "
            f"and {len(targets)}"
        )
    if not predictions:
        raise ValueError("predictions and targets must not be empty")

    valid_count = sum(1 for p in predictions if validate_effect_params(p))
    effect_match_count = sum(
        1
        for p, t in zip(predictions, targets)
        if isinstance(p, dict) and p.get("effect") == t.get("effect")
    )

    errors = []
    for p, t in zip(predictions, targets):
        error = parameter_error(p, t)
        if error is not None:
            errors.append(error)

    return {
        "valid_output_ratio": valid_count / len(predictions),
        "effect_match_ratio": effect_match_count / len(predictions),
        "mean_parameter_error": (sum(errors) / len(errors)) if errors else None,
        "num_comparable": len(errors),
    }


def repetition_stability(predictions: Sequence[Any]) -> Dict[str, Any]:
    """Measure how consistent repeated predictions for the same input are.

    Meant for a list of outputs from calling the same model on the same
    input multiple times with sampling enabled (see
    llm_music.baseline_model.generate_with_model's `temperature` argument) —
    with greedy decoding, repeated calls are identical by construction, so
    this metric is only informative when some randomness is introduced.

    Args:
        predictions: Repeated outputs for one fixed input.

    Returns:
        Dict with:
            "effect_agreement_ratio": fraction of predictions whose "effect"
                matches the most common ("majority") effect among them.
                Predictions with no effect at all (not a dict, or missing/
                None "effect" — e.g. unparseable model output) never count
                as agreeing with each other: that would reward repeated
                failures as "stability" instead of penalizing them. 0.0 if
                every prediction is like this (no named effect anywhere).
            "mean_param_spread": among predictions that both match the
                majority effect and are schema-valid, the average normalized
                standard deviation (via FIELD_RANGES) of each required
                numeric field. None if fewer than 2 such predictions exist.
    """
    if not predictions:
        raise ValueError("predictions must not be empty")

    effects = [p.get("effect") if isinstance(p, dict) else None for p in predictions]
    named_effects = [e for e in effects if e is not None]
    if not named_effects:
        return {"effect_agreement_ratio": 0.0, "mean_param_spread": None}

    majority_effect = Counter(named_effects).most_common(1)[0][0]
    agreement_ratio = sum(1 for e in effects if e == majority_effect) / len(effects)

    matching_valid = [
        p
        for p in predictions
        if isinstance(p, dict) and p.get("effect") == majority_effect and validate_effect_params(p)
    ]

    mean_param_spread = None
    if len(matching_valid) >= 2 and majority_effect in EFFECT_SCHEMAS:
        spreads = []
        for key in EFFECT_SCHEMAS[majority_effect]:
            values = torch.tensor([p[key] for p in matching_valid], dtype=torch.float32)
            low, high = FIELD_RANGES[key]
            spreads.append((values.std().item()) / (high - low))
        mean_param_spread = sum(spreads) / len(spreads)

    return {"effect_agreement_ratio": agreement_ratio, "mean_param_spread": mean_param_spread}


def acoustic_feature_change(
    original_waveform: torch.Tensor, processed_waveform: torch.Tensor, sample_rate: int
) -> Dict[str, float]:
    """Compare RMS, spectral centroid, and spectral rolloff before vs. after processing.

    Args:
        original_waveform: 1D tensor, the pre-effect audio.
        processed_waveform: 1D tensor, the post-effect audio (any length;
            features are computed independently on each).
        sample_rate: Sample rate in Hz.

    Returns:
        Dict with "<feature>_before", "<feature>_after", and "<feature>_delta"
        (after - before) for "rms", "spectral_centroid_hz", and
        "spectral_rolloff_hz".
    """
    feature_fns = {
        "rms": lambda w: compute_rms(w).item(),
        "spectral_centroid_hz": lambda w: spectral_centroid(w, sample_rate),
        "spectral_rolloff_hz": lambda w: spectral_rolloff(w, sample_rate),
    }

    result: Dict[str, float] = {}
    for name, fn in feature_fns.items():
        before = fn(original_waveform)
        after = fn(processed_waveform)
        result[f"{name}_before"] = before
        result[f"{name}_after"] = after
        result[f"{name}_delta"] = after - before
    return result
