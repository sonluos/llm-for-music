"""Tests for the baseline model: instruction -> JSON effect parameters.

Most tests inject a fake `generate_fn` so they run fast with no model
download/inference. A small real-model integration test at the bottom proves
the actual LLM call (generate_raw_response / the default predict_effect_params
path) produces parseable JSON end to end.
"""

import pytest

from llm_music.baseline_model import (
    extract_json,
    generate_raw_response,
    predict_effect_params,
    validate_effect_params,
)

# --- extract_json ---------------------------------------------------------


def test_extract_json_parses_clean_object() -> None:
    assert extract_json('{"effect": "lowpass", "freq_hz": 1000, "q": 0.7}') == {
        "effect": "lowpass",
        "freq_hz": 1000,
        "q": 0.7,
    }


def test_extract_json_parses_object_wrapped_in_code_fence() -> None:
    text = '```json\n{"effect": "reverb", "room_size": 0.5, "wet_dry": 0.3}\n```'
    assert extract_json(text) == {"effect": "reverb", "room_size": 0.5, "wet_dry": 0.3}


def test_extract_json_raises_on_no_json() -> None:
    with pytest.raises(ValueError):
        extract_json("I'm not sure what effect to apply.")


def test_extract_json_raises_on_invalid_json_syntax() -> None:
    with pytest.raises(ValueError):
        extract_json('{"effect": "lowpass", "freq_hz": }')


def test_extract_json_takes_only_the_first_object_when_multiple_present() -> None:
    text = 'Sure: {"effect": "lowpass", "freq_hz": 1000, "q": 0.7} and maybe {"other": true}'
    assert extract_json(text) == {"effect": "lowpass", "freq_hz": 1000, "q": 0.7}


def test_extract_json_handles_nested_braces() -> None:
    text = '{"effect": "lowpass", "meta": {"nested": 1}, "freq_hz": 1000, "q": 0.7}'
    parsed = extract_json(text)
    assert parsed["meta"] == {"nested": 1}


# --- validate_effect_params -------------------------------------------------


def test_validate_effect_params_accepts_valid_lowpass() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": 1000, "q": 0.7}) is True


def test_validate_effect_params_accepts_valid_reverb() -> None:
    assert validate_effect_params({"effect": "reverb", "room_size": 0.5, "wet_dry": 0.3}) is True


def test_validate_effect_params_rejects_unknown_effect() -> None:
    assert validate_effect_params({"effect": "flanger", "depth": 0.5}) is False


def test_validate_effect_params_rejects_missing_field() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": 1000}) is False


def test_validate_effect_params_rejects_non_numeric_field() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": "high", "q": 0.7}) is False


def test_validate_effect_params_rejects_extra_field() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": 1000, "q": 0.7, "extra": 1}) is False


def test_validate_effect_params_rejects_negative_freq() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": -1000, "q": 0.7}) is False


def test_validate_effect_params_rejects_zero_q() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": 1000, "q": 0}) is False


def test_validate_effect_params_rejects_out_of_range_reverb() -> None:
    assert validate_effect_params({"effect": "reverb", "room_size": 1.5, "wet_dry": 0.3}) is False


def test_validate_effect_params_rejects_nan_field() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": float("nan"), "q": 0.7}) is False


def test_validate_effect_params_rejects_bool_field() -> None:
    assert validate_effect_params({"effect": "lowpass", "freq_hz": 1000, "q": True}) is False


def test_validate_effect_params_rejects_non_dict() -> None:
    assert validate_effect_params(["lowpass", 1000]) is False
    assert validate_effect_params(None) is False


# --- predict_effect_params (fake generate_fn, no model load) ---------------


def test_predict_effect_params_uses_injected_generate_fn() -> None:
    calls = []

    def fake_generate(instruction: str) -> str:
        calls.append(instruction)
        return '{"effect": "high_shelf", "freq_hz": 6000, "gain_db": 5}'

    result = predict_effect_params("보컬을 더 선명하게 해줘", generate_fn=fake_generate)

    assert result == {"effect": "high_shelf", "freq_hz": 6000, "gain_db": 5}
    assert calls == ["보컬을 더 선명하게 해줘"]


def test_predict_effect_params_rejects_empty_instruction() -> None:
    with pytest.raises(ValueError):
        predict_effect_params("   ", generate_fn=lambda instr: "{}")


def test_predict_effect_params_propagates_parse_failure() -> None:
    with pytest.raises(ValueError):
        predict_effect_params("make it louder", generate_fn=lambda instr: "no json here")


# --- real model integration (loads the actual LLM; slower, needs network/model) ---
# Deselected by default (see pytest.ini); run explicitly with `pytest -m integration`.


@pytest.mark.integration
def test_generate_raw_response_real_model_produces_parseable_json() -> None:
    raw = generate_raw_response("make the vocals brighter", max_new_tokens=80)
    parsed = extract_json(raw)
    assert isinstance(parsed, dict)
    assert "effect" in parsed


@pytest.mark.integration
def test_predict_effect_params_real_model_end_to_end() -> None:
    result = predict_effect_params("make the bass louder")
    assert isinstance(result, dict)
    assert "effect" in result
