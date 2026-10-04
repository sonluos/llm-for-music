"""Tests for the proposed (audio + text -> JSON effect parameters) model.

Most tests inject a fake generate_fn so they run fast with no model
download/inference. One real-model integration test at the bottom proves the
actual LLM call produces parseable JSON from a combined audio+instruction
prompt.
"""

import json

import pytest

from llm_music.proposed_model import build_user_content, predict_effect_params_with_audio
from llm_music.signals import generate_sine

SAMPLE_RATE = 44100


def test_build_user_content_includes_instruction() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    content = build_user_content("make it brighter", waveform, SAMPLE_RATE)
    assert content.endswith("Instruction: make it brighter")


def test_build_user_content_embeds_valid_feature_json() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    content = build_user_content("make it brighter", waveform, SAMPLE_RATE)

    # Everything between "Audio features: " and the newline before "Instruction:"
    features_line = content.split("Audio features: ")[1].split("\nInstruction:")[0]
    features = json.loads(features_line)

    assert "rms" in features
    assert "spectral_centroid_hz" in features
    for value in features.values():
        assert 0.0 <= value <= 1.0


def test_predict_effect_params_with_audio_uses_injected_generate_fn() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    calls = []

    def fake_generate(user_content: str) -> str:
        calls.append(user_content)
        return '{"effect": "lowpass", "freq_hz": 1000, "q": 0.7}'

    result = predict_effect_params_with_audio(
        "저음을 살려줘", waveform, SAMPLE_RATE, generate_fn=fake_generate
    )

    assert result == {"effect": "lowpass", "freq_hz": 1000, "q": 0.7}
    assert len(calls) == 1
    assert "저음을 살려줘" in calls[0]
    assert "Audio features:" in calls[0]


def test_predict_effect_params_with_audio_rejects_empty_instruction() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        predict_effect_params_with_audio("   ", waveform, SAMPLE_RATE, generate_fn=lambda c: "{}")


def test_predict_effect_params_with_audio_propagates_parse_failure() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        predict_effect_params_with_audio(
            "make it louder", waveform, SAMPLE_RATE, generate_fn=lambda c: "no json here"
        )


@pytest.mark.integration
def test_predict_effect_params_with_audio_real_model_end_to_end() -> None:
    waveform = generate_sine(8000.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    result = predict_effect_params_with_audio("make it brighter", waveform, SAMPLE_RATE)
    assert isinstance(result, dict)
    assert "effect" in result
