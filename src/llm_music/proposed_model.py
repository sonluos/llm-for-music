"""Proposed model: instruction + audio features -> EQ/Reverb parameters.

Extends llm_music.baseline_model's text-only baseline by also embedding the
source audio's normalized features (llm_music.llm_input) into the prompt, so
the same underlying LLM can ground its prediction in what the audio actually
sounds like, not just the instruction's wording. This is the audio-conditioned
side of the project's planned comparison against
llm_music.baseline_model.predict_effect_params (text-only).

Fusion here is at the prompt (text) level — audio features are serialized as
JSON into the user message — rather than a learned embedding-space
projection. That is a deliberate scope choice: it reuses the same model and
generation code as the baseline with no change to its internals, and is far
simpler to implement, train (see llm_music.train_lora), and debug than
injecting projected feature embeddings into the transformer's input sequence.
"""

from typing import Any, Callable, Dict, Optional

import torch

from llm_music.baseline_model import extract_json, generate_from_user_content
from llm_music.llm_input import extract_feature_vector, format_for_llm_input, normalize_features


def build_user_content(instruction: str, waveform: torch.Tensor, sample_rate: int) -> str:
    """Build the user message combining the source audio's features and the instruction.

    Args:
        instruction: Natural-language music production command.
        waveform: 1D tensor of the source (pre-effect) audio, shape (samples,).
        sample_rate: Sample rate in Hz.

    Returns:
        A single string: the audio's normalized features as JSON, followed by
        the instruction, for use as the LLM's user message.
    """
    features = extract_feature_vector(waveform, sample_rate)
    normalized = normalize_features(features, sample_rate)
    features_json = format_for_llm_input(normalized)
    return f"Audio features: {features_json}\nInstruction: {instruction}"


def predict_effect_params_with_audio(
    instruction: str,
    waveform: torch.Tensor,
    sample_rate: int,
    generate_fn: Optional[Callable[[str], str]] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Predict an effect type and parameters from an instruction AND the source audio.

    Args:
        instruction: Natural-language music production command.
        waveform: 1D tensor of the source (pre-effect) audio, shape (samples,).
        sample_rate: Sample rate in Hz.
        generate_fn: Overrides the LLM call with a custom
            user_content -> raw text function. Defaults to
            generate_from_user_content with the real baseline model (or a
            LoRA-fine-tuned copy, if the caller's generate_fn wraps one);
            tests inject a fake here to avoid loading it.
        max_new_tokens: Passed to generate_from_user_content when generate_fn
            is not given.

    Returns:
        The parsed JSON value from the model's response (typically a dict
        with an "effect" key and that effect's parameters). Not guaranteed to
        be schema-valid — see llm_music.baseline_model.validate_effect_params.
    """
    if not instruction.strip():
        raise ValueError("instruction must not be empty")

    user_content = build_user_content(instruction, waveform, sample_rate)

    if generate_fn is None:
        generate_fn = lambda content: generate_from_user_content(content, max_new_tokens)  # noqa: E731

    raw_response = generate_fn(user_content)
    return extract_json(raw_response)
