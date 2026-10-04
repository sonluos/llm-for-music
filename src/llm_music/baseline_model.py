"""Baseline model: natural-language instruction -> structured EQ/Reverb parameters.

Wraps a small open-source instruction-tuned LLM (loaded via `transformers`)
behind a system prompt constraining its output to a single JSON object naming
an effect and its parameters. This is the *baseline* in the project's planned
comparison: text-only input, no audio conditioning (an audio-conditioned
"proposed model" is a separate, later addition). The model is loaded lazily
and cached at module level, since loading takes a few seconds.
"""

import json
import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Required numeric fields for each supported effect. Mirrors llm_music.eq's
# filter types, plus "reverb" as a parameter spec only (no DSP implementation
# yet — that arrives with the reverb module).
EFFECT_SCHEMAS: Dict[str, frozenset] = {
    "lowpass": frozenset({"freq_hz", "q"}),
    "highpass": frozenset({"freq_hz", "q"}),
    "peaking": frozenset({"freq_hz", "gain_db", "q"}),
    "low_shelf": frozenset({"freq_hz", "gain_db"}),
    "high_shelf": frozenset({"freq_hz", "gain_db"}),
    "reverb": frozenset({"room_size", "wet_dry"}),
}

SYSTEM_PROMPT = """You are a music production assistant. Given a user's instruction, output exactly ONE JSON object describing a single audio effect to apply. Output ONLY the JSON object, with no other text, no markdown, no code fences.

The "effect" field must be exactly one of these strings (spelled exactly as shown, no underscore between words): "lowpass", "highpass", "peaking", "low_shelf", "high_shelf", "reverb".

Use exactly the listed fields for the chosen effect:
- {"effect": "lowpass", "freq_hz": <number>, "q": <number>}
- {"effect": "highpass", "freq_hz": <number>, "q": <number>}
- {"effect": "peaking", "freq_hz": <number>, "gain_db": <number>, "q": <number>}
- {"effect": "low_shelf", "freq_hz": <number>, "gain_db": <number>}
- {"effect": "high_shelf", "freq_hz": <number>, "gain_db": <number>}
- {"effect": "reverb", "room_size": <number 0-1>, "wet_dry": <number 0-1>}

Examples:
User: make the bass louder
Output: {"effect": "low_shelf", "freq_hz": 150, "gain_db": 5}
User: cut the harsh high frequencies
Output: {"effect": "lowpass", "freq_hz": 8000, "q": 0.7}
"""

_model = None
_tokenizer = None


def _load_model():
    """Lazily load and cache the baseline LLM and its tokenizer."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def generate_raw_response(instruction: str, max_new_tokens: int = 128) -> str:
    """Run the baseline LLM on an instruction and return its raw text response.

    Args:
        instruction: Natural-language music production command.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's decoded response text (greedy decoding, deterministic).
    """
    model, tokenizer = _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _find_first_json_object(text: str) -> str:
    """Find the span of the first complete, brace-balanced {...} object in text.

    Scans by brace depth so trailing text (including a second, separate JSON
    object) cannot extend the match the way a greedy `\\{.*\\}` regex would.
    Does not account for braces inside string literal values (e.g. a field
    value containing a literal "}"); not a concern for this module's flat,
    numeric-valued schemas.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"No complete (brace-balanced) JSON object found in model output: {text!r}")


def extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse the first {...} JSON object found in free-form text.

    Args:
        text: Text that should contain exactly one JSON object (e.g. the raw
            model response, possibly wrapped in markdown code fences).

    Returns:
        The parsed JSON value (typically a dict).

    Raises:
        ValueError: If no complete `{...}` substring is found, or it is not
            valid JSON.
    """
    json_text = _find_first_json_object(text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output contained invalid JSON: {e}") from e


# Inclusive (min, max) bounds for each field's value, in the units used by the
# prompt/schema. freq_hz is bounded to the audible range; q and gain_db to
# typical parametric-EQ ranges; room_size/wet_dry to the 0-1 range the prompt
# itself specifies.
FIELD_RANGES: Dict[str, Tuple[float, float]] = {
    "freq_hz": (20.0, 20000.0),
    "q": (1e-6, 20.0),
    "gain_db": (-24.0, 24.0),
    "room_size": (0.0, 1.0),
    "wet_dry": (0.0, 1.0),
}


def _is_valid_number(value: Any) -> bool:
    """True if value is a real (non-bool, finite) int/float."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_effect_params(params: Any) -> bool:
    """Check whether parsed params form a schema-conforming effect spec.

    Does not raise: intended for computing a "valid output ratio" over many
    predictions, so an invalid/hallucinated response is a normal, measurable
    outcome rather than an exception.

    Args:
        params: A value as returned by extract_json/predict_effect_params.

    Returns:
        True if `params` is a dict with exactly the keys "effect" plus that
        effect's required fields (no missing or extra keys), each field a
        finite, non-bool number within its range in FIELD_RANGES.
    """
    if not isinstance(params, dict):
        return False
    effect = params.get("effect")
    if effect not in EFFECT_SCHEMAS:
        return False

    required_keys = EFFECT_SCHEMAS[effect]
    if set(params.keys()) != required_keys | {"effect"}:
        return False

    for key in required_keys:
        value = params[key]
        if not _is_valid_number(value):
            return False
        low, high = FIELD_RANGES[key]
        if not low <= value <= high:
            return False
    return True


def predict_effect_params(
    instruction: str,
    generate_fn: Optional[Callable[[str], str]] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Predict an effect type and parameters from a natural-language instruction.

    This is the baseline model's full interface: text in, structured params
    out. The result is not guaranteed to be schema-valid (see
    validate_effect_params) — only that it was parseable as JSON.

    Args:
        instruction: Natural-language music production command.
        generate_fn: Overrides the LLM call with a custom
            instruction -> raw text function. Defaults to
            generate_raw_response with the real baseline model; tests inject
            a fake here to avoid loading it.
        max_new_tokens: Passed to generate_raw_response when generate_fn is
            not given.

    Returns:
        The parsed JSON value from the model's response (typically a dict
        with an "effect" key and that effect's parameters).
    """
    if not instruction.strip():
        raise ValueError("instruction must not be empty")

    if generate_fn is None:
        generate_fn = lambda instr: generate_raw_response(instr, max_new_tokens)  # noqa: E731

    raw_response = generate_fn(instruction)
    return extract_json(raw_response)
