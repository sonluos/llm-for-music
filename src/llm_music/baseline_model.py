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
# filter types, plus "reverb", which llm_music.reverb implements as a
# Schroeder/Freeverb-style parallel-comb + series-allpass reverberator.
EFFECT_SCHEMAS: Dict[str, frozenset] = {
    "lowpass": frozenset({"freq_hz", "q"}),
    "highpass": frozenset({"freq_hz", "q"}),
    "peaking": frozenset({"freq_hz", "gain_db", "q"}),
    "low_shelf": frozenset({"freq_hz", "gain_db"}),
    "high_shelf": frozenset({"freq_hz", "gain_db"}),
    "reverb": frozenset({"room_size", "damping", "decay", "wet_dry"}),
}

SYSTEM_PROMPT = """You are a music production assistant. Given a user's instruction, output exactly ONE JSON object describing a single audio effect to apply. Output ONLY the JSON object, with no other text, no markdown, no code fences.

The "effect" field must be exactly one of these strings (spelled exactly as shown, no underscore between words): "lowpass", "highpass", "peaking", "low_shelf", "high_shelf", "reverb".

Use exactly the listed fields for the chosen effect:
- {"effect": "lowpass", "freq_hz": <number>, "q": <number>}
- {"effect": "highpass", "freq_hz": <number>, "q": <number>}
- {"effect": "peaking", "freq_hz": <number>, "gain_db": <number>, "q": <number>}
- {"effect": "low_shelf", "freq_hz": <number>, "gain_db": <number>}
- {"effect": "high_shelf", "freq_hz": <number>, "gain_db": <number>}
- {"effect": "reverb", "room_size": <number 0-1>, "damping": <number 0-1>, "decay": <number 0-1>, "wet_dry": <number 0-1>}

Examples:
User: make the bass louder
Output: {"effect": "low_shelf", "freq_hz": 150, "gain_db": 5}
User: cut the harsh high frequencies
Output: {"effect": "lowpass", "freq_hz": 8000, "q": 0.7}
"""

_model = None
_tokenizer = None


def disable_sampling_generation_defaults(model) -> None:
    """Clear sampling-only generation-config fields so greedy decoding doesn't warn.

    Qwen2.5-Instruct's default generation_config sets temperature/top_p/top_k
    for its typical sampling use. This project always decodes greedily
    (do_sample=False), so those fields are irrelevant and otherwise trigger a
    "generation flags are not valid" warning on every .generate() call.

    Args:
        model: A freshly loaded causal LM (mutated in place).
    """
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False


def get_model_and_tokenizer():
    """Lazily load and cache the baseline LLM and its tokenizer.

    Exposed (not just internal) so callers that need the raw transformers
    objects directly — e.g. wrapping the model with a LoRA adapter for
    fine-tuning — can reuse the same cached instance instead of loading a
    second copy.
    """
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _model.eval()
        disable_sampling_generation_defaults(_model)
    return _model, _tokenizer


def generate_with_model(user_content: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    """Run a given model/tokenizer on a user message and return its raw text response.

    Lower-level than generate_raw_response: takes the model and tokenizer
    explicitly (rather than this module's cached singleton), so it also works
    with e.g. a LoRA-adapted copy of the model after fine-tuning.

    Args:
        user_content: The user message content (e.g. a bare instruction, or a
            richer message that also embeds audio features).
        model: A causal LM compatible with `.generate()`.
        tokenizer: The matching tokenizer, with a chat template.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's decoded response text (greedy decoding, deterministic).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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


def generate_from_user_content(user_content: str, max_new_tokens: int = 128) -> str:
    """Run the baseline LLM (this module's cached instance) on a user message.

    Args:
        user_content: The user message content.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's decoded response text.
    """
    model, tokenizer = get_model_and_tokenizer()
    return generate_with_model(user_content, model, tokenizer, max_new_tokens)


def generate_raw_response(instruction: str, max_new_tokens: int = 128) -> str:
    """Run the baseline LLM on a bare instruction and return its raw text response.

    Args:
        instruction: Natural-language music production command.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's decoded response text (greedy decoding, deterministic).
    """
    return generate_from_user_content(instruction, max_new_tokens)


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
# typical parametric-EQ ranges; the reverb fields to the 0-1 range the prompt
# itself specifies.
FIELD_RANGES: Dict[str, Tuple[float, float]] = {
    "freq_hz": (20.0, 20000.0),
    "q": (1e-6, 20.0),
    "gain_db": (-24.0, 24.0),
    "room_size": (0.0, 1.0),
    "damping": (0.0, 1.0),
    "decay": (0.0, 1.0),
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
