"""Link (instruction, audio features, target effect parameters) into training examples.

Each record pairs a natural-language music-production instruction with the
source audio's normalized features (llm_music.llm_input) and the effect
parameters that instruction should produce — the supervised training/
evaluation unit for the eventual instruction+audio -> EQ/Reverb-parameters
model. No model or real labeled-data collection is implemented here; this
only builds the data structure and JSON Lines I/O for it.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch

from llm_music.llm_input import extract_feature_vector, normalize_features

_REQUIRED_KEYS = {"instruction": str, "audio_features": dict, "target_params": dict}


def _validate_example(example: Any, context: str = "") -> None:
    """Raise ValueError if `example` is not a well-formed training example dict."""
    if not isinstance(example, dict):
        raise ValueError(f"{context}example must be a dict, got {type(example).__name__}")
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in example:
            raise ValueError(f"{context}example missing required key '{key}'")
        if not isinstance(example[key], expected_type):
            raise ValueError(
                f"{context}example['{key}'] must be a {expected_type.__name__}, "
                f"got {type(example[key]).__name__}"
            )
    if not example["instruction"].strip():
        raise ValueError(f"{context}example['instruction'] must not be empty")


def build_example(
    instruction: str,
    waveform: torch.Tensor,
    sample_rate: int,
    target_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one training example linking an instruction, audio features, and target params.

    Args:
        instruction: Natural-language music production command, e.g.
            "보컬을 더 선명하게 해줘" ("make the vocals clearer").
        waveform: 1D tensor of the source (pre-effect) audio, shape (samples,).
        sample_rate: Sample rate in Hz.
        target_params: The effect parameters the instruction should produce,
            e.g. {"effect": "high_shelf", "freq_hz": 5000.0, "gain_db": 6.0}.
            Stored as-is (not validated against any effect schema), but must
            be JSON-serializable (str/int/float/bool/None/list/dict, no NaN/
            Infinity, no tensors) — checked immediately, not deferred to
            save_dataset.

    Returns:
        Dict with keys "instruction", "audio_features" (the normalized
        feature dict from llm_music.llm_input.extract_feature_vector +
        normalize_features), and "target_params".
    """
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    if not isinstance(target_params, dict):
        raise ValueError(f"target_params must be a dict, got {type(target_params).__name__}")
    try:
        json.dumps(target_params, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"target_params must be JSON-serializable: {e}") from e

    features = extract_feature_vector(waveform, sample_rate)
    normalized = normalize_features(features, sample_rate)
    return {
        "instruction": instruction,
        "audio_features": normalized,
        "target_params": target_params,
    }


def save_dataset(examples: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save examples as JSON Lines (one JSON object per line), creating parent dirs.

    Every example is validated and serialized in memory first, and the result
    is written to a temporary file that is then atomically moved into place.
    This means a malformed example or non-finite value (NaN/Infinity, which
    `json` would otherwise render as non-standard tokens) raises ValueError
    before any existing file at `path` is touched, rather than leaving it
    truncated or partially overwritten.

    Args:
        examples: Records to save, each matching the shape build_example
            returns (a dict with "instruction": str, "audio_features": dict,
            "target_params": dict).
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, example in enumerate(examples):
        _validate_example(example, context=f"examples[{i}]: ")
        lines.append(json.dumps(example, ensure_ascii=False, allow_nan=False))

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load examples from a JSON Lines file written by save_dataset.

    Args:
        path: Input file path.

    Returns:
        List of example dicts, in file order. Blank lines are skipped.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If any non-blank line is not a well-formed example (not a
            JSON object, or missing/mistyped "instruction"/"audio_features"/
            "target_params").
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            _validate_example(example, context=f"line {line_number}: ")
            examples.append(example)
    return examples
