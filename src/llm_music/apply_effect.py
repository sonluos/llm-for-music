"""Dispatch a predicted effect dict (baseline/proposed model output) to real DSP.

Connects llm_music.baseline_model's structured output schema to the actual
signal-processing functions that implement each effect (llm_music.eq's biquad
filters, llm_music.reverb's reverberator), so a model's JSON prediction can be
applied to real audio — closing the loop from natural-language instruction to
processed sound.
"""

from typing import Any, Dict

import torch

from llm_music.baseline_model import validate_effect_params
from llm_music.eq import apply_biquad, high_shelf, highpass, low_shelf, lowpass, peaking
from llm_music.reverb import apply_reverb


def apply_predicted_effect(
    waveform: torch.Tensor, sample_rate: int, params: Dict[str, Any]
) -> torch.Tensor:
    """Apply a predicted effect dict to a waveform.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        params: A schema-valid effect dict, as from
            llm_music.baseline_model.predict_effect_params or
            llm_music.proposed_model.predict_effect_params_with_audio (see
            llm_music.baseline_model.validate_effect_params for the schema).
            Its "effect" field selects which DSP function runs.

    Returns:
        1D tensor of shape (samples,), the processed signal.

    Raises:
        ValueError: If params is not schema-valid (see validate_effect_params)
            — a model's invalid/hallucinated output should be checked for
            before calling this, not passed straight through.
    """
    if not validate_effect_params(params):
        raise ValueError(f"params is not a schema-valid effect spec: {params!r}")

    effect = params["effect"]
    if effect == "lowpass":
        b, a = lowpass(params["freq_hz"], params["q"], sample_rate)
        return apply_biquad(waveform, b, a)
    if effect == "highpass":
        b, a = highpass(params["freq_hz"], params["q"], sample_rate)
        return apply_biquad(waveform, b, a)
    if effect == "peaking":
        b, a = peaking(params["freq_hz"], params["gain_db"], params["q"], sample_rate)
        return apply_biquad(waveform, b, a)
    if effect == "low_shelf":
        b, a = low_shelf(params["freq_hz"], params["gain_db"], sample_rate)
        return apply_biquad(waveform, b, a)
    if effect == "high_shelf":
        b, a = high_shelf(params["freq_hz"], params["gain_db"], sample_rate)
        return apply_biquad(waveform, b, a)
    if effect == "reverb":
        return apply_reverb(
            waveform,
            sample_rate,
            params["room_size"],
            params["damping"],
            params["decay"],
            params["wet_dry"],
        )

    raise AssertionError(f"Unhandled effect in EFFECT_SCHEMAS: {effect!r}")  # pragma: no cover
