"""Normalize extracted audio features and format them for an LLM prompt.

Builds on llm_music.features (RMS, spectral centroid, spectral rolloff, band
energy) to turn one waveform/segment into a compact, named feature dict with
every value in a comparable [0, 1] range, then serializes it to a JSON string
that can sit alongside a natural-language instruction in an LLM prompt. No LLM
call is made here; this only prepares that side of the eventual input.
"""

import json
from typing import Dict, Sequence, Tuple

import torch

from llm_music.features import band_energy, compute_rms, spectral_centroid, spectral_rolloff
from llm_music.spectrum import compute_fft

# A reasonable default split for sample rates of 44.1 kHz and up. For lower
# sample rates, bands above the Nyquist frequency simply contribute 0 energy
# (band_energy finds no bins there) rather than raising an error.
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = ((20.0, 250.0), (250.0, 4000.0), (4000.0, 20000.0))
DEFAULT_BAND_NAMES: Tuple[str, ...] = ("low_energy_frac", "mid_energy_frac", "high_energy_frac")

RESERVED_FEATURE_KEYS = frozenset({"rms", "spectral_centroid_hz", "spectral_rolloff_hz"})


def extract_feature_vector(
    waveform: torch.Tensor,
    sample_rate: int,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    band_names: Sequence[str] = DEFAULT_BAND_NAMES,
) -> Dict[str, float]:
    """Compute a named dict of raw audio features for one waveform/segment.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        bands: Frequency bands passed to llm_music.features.band_energy.
        band_names: Output key for each band, same length as `bands`, each
            unique and none equal to a reserved key ("rms",
            "spectral_centroid_hz", "spectral_rolloff_hz").

    Returns:
        Dict with keys "rms", "spectral_centroid_hz", "spectral_rolloff_hz",
        and one key per `band_names` holding that band's share of the
        waveform's *total* spectral energy (summed across all frequencies,
        not just the selected bands). These band fractions sum to 1 only if
        `bands` together cover every frequency where the waveform has energy;
        otherwise they sum to less than 1, correctly reflecting energy that
        falls outside the selected bands. All are 0 for a silent waveform.
    """
    if len(bands) != len(band_names):
        raise ValueError(
            f"bands and band_names must have the same length, got {len(bands)} and "
            f"{len(band_names)}"
        )
    if len(set(band_names)) != len(band_names):
        raise ValueError(f"band_names must be unique, got {list(band_names)}")
    conflicts = set(band_names) & RESERVED_FEATURE_KEYS
    if conflicts:
        raise ValueError(
            f"band_names must not reuse reserved feature keys {sorted(RESERVED_FEATURE_KEYS)}; "
            f"got conflicting name(s) {sorted(conflicts)}"
        )

    features: Dict[str, float] = {
        "rms": compute_rms(waveform).item(),
        "spectral_centroid_hz": spectral_centroid(waveform, sample_rate),
        "spectral_rolloff_hz": spectral_rolloff(waveform, sample_rate),
    }

    _, magnitude = compute_fft(waveform, sample_rate)
    total_energy = (magnitude**2).sum()
    energies = band_energy(waveform, sample_rate, bands)
    for name, energy in zip(band_names, energies):
        features[name] = (energy / total_energy).item() if total_energy > 0 else 0.0

    return features


def normalize_features(features: Dict[str, float], sample_rate: int) -> Dict[str, float]:
    """Scale a raw feature dict (from extract_feature_vector) to comparable [0, 1] values.

    Args:
        features: Raw feature dict, as returned by extract_feature_vector.
        sample_rate: Sample rate in Hz, used to convert Hz-valued features to
            a fraction of the Nyquist frequency.

    Returns:
        A new dict with the same keys. Any key ending in "_hz" is divided by
        the Nyquist frequency and clamped to [0, 1]; "rms" is clamped to
        [0, 1] (it is already in that range for audio with peak amplitude
        <= 1); every other key (e.g. the band-fraction features, already
        normalized by construction) is passed through unchanged.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    nyquist = sample_rate / 2
    normalized: Dict[str, float] = {}
    for key, value in features.items():
        if key.endswith("_hz"):
            normalized[key] = min(max(value / nyquist, 0.0), 1.0)
        elif key == "rms":
            normalized[key] = min(max(value, 0.0), 1.0)
        else:
            normalized[key] = value
    return normalized


def format_for_llm_input(normalized_features: Dict[str, float], precision: int = 4) -> str:
    """Serialize normalized audio features into a compact JSON string.

    Args:
        normalized_features: Feature dict with values in [0, 1], as returned
            by normalize_features.
        precision: Decimal digits to round each value to.

    Returns:
        A JSON string with sorted keys, e.g.
        '{"high_energy_frac": 0.0, "rms": 0.4014, ...}'.

    Raises:
        ValueError: If any value is NaN or infinite, which `json` would
            otherwise render as the non-standard tokens NaN/Infinity.
    """
    rounded = {key: round(value, precision) for key, value in normalized_features.items()}
    return json.dumps(rounded, sort_keys=True, allow_nan=False)


def audio_to_llm_input(
    waveform: torch.Tensor,
    sample_rate: int,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    band_names: Sequence[str] = DEFAULT_BAND_NAMES,
    precision: int = 4,
) -> str:
    """Extract, normalize, and format a waveform's audio features in one call.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        bands: Frequency bands passed to extract_feature_vector.
        band_names: Output key for each band, same length as `bands`.
        precision: Decimal digits to round each value to.

    Returns:
        A JSON string of normalized audio features, ready to embed in an LLM
        prompt alongside a natural-language instruction.
    """
    features = extract_feature_vector(waveform, sample_rate, bands, band_names)
    normalized = normalize_features(features, sample_rate)
    return format_for_llm_input(normalized, precision)
