"""DSP foundation package for LLM-driven music production."""

from llm_music.audio_io import load_audio, save_audio
from llm_music.convolution import convolve, frequency_response, moving_average_kernel
from llm_music.eq import (
    apply_biquad,
    apply_eq_chain,
    high_shelf,
    highpass,
    low_shelf,
    lowpass,
    peaking,
)
from llm_music.features import band_energy, compute_rms, spectral_centroid, spectral_rolloff
from llm_music.pipeline import PipelineResult, run_pipeline
from llm_music.preprocess import (
    normalize_peak,
    normalize_rms,
    preprocess_batch,
    resample,
    segment,
    to_mono,
)
from llm_music.quantize import quantize_bit_depth, signal_to_noise_ratio, theoretical_sqnr_db
from llm_music.signals import generate_multitone, generate_sine
from llm_music.spectrum import compute_fft, magnitude_to_db
from llm_music.stft import compute_stft

__all__ = [
    "load_audio",
    "save_audio",
    "generate_sine",
    "generate_multitone",
    "compute_fft",
    "magnitude_to_db",
    "convolve",
    "moving_average_kernel",
    "frequency_response",
    "lowpass",
    "highpass",
    "peaking",
    "low_shelf",
    "high_shelf",
    "apply_biquad",
    "apply_eq_chain",
    "resample",
    "to_mono",
    "normalize_peak",
    "normalize_rms",
    "segment",
    "preprocess_batch",
    "quantize_bit_depth",
    "signal_to_noise_ratio",
    "theoretical_sqnr_db",
    "run_pipeline",
    "PipelineResult",
    "compute_stft",
    "compute_rms",
    "spectral_centroid",
    "spectral_rolloff",
    "band_energy",
]
