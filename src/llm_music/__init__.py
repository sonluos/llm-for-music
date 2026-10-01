"""DSP foundation package for LLM-driven music production."""

from llm_music.audio_io import load_audio, save_audio
from llm_music.convolution import convolve, frequency_response, moving_average_kernel
from llm_music.eq import apply_biquad, high_shelf, highpass, low_shelf, lowpass, peaking
from llm_music.preprocess import (
    normalize_peak,
    normalize_rms,
    preprocess_batch,
    resample,
    segment,
    to_mono,
)
from llm_music.signals import generate_multitone, generate_sine
from llm_music.spectrum import compute_fft, magnitude_to_db

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
    "resample",
    "to_mono",
    "normalize_peak",
    "normalize_rms",
    "segment",
    "preprocess_batch",
]
