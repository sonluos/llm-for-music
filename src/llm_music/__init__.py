"""DSP foundation package for LLM-driven music production."""

from llm_music.audio_io import load_audio, save_audio
from llm_music.convolution import convolve, frequency_response, moving_average_kernel
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
]
