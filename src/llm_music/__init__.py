"""DSP foundation package for LLM-driven music production."""

from llm_music.apply_effect import apply_predicted_effect
from llm_music.audio_io import load_audio, save_audio
from llm_music.baseline_model import (
    disable_sampling_generation_defaults,
    extract_json,
    generate_from_user_content,
    generate_raw_response,
    generate_with_model,
    get_model_and_tokenizer,
    predict_effect_params,
    validate_effect_params,
)
from llm_music.convolution import convolve, frequency_response, moving_average_kernel
from llm_music.dataset import build_example, load_dataset, save_dataset
from llm_music.eq import (
    apply_biquad,
    apply_eq_chain,
    high_shelf,
    highpass,
    low_shelf,
    lowpass,
    peaking,
)
from llm_music.evaluate import (
    acoustic_feature_change,
    evaluate_predictions,
    parameter_error,
    repetition_stability,
)
from llm_music.features import band_energy, compute_rms, spectral_centroid, spectral_rolloff
from llm_music.llm_input import (
    audio_to_llm_input,
    extract_feature_vector,
    format_for_llm_input,
    normalize_features,
)
from llm_music.pipeline import PipelineResult, run_pipeline
from llm_music.preprocess import (
    normalize_peak,
    normalize_rms,
    preprocess_batch,
    resample,
    segment,
    to_mono,
)
from llm_music.proposed_model import build_user_content, predict_effect_params_with_audio
from llm_music.quantize import quantize_bit_depth, signal_to_noise_ratio, theoretical_sqnr_db
from llm_music.reverb import apply_reverb
from llm_music.signals import generate_multitone, generate_sine
from llm_music.spectrum import compute_fft, magnitude_to_db
from llm_music.stft import compute_stft
from llm_music.train_lora import (
    build_lora_model,
    build_optimizer,
    load_lora_adapter,
    save_lora_adapter,
    train_one_epoch,
)

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
    "extract_feature_vector",
    "normalize_features",
    "format_for_llm_input",
    "audio_to_llm_input",
    "build_example",
    "save_dataset",
    "load_dataset",
    "predict_effect_params",
    "validate_effect_params",
    "generate_raw_response",
    "generate_from_user_content",
    "generate_with_model",
    "get_model_and_tokenizer",
    "disable_sampling_generation_defaults",
    "extract_json",
    "build_user_content",
    "predict_effect_params_with_audio",
    "build_lora_model",
    "build_optimizer",
    "train_one_epoch",
    "save_lora_adapter",
    "load_lora_adapter",
    "apply_reverb",
    "apply_predicted_effect",
    "parameter_error",
    "evaluate_predictions",
    "repetition_stability",
    "acoustic_feature_change",
]
