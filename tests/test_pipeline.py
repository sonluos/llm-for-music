"""Tests for the integrated preprocessing -> EQ chain -> RMS readout pipeline."""

import torch

from llm_music.audio_io import save_audio
from llm_music.eq import apply_biquad, apply_eq_chain, highpass, lowpass
from llm_music.pipeline import run_pipeline
from llm_music.preprocess import segment
from llm_music.signals import generate_sine

SAMPLE_RATE = 44100


def test_apply_eq_chain_matches_single_filter() -> None:
    signal = torch.randn(2000)
    b, a = lowpass(1000.0, 0.7071, SAMPLE_RATE)
    chained = apply_eq_chain(signal, [(b, a)])
    direct = apply_biquad(signal, b, a)
    assert torch.equal(chained, direct)


def test_apply_eq_chain_empty_is_passthrough() -> None:
    signal = torch.randn(500)
    assert torch.equal(apply_eq_chain(signal, []), signal)


def test_apply_eq_chain_applies_filters_in_order() -> None:
    signal = torch.randn(2000)
    low_b, low_a = lowpass(1000.0, 0.7071, SAMPLE_RATE)
    high_b, high_a = highpass(1000.0, 0.7071, SAMPLE_RATE)
    chained = apply_eq_chain(signal, [(low_b, low_a), (high_b, high_a)])
    expected = apply_biquad(apply_biquad(signal, low_b, low_a), high_b, high_a)
    assert torch.equal(chained, expected)


def test_run_pipeline_output_shapes(tmp_path) -> None:
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    save_audio(path_a, generate_sine(300.0, 1.0, SAMPLE_RATE), SAMPLE_RATE)
    save_audio(path_b, generate_sine(600.0, 0.5, SAMPLE_RATE), SAMPLE_RATE)

    segment_length = SAMPLE_RATE // 4
    eq_chain = [lowpass(1000.0, 0.7071, SAMPLE_RATE)]
    result = run_pipeline([path_a, path_b], SAMPLE_RATE, segment_length, eq_chain)

    assert result.segments.shape == result.processed_segments.shape
    assert result.segments.shape[-1] == segment_length
    num_segments = result.segments.shape[0]
    assert result.rms_before.shape == (num_segments,)
    assert result.rms_after.shape == (num_segments,)


def test_run_pipeline_empty_chain_leaves_rms_unchanged(tmp_path) -> None:
    path = tmp_path / "tone.wav"
    save_audio(path, generate_sine(440.0, 1.0, SAMPLE_RATE), SAMPLE_RATE)

    result = run_pipeline([path], SAMPLE_RATE, segment_length=SAMPLE_RATE // 2, eq_chain=[])
    assert torch.allclose(result.rms_before, result.rms_after)


def test_run_pipeline_lowpass_reduces_rms_of_high_frequency_tone(tmp_path) -> None:
    path = tmp_path / "high_tone.wav"
    # 10 kHz is far above the 500 Hz low-pass cutoff below; its RMS should drop sharply.
    save_audio(path, generate_sine(10000.0, 1.0, SAMPLE_RATE, amplitude=0.8), SAMPLE_RATE)

    eq_chain = [lowpass(500.0, 0.7071, SAMPLE_RATE)]
    result = run_pipeline([path], SAMPLE_RATE, segment_length=SAMPLE_RATE // 2, eq_chain=eq_chain)

    assert torch.all(result.rms_after < 0.1 * result.rms_before)


def test_run_pipeline_filters_continuously_across_segment_boundaries(tmp_path) -> None:
    # EQ must run on the full waveform before segmenting, not per-segment:
    # filtering each segment independently would reset the IIR filter's state
    # at every boundary, producing a transient that continuous filtering would
    # not have. Reproduce the reference (continuous) result by hand and compare.
    path = tmp_path / "tone.wav"
    waveform = generate_sine(300.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    save_audio(path, waveform, SAMPLE_RATE)

    segment_length = SAMPLE_RATE // 4
    eq_chain = [lowpass(1000.0, 0.7071, SAMPLE_RATE)]
    # target_peak=None: skip normalization so the pipeline's internal waveform
    # matches `waveform` below exactly, keeping the comparison simple.
    result = run_pipeline([path], SAMPLE_RATE, segment_length, eq_chain, target_peak=None)

    continuously_filtered = apply_eq_chain(waveform, eq_chain)
    expected_segments = segment(continuously_filtered, segment_length)

    assert torch.allclose(result.processed_segments, expected_segments, atol=1e-4)

    # If each segment were filtered independently (resetting the IIR filter's
    # state at every boundary), the first sample of every non-initial segment
    # would instead be near zero; demonstrate the fix actually changes that.
    reset_per_segment = torch.stack(
        [apply_eq_chain(s, eq_chain) for s in segment(waveform, segment_length)]
    )
    boundary_outputs = result.processed_segments[1:, 0]
    reset_boundary_outputs = reset_per_segment[1:, 0]
    assert torch.all((boundary_outputs - reset_boundary_outputs).abs() > 0.1)
