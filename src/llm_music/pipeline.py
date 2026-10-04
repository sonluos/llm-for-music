"""Integrated audio processing pipeline: preprocessing -> EQ chain -> level readout.

Ties together llm_music.preprocess (loading, resampling, mono downmix,
normalization, segmentation), llm_music.eq (parametric EQ filtering), and a
lightweight per-segment RMS readout into a single call, as one execution
pipeline rather than separate manual steps. A fuller feature-extraction module
(spectral centroid, rolloff, band energies) is a separate, later addition;
RMS here is just enough to confirm what the EQ chain did to signal level.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import torch

from llm_music.audio_io import load_audio
from llm_music.eq import apply_eq_chain
from llm_music.preprocess import normalize_peak, segment, to_mono


class PipelineResult(NamedTuple):
    """Output of run_pipeline."""

    segments: torch.Tensor  # (num_segments, segment_length), preprocessed, before EQ
    processed_segments: torch.Tensor  # (num_segments, segment_length), after EQ
    rms_before: torch.Tensor  # (num_segments,)
    rms_after: torch.Tensor  # (num_segments,)


def run_pipeline(
    paths: Sequence[Union[str, Path]],
    sample_rate: int,
    segment_length: int,
    eq_chain: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    hop_length: Optional[int] = None,
    target_peak: Optional[float] = 1.0,
) -> PipelineResult:
    """Load, preprocess, EQ-filter, and measure a batch of audio files in one call.

    The EQ chain is applied to each file's full continuous waveform (after
    resampling/mono/normalization), and only then cut into segments. This
    matters because the chain is a stateful IIR filter: filtering each
    segment independently would reset the filter's internal state at every
    segment boundary, producing a spurious transient there that would not
    occur in continuous playback. Segmenting after filtering avoids that.

    Args:
        paths: Audio file paths to process.
        sample_rate: Sample rate every file is resampled to.
        segment_length: Number of samples per segment.
        eq_chain: Sequence of (b, a) biquad coefficient pairs (from
            llm_music.eq design functions), applied in order to the full
            waveform. An empty sequence leaves it unchanged.
        hop_length: Stride between segment starts. Defaults to segment_length.
        target_peak: If given, peak-normalize each file's waveform (before EQ)
            to this level.

    Returns:
        A PipelineResult with the preprocessed segments, the EQ-processed
        segments, and per-segment RMS levels before and after EQ.
    """
    if not paths:
        raise ValueError("paths must not be empty")

    all_segments = []
    all_processed_segments = []
    for path in paths:
        waveform, _ = load_audio(path, target_sr=sample_rate, mono=True)
        waveform = to_mono(waveform)
        if target_peak is not None:
            waveform = normalize_peak(waveform, target_peak)
        processed_waveform = apply_eq_chain(waveform, eq_chain)

        all_segments.append(segment(waveform, segment_length, hop_length))
        all_processed_segments.append(segment(processed_waveform, segment_length, hop_length))

    segments = torch.cat(all_segments, dim=0)
    processed_segments = torch.cat(all_processed_segments, dim=0)

    rms_before = torch.sqrt(torch.mean(segments**2, dim=-1))
    rms_after = torch.sqrt(torch.mean(processed_segments**2, dim=-1))

    return PipelineResult(segments, processed_segments, rms_before, rms_after)
