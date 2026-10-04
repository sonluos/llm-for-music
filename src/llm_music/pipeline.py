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

from llm_music.eq import apply_eq_chain
from llm_music.preprocess import preprocess_batch


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
) -> PipelineResult:
    """Load, preprocess, EQ-filter, and measure a batch of audio files in one call.

    Args:
        paths: Audio file paths to process.
        sample_rate: Sample rate every file is resampled to.
        segment_length: Number of samples per segment.
        eq_chain: Sequence of (b, a) biquad coefficient pairs (from
            llm_music.eq design functions), applied in order to each segment.
            An empty sequence leaves segments unchanged.
        hop_length: Stride between segment starts. Defaults to segment_length.

    Returns:
        A PipelineResult with the preprocessed segments, the EQ-processed
        segments, and per-segment RMS levels before and after EQ.
    """
    segments = preprocess_batch(paths, sample_rate, segment_length, hop_length)
    processed_segments = torch.stack([apply_eq_chain(seg, eq_chain) for seg in segments])

    rms_before = torch.sqrt(torch.mean(segments**2, dim=-1))
    rms_after = torch.sqrt(torch.mean(processed_segments**2, dim=-1))

    return PipelineResult(segments, processed_segments, rms_before, rms_after)
