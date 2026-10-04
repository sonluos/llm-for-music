"""Short-time Fourier transform (STFT) and spectrogram computation.

Builds a time-frequency representation by reusing llm_music.preprocess.segment
for framing, applying a Hann window to each frame to limit spectral leakage,
and batch-computing each frame's FFT magnitude with torch.fft.rfft.
"""

from typing import Optional, Tuple

import torch

from llm_music.preprocess import segment


def compute_stft(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the magnitude short-time Fourier transform (spectrogram) of a waveform.

    Args:
        waveform: 1D tensor of shape (samples,).
        sample_rate: Sample rate in Hz.
        n_fft: Frame length in samples (also the FFT size per frame).
        hop_length: Stride between frame starts, in samples. Defaults to
            n_fft // 4 (75% overlap between consecutive frames).

    Returns:
        Tuple of (freqs, times, magnitude):
            freqs: 1D tensor of shape (n_fft // 2 + 1,), frequency axis in Hz.
            times: 1D tensor of shape (num_frames,), each frame's center time
                in seconds.
            magnitude: 2D tensor of shape (n_fft // 2 + 1, num_frames), the
                FFT magnitude of each frame.
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(waveform.shape)}")
    if not waveform.is_floating_point():
        raise ValueError(f"Expected a floating-point waveform, got dtype {waveform.dtype}")
    if n_fft < 1:
        raise ValueError(f"n_fft must be >= 1, got {n_fft}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    hop_length = hop_length if hop_length is not None else max(n_fft // 4, 1)

    frames = segment(waveform, segment_length=n_fft, hop_length=hop_length)
    window = torch.hann_window(n_fft, dtype=frames.dtype, device=frames.device)
    spectrum = torch.fft.rfft(frames * window, dim=-1)
    magnitude = spectrum.abs().transpose(0, 1)

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    num_frames = frames.shape[0]
    frame_starts = torch.arange(num_frames, dtype=torch.float64) * hop_length
    times = (frame_starts + n_fft / 2) / sample_rate

    return freqs, times, magnitude
