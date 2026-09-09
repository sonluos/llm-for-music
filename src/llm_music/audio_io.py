"""Loading and saving audio waveforms as tensors."""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file into a waveform tensor.

    Args:
        path: Path to the audio file.
        target_sr: If given, resample the waveform to this sample rate.
        mono: If True, downmix multi-channel audio to a single channel.

    Returns:
        Tuple of (waveform, sample_rate). waveform has shape (channels, samples).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = torchaudio.load(str(path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sr is not None and target_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr

    return waveform, sample_rate


def save_audio(
    path: Union[str, Path],
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Save a waveform tensor to an audio file, creating parent directories as needed.

    Args:
        path: Output file path.
        waveform: Tensor of shape (samples,) or (channels, samples).
        sample_rate: Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 2:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(waveform.shape)}")

    torchaudio.save(str(path), waveform.detach().cpu(), sample_rate)
