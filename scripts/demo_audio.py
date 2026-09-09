"""Demo: generate a two-tone signal, save audio + plots, report FFT peaks."""

import sys
from pathlib import Path

# Allow running directly via `python scripts/demo_audio.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

# The demo only writes image files, so avoid GUI backends (and their Tcl/Tk
# dependency) when running on Windows or in a headless environment.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from llm_music.audio_io import save_audio
from llm_music.convolution import convolve, moving_average_kernel
from llm_music.signals import generate_multitone
from llm_music.spectrum import compute_fft, magnitude_to_db

SAMPLE_RATE = 44100
DURATION = 2.0
FREQS = (440.0, 1000.0)
TOP_K = 5

FILTER_FREQS = (440.0, 4000.0)
FILTER_KERNEL_SIZE = 9

DATA_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "output"


def main() -> None:
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

    waveform = generate_multitone(FREQS, DURATION, SAMPLE_RATE, amplitude=0.8)
    save_audio(DATA_OUTPUT / "test_signal.wav", waveform, SAMPLE_RATE)

    # Waveform plot: first 10 ms, enough to see both tones' shape.
    window = int(0.01 * SAMPLE_RATE)
    t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
    plt.figure(figsize=(10, 3))
    plt.plot(t, waveform[:window])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform (first 10 ms)")
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "waveform.png")
    plt.close()

    freqs, magnitude = compute_fft(waveform, SAMPLE_RATE)
    magnitude_db = magnitude_to_db(magnitude)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude_db)
    plt.xlim(0, 5000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT Spectrum")
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "spectrum.png")
    plt.close()

    top_indices = torch.topk(magnitude, TOP_K).indices
    print("Strongest FFT frequencies:")
    for idx in top_indices.tolist():
        print(f"  {freqs[idx]:8.1f} Hz  ({magnitude_db[idx]:6.1f} dB)")

    # --- Moving-average filtering demo (440 Hz + 4000 Hz) ---
    filter_waveform = generate_multitone(FILTER_FREQS, DURATION, SAMPLE_RATE, amplitude=0.8)
    kernel = moving_average_kernel(FILTER_KERNEL_SIZE)
    filtered_full = convolve(filter_waveform, kernel)

    # convolve() returns the full convolution, which is longer than the input.
    # Trim it back to the original length (centered) so the original and
    # filtered spectra share the same FFT bin frequencies for a fair comparison.
    # The few samples nearest each edge are affected by convolve()'s zero
    # padding, so the filtered spectrum's noise floor is expectedly higher
    # than the original's; the 440/4000 Hz peaks are unaffected by this.
    trim_start = (FILTER_KERNEL_SIZE - 1) // 2
    filtered_waveform = filtered_full[trim_start : trim_start + filter_waveform.shape[-1]]

    freqs_orig, magnitude_orig = compute_fft(filter_waveform, SAMPLE_RATE)
    freqs_filt, magnitude_filt = compute_fft(filtered_waveform, SAMPLE_RATE)
    db_orig = magnitude_to_db(magnitude_orig)
    db_filt = magnitude_to_db(magnitude_filt)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_orig, db_orig, label="Original")
    plt.plot(freqs_filt, db_filt, label=f"{FILTER_KERNEL_SIZE}-tap moving average", alpha=0.8)
    plt.xlim(0, 5000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Effect of Moving-Average Filter (440 Hz + 4000 Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "filter_comparison.png")
    plt.close()

    print(f"Saved filter comparison plot to {DATA_OUTPUT / 'filter_comparison.png'}")


if __name__ == "__main__":
    main()
