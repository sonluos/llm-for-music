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
from llm_music.convolution import convolve, frequency_response, moving_average_kernel
from llm_music.eq import apply_biquad, high_shelf, highpass, low_shelf, lowpass, peaking
from llm_music.preprocess import preprocess_batch
from llm_music.signals import generate_multitone, generate_sine
from llm_music.spectrum import compute_fft, magnitude_to_db

SAMPLE_RATE = 44100
DURATION = 2.0
FREQS = (440.0, 1000.0)
TOP_K = 5

FILTER_FREQS = (440.0, 4000.0)
FILTER_KERNEL_SIZE = 9

DATA_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "output"
DATA_INPUT = Path(__file__).resolve().parent.parent / "data" / "input"


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

    # --- Filter frequency response (independent of any input signal) ---
    freqs_response, magnitude_response = frequency_response(kernel, SAMPLE_RATE)
    db_response = magnitude_to_db(magnitude_response)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_response, db_response)
    plt.xlim(0, SAMPLE_RATE / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Frequency Response — {FILTER_KERNEL_SIZE}-tap Moving Average")
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "filter_frequency_response.png")
    plt.close()

    print(f"Saved filter frequency response plot to {DATA_OUTPUT / 'filter_frequency_response.png'}")

    # --- Parametric EQ: frequency response of each biquad filter type ---
    def _biquad_response(b: torch.Tensor, a: torch.Tensor, length: int = 8192):
        impulse = torch.zeros(length)
        impulse[0] = 1.0
        ir = apply_biquad(impulse, b, a)
        freqs_ir, magnitude_ir = compute_fft(ir, SAMPLE_RATE)
        return freqs_ir[1:], magnitude_ir[1:]  # drop DC bin (invalid on a log-frequency axis)

    eq_filters = {
        "Low-pass 1kHz": lowpass(1000.0, 0.7071, SAMPLE_RATE),
        "High-pass 1kHz": highpass(1000.0, 0.7071, SAMPLE_RATE),
        "Peaking +12dB @1kHz": peaking(1000.0, 12.0, 1.0, SAMPLE_RATE),
        "Low-shelf +12dB @200Hz": low_shelf(200.0, 12.0, SAMPLE_RATE),
        "High-shelf +12dB @5kHz": high_shelf(5000.0, 12.0, SAMPLE_RATE),
    }

    plt.figure(figsize=(10, 5))
    for label, (b, a) in eq_filters.items():
        freqs_eq, magnitude_eq = _biquad_response(b, a)
        plt.plot(freqs_eq, magnitude_to_db(magnitude_eq), label=label)
    plt.xscale("log")
    plt.xlim(20, SAMPLE_RATE / 2)
    plt.xlabel("Frequency (Hz, log scale)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Parametric EQ — Filter Type Frequency Responses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "eq_frequency_responses.png")
    plt.close()

    print(f"Saved parametric EQ frequency response plot to {DATA_OUTPUT / 'eq_frequency_responses.png'}")

    # --- Batch preprocessing pipeline (resample, mono, normalize, segment) ---
    # Two "source files" with different sample rates and loudness, to show that
    # preprocess_batch brings them to a common rate and peak level before segmenting.
    DATA_INPUT.mkdir(parents=True, exist_ok=True)
    quiet_tone_sr = 22050
    quiet_tone = generate_sine(300.0, 1.0, quiet_tone_sr, amplitude=0.3)
    loud_tone = generate_sine(600.0, 0.5, SAMPLE_RATE, amplitude=0.9)

    quiet_path = DATA_INPUT / "preprocess_quiet_22050hz.wav"
    loud_path = DATA_INPUT / "preprocess_loud_44100hz.wav"
    save_audio(quiet_path, quiet_tone, quiet_tone_sr)
    save_audio(loud_path, loud_tone, SAMPLE_RATE)

    segment_length = SAMPLE_RATE // 4  # 0.25 s segments
    batch = preprocess_batch(
        [quiet_path, loud_path], sample_rate=SAMPLE_RATE, segment_length=segment_length
    )
    print(
        f"Preprocessed batch shape: {tuple(batch.shape)} "
        f"(segments, segment_length) at {SAMPLE_RATE} Hz, peak-normalized to {batch.abs().max():.2f}"
    )


if __name__ == "__main__":
    main()
