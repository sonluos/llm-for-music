"""Demo: generate a two-tone signal, save audio + plots, report FFT peaks."""

import math
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
from llm_music.pipeline import run_pipeline
from llm_music.preprocess import preprocess_batch, resample
from llm_music.quantize import quantize_bit_depth, signal_to_noise_ratio, theoretical_sqnr_db
from llm_music.signals import generate_multitone, generate_sine
from llm_music.spectrum import compute_fft, magnitude_to_db
from llm_music.stft import compute_stft

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

    # --- Bit-depth quantization: measured SNR vs the theoretical SQNR formula ---
    # The 6.02*bits+1.76 formula assumes a full-scale (0 dBFS) single sine wave;
    # using a dedicated near-full-scale tone (rather than the 0.8-peak two-tone
    # `waveform` above) lets the measured curve match theory directly.
    quantization_test_tone = generate_sine(437.0, DURATION, SAMPLE_RATE, amplitude=0.99)
    bit_depths = [16, 12, 10, 8, 6, 4, 2]
    measured_snrs = []
    theoretical_snrs = []
    print("Bit-depth quantization quality (SQNR approx. 6.02*bits + 1.76 dB):")
    for bits in bit_depths:
        quantized = quantize_bit_depth(quantization_test_tone, bits)
        measured = signal_to_noise_ratio(quantization_test_tone, quantized)
        theoretical = theoretical_sqnr_db(bits)
        measured_snrs.append(measured)
        theoretical_snrs.append(theoretical)
        print(f"  {bits:2d}-bit:  measured {measured:6.2f} dB   theoretical {theoretical:6.2f} dB")

    plt.figure(figsize=(8, 5))
    plt.plot(bit_depths, measured_snrs, "o-", label="Measured SNR")
    plt.plot(bit_depths, theoretical_snrs, "--", label="Theoretical SQNR (6.02·bits + 1.76)")
    plt.xlabel("Bit depth")
    plt.ylabel("SNR (dB)")
    plt.title("Quantization Quality vs. Bit Depth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "bit_depth_quality.png")
    plt.close()

    print(f"Saved bit-depth quality plot to {DATA_OUTPUT / 'bit_depth_quality.png'}")

    # --- Sample-rate quality: downsample-then-restore round trip vs. original ---
    round_trip_rates = [44100, 8000, 2000, 800]
    rate_snrs = []
    print("Sample-rate round-trip quality (downsample, then resample back to 44100 Hz):")
    for target_sr in round_trip_rates:
        downsampled = resample(waveform, SAMPLE_RATE, target_sr)
        restored = resample(downsampled, target_sr, SAMPLE_RATE)
        n = min(waveform.shape[-1], restored.shape[-1])
        snr = signal_to_noise_ratio(waveform[:n], restored[:n])
        rate_snrs.append(snr)
        print(f"  {target_sr:6d} Hz:  SNR {snr:7.2f} dB")

    # Drop infinite values (the no-op 44100 Hz case) so they don't break axis scaling;
    # the printed table above already reports the identity case as "inf dB".
    finite_rates = [r for r, s in zip(round_trip_rates, rate_snrs) if math.isfinite(s)]
    finite_snrs = [s for s in rate_snrs if math.isfinite(s)]

    plt.figure(figsize=(8, 5))
    plt.semilogx(finite_rates, finite_snrs, "o-")
    plt.xlabel("Intermediate sample rate (Hz, log scale)")
    plt.ylabel("SNR (dB)")
    plt.title("Round-Trip Resampling Quality vs. Sample Rate")
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "sample_rate_quality.png")
    plt.close()

    print(f"Saved sample-rate quality plot to {DATA_OUTPUT / 'sample_rate_quality.png'}")

    # Spectral comparison at the most aggressive sample rate: both 440/1000 Hz tones
    # fall above the 400 Hz Nyquist limit of an 800 Hz intermediate rate and should
    # alias/disappear after the round trip.
    worst_sr = round_trip_rates[-1]
    worst_restored = resample(resample(waveform, SAMPLE_RATE, worst_sr), worst_sr, SAMPLE_RATE)
    n = min(waveform.shape[-1], worst_restored.shape[-1])
    freqs_before, magnitude_before = compute_fft(waveform[:n], SAMPLE_RATE)
    freqs_after, magnitude_after = compute_fft(worst_restored[:n], SAMPLE_RATE)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_before, magnitude_to_db(magnitude_before), label="Original (44100 Hz)")
    plt.plot(
        freqs_after,
        magnitude_to_db(magnitude_after),
        label=f"After {worst_sr} Hz round trip",
        alpha=0.8,
    )
    plt.xlim(0, 5000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Spectral Loss from {worst_sr} Hz Round-Trip Resampling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "sample_rate_spectral_loss.png")
    plt.close()

    print(f"Saved sample-rate spectral loss plot to {DATA_OUTPUT / 'sample_rate_spectral_loss.png'}")

    # --- Integrated pipeline: preprocessing -> EQ chain -> RMS readout, in one call ---
    # Reuses the two input files from the preprocessing section above (different
    # sample rates and loudness), now run through an EQ chain (high-pass to remove
    # rumble, then a peaking boost) with before/after level measurement.
    eq_chain = [
        highpass(100.0, 0.7071, SAMPLE_RATE),
        peaking(1000.0, 6.0, 1.0, SAMPLE_RATE),
    ]
    pipeline_result = run_pipeline(
        [quiet_path, loud_path], SAMPLE_RATE, segment_length=segment_length, eq_chain=eq_chain
    )

    print("Integrated pipeline (preprocess -> EQ chain -> RMS):")
    for i, (before, after) in enumerate(zip(pipeline_result.rms_before, pipeline_result.rms_after)):
        print(f"  segment {i}: RMS before {before:.4f} -> after {after:.4f}")

    segment_indices = range(pipeline_result.rms_before.shape[0])
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - width / 2 for i in segment_indices], pipeline_result.rms_before, width, label="Before EQ"
    )
    plt.bar(
        [i + width / 2 for i in segment_indices], pipeline_result.rms_after, width, label="After EQ"
    )
    plt.xlabel("Segment index")
    plt.ylabel("RMS level")
    plt.title("Integrated Pipeline — RMS Before/After EQ Chain")
    plt.xticks(list(segment_indices))
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "pipeline_rms_comparison.png")
    plt.close()

    print(f"Saved pipeline RMS comparison plot to {DATA_OUTPUT / 'pipeline_rms_comparison.png'}")

    # --- STFT / spectrogram: visualize frequency content changing over time ---
    # Three 1-second tones back to back, so the spectrogram should show three
    # distinct horizontal bands at different frequencies over time.
    stft_tone_durations = 1.0
    stft_freqs_sequence = (440.0, 1000.0, 4000.0)
    stft_waveform = torch.cat(
        [generate_sine(f, stft_tone_durations, SAMPLE_RATE, amplitude=0.8) for f in stft_freqs_sequence]
    )

    n_fft = 2048
    hop_length = 512
    stft_freq_axis, stft_time_axis, stft_magnitude = compute_stft(
        stft_waveform, SAMPLE_RATE, n_fft, hop_length
    )
    stft_db = magnitude_to_db(stft_magnitude)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        stft_db,
        aspect="auto",
        origin="lower",
        extent=[stft_time_axis[0], stft_time_axis[-1], stft_freq_axis[0], stft_freq_axis[-1]],
        cmap="magma",
    )
    plt.ylim(0, 5000)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram — 440 Hz -> 1000 Hz -> 4000 Hz")
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(DATA_OUTPUT / "spectrogram.png")
    plt.close()

    print(f"Saved spectrogram plot to {DATA_OUTPUT / 'spectrogram.png'}")

    # Confirm the frequency tracked by the spectrogram at the center of each tone.
    print("Spectrogram frequency tracking (peak bin per tone's center time):")
    for i, expected_freq in enumerate(stft_freqs_sequence):
        center_time = (i + 0.5) * stft_tone_durations
        frame_idx = (stft_time_axis - center_time).abs().argmin()
        peak_freq = stft_freq_axis[stft_magnitude[:, frame_idx].argmax()].item()
        print(f"  t={center_time:.1f}s: expected {expected_freq:.0f} Hz, detected {peak_freq:.0f} Hz")


if __name__ == "__main__":
    main()
