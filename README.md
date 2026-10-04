# LLM for Music Production — DSP Foundation

Research project goal: **natural-language instruction + audio information → LLM → structured EQ/Reverb parameters.**

This stage implements only the DSP foundation the rest of the pipeline will build on:
audio I/O, synthetic test signal generation, FFT-based spectral analysis, short-time
Fourier transform/spectrograms, 1D convolution/filtering, parametric EQ (biquad IIR
filters, applied individually or as a chain), a batch audio preprocessing pipeline,
analog-vs-digital quality tradeoffs (bit-depth quantization, sample-rate round-trip
loss), an integrated preprocessing-to-EQ execution pipeline, a basic audio feature
extraction module (RMS, spectral centroid, spectral rolloff, band energy),
normalization/formatting of those features into an LLM-prompt-ready JSON string, a
training-example data structure linking an instruction, audio features, and target
effect parameters (with JSON Lines save/load), and a text-only baseline model (a small
open-source LLM) that predicts an effect + parameters from a natural-language
instruction. No reverb DSP or audio-conditioned ("proposed") model is included yet.

## Structure

```
src/llm_music/
    audio_io.py     # load_audio / save_audio (torchaudio-backed)
    signals.py      # generate_sine / generate_multitone
    spectrum.py     # compute_fft / magnitude_to_db
    stft.py         # compute_stft: framed, windowed, time-varying spectrum
                    # (spectrogram), built on preprocess.segment + torch.fft.rfft
    features.py     # compute_rms, spectral_centroid, spectral_rolloff, band_energy
    llm_input.py    # extract_feature_vector / normalize_features / format_for_llm_input,
                    # and audio_to_llm_input tying them into one call (a JSON string)
    dataset.py      # build_example: links an instruction + audio features + target
                    # effect params into one record; save_dataset / load_dataset (JSONL)
    baseline_model.py # predict_effect_params: text-only instruction -> JSON effect
                    # params via a small instruction-tuned LLM (Qwen2.5-0.5B-Instruct,
                    # lazily loaded); validate_effect_params checks schema conformance
    convolution.py  # convolve (full 1D linear convolution) / moving_average_kernel /
                    # frequency_response (a filter's own magnitude response)
    eq.py           # biquad filter design (lowpass, highpass, peaking, low_shelf,
                    # high_shelf), apply_biquad (single-filter IIR via
                    # torchaudio.lfilter), and apply_eq_chain (several filters in series)
    preprocess.py   # resample / to_mono / normalize_peak / normalize_rms / segment,
                    # and preprocess_batch tying them into one pipeline
    quantize.py     # quantize_bit_depth (ADC/DAC bit-depth simulation),
                    # signal_to_noise_ratio, theoretical_sqnr_db
    pipeline.py     # run_pipeline: preprocess (per file) -> apply_eq_chain on the full
                    # waveform -> segment -> per-segment RMS before/after, as one
                    # integrated call (PipelineResult)
scripts/
    demo_audio.py # generates a 440 Hz + 1000 Hz tone (saves WAV + plots, prints FFT
                  # peaks); a 440 Hz + 4000 Hz tone filtered with a moving-average
                  # kernel to compare original vs. filtered spectra and the kernel's own
                  # frequency response; each parametric EQ filter type's response; a
                  # batch preprocessing run over two files with different sample rates
                  # and loudness; bit-depth quantization quality vs. the theoretical SQNR
                  # formula; sample-rate round-trip quality/spectral loss; the integrated
                  # pipeline's RMS before/after an EQ chain; a spectrogram of a tone that
                  # changes frequency over time (440 -> 1000 -> 4000 Hz); RMS/spectral
                  # centroid/rolloff/band energy compared across two tones; each tone's
                  # normalized features formatted as an LLM-prompt-ready JSON string; a
                  # small synthetic training set (Korean instructions + target EQ params)
                  # saved/reloaded as JSON Lines; and the baseline LLM's predicted effect
                  # params for those same instructions, compared against the targets
tests/
    test_dsp.py         # sine length, FFT peak accuracy, save/load consistency
    test_convolution.py # convolution output length, impulse response, kernel normalization
    test_eq.py          # each biquad filter type's frequency response behaves as designed
    test_preprocess.py  # resample/mono/normalize/segment correctness, batch pipeline shape
    test_quantize.py    # quantization level limits, SNR metric, measured vs. theoretical SQNR
    test_pipeline.py    # EQ chain ordering/passthrough, integrated pipeline shapes/RMS/IIR continuity
    test_stft.py        # STFT shapes, validation, and tracking stationary/changing tone frequency
    test_features.py    # RMS/centroid/rolloff against known tones, band energy concentration/coverage
    test_llm_input.py   # feature-dict keys/validation, Hz normalization/clamping, valid-JSON output
    test_dataset.py      # example structure/validation, JSONL save/load roundtrip, Korean text I/O
    test_baseline_model.py # JSON extraction/validation (fake generate_fn, no model load),
                    # plus real-model integration tests that load the actual baseline LLM
data/
    input/        # place source audio here (demo also writes its preprocessing inputs here)
    output/       # generated audio and plots land here
```

## Setup

```powershell
pip install -r requirements.txt
```

The baseline model (`llm_music.baseline_model`) downloads `Qwen/Qwen2.5-0.5B-Instruct`
(~1 GB) from Hugging Face on first use and caches it under `~/.cache/huggingface`;
later runs load from the local cache. It runs on CPU — no GPU is required.

## Usage

```powershell
python scripts/demo_audio.py
```

Outputs `test_signal.wav`, `waveform.png`, `spectrum.png`, `filter_comparison.png`,
`filter_frequency_response.png`, and `eq_frequency_responses.png` under `data/output/`,
and prints the strongest FFT frequency bins to the console.

## Tests

```powershell
pytest tests/
```

Tests marked `integration` (the two that load the real baseline LLM) are deselected
by default — see `pytest.ini`. Run them explicitly, once the model is downloaded/
cached, with:

```powershell
pytest tests/ -m integration
```

## Conventions

- Waveform tensors: `torch.Tensor` of shape `(samples,)` or `(channels, samples)`;
  functions handle both.
- All paths use `pathlib.Path` for Windows compatibility.
- `src/` layout — scripts and tests add `src/` to `sys.path` at runtime rather than
  requiring an editable install.
