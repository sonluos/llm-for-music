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
effect parameters (with JSON Lines save/load), a text-only baseline model (a small
open-source LLM) that predicts an effect + parameters from a natural-language
instruction, an audio-conditioned "proposed model" that embeds the source audio's
features into the same LLM's prompt, LoRA fine-tuning code for that proposed model,
a Schroeder/Freeverb-style reverb module, a dispatcher that applies a model's
predicted effect dict (EQ or reverb) to real audio — closing the loop from
instruction to processed sound — and the evaluation metrics (parameter error, valid
output ratio, repetition stability, acoustic feature change) the proposal names for
comparing the baseline and proposed models. This is the full DSP + model foundation
the project's research comparisons and further training will build on.

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
                    # lazily loaded); validate_effect_params checks schema conformance;
                    # generate_with_model/generate_from_user_content/
                    # get_model_and_tokenizer expose the model for reuse by
                    # proposed_model.py and train_lora.py
    proposed_model.py # predict_effect_params_with_audio: instruction + source audio's
                    # features (as JSON in the prompt) -> JSON effect params, using the
                    # same LLM/schema as baseline_model (prompt-level fusion, not a
                    # learned embedding-space projection)
    train_lora.py   # build_lora_model / train_one_epoch / save_lora_adapter /
                    # load_lora_adapter: LoRA (via peft) fine-tuning of a *separate*
                    # copy of the LLM on an llm_music.dataset JSON Lines dataset, with
                    # loss masked to the target_params tokens only
    reverb.py       # apply_reverb: a Schroeder/Freeverb-style parallel-comb +
                    # series-allpass reverberator (room_size, damping, decay, wet_dry),
                    # each stage expressed as a closed-form IIR via torchaudio.lfilter
    apply_effect.py # apply_predicted_effect: dispatches a schema-valid predicted
                    # effect dict (from baseline_model/proposed_model) to the matching
                    # llm_music.eq filter or llm_music.reverb.apply_reverb call
    evaluate.py     # parameter_error / evaluate_predictions (valid output ratio,
                    # effect match ratio, mean parameter error) / repetition_stability /
                    # acoustic_feature_change — pure metric functions, no model calls
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
                  # saved/reloaded as JSON Lines; the baseline LLM's predicted effect
                  # params for those instructions; the proposed (audio-conditioned)
                  # model's predictions for the same instructions; a LoRA fine-tuning
                  # run on the dataset, comparing one instruction's prediction before vs.
                  # after training; an aggregate baseline-vs-proposed evaluation (valid
                  # output ratio, effect match ratio, mean parameter error) over those
                  # instructions; repetition stability over 5 sampled trials on one
                  # instruction; the reverb module's impulse response at different decay
                  # settings; the fine-tuned model's (or, if not schema-valid, the
                  # dataset target's) predicted effect applied to real audio and saved
                  # as a WAV file; and that effect's acoustic feature change
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
    test_proposed_model.py # audio+instruction prompt construction, fake-generate_fn
                    # inference, plus a real-model integration test
    test_train_lora.py  # LoRA wrapping, training reduces loss, adapter save/load
                    # roundtrip — entirely `integration` (needs the real model)
    test_reverb.py       # output shape/finiteness, wet/dry extremes, decay behavior
    test_apply_effect.py # each effect type dispatches correctly, rejects invalid params
    test_evaluate.py     # parameter_error/evaluate_predictions/repetition_stability
                    # numeric correctness, acoustic_feature_change before/after/delta
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
