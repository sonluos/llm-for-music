# LLM for Music Production — Project Context for Claude Code

This file exists so a fresh Claude Code session (on any machine, with zero prior
context) can immediately understand what this project is, what's been built,
and how to keep working on it. Read this whole file before making changes.

## Research goal

**natural-language instruction + audio information → LLM → structured EQ/Reverb
parameters.**

This is a 16-week university 프로젝트학기제 project (학생: 손우진, 응용수리과학부).
The full plan lives in `proposal/계획서.pdf` (not in git — see "What's NOT in this
repo" below). The research question: does giving an LLM the *audio's own
features* (not just a text instruction) improve its ability to predict correct
EQ/Reverb parameters, compared to a text-only baseline?

## Current status: weeks 1–15 complete, week 16 (final report) remaining

Everything through week 15 is implemented, tested, and pushed to GitHub at
**https://github.com/sonluos/llm-for-music** (branch `master`). Run
`git log --oneline` for the full commit history — each week has its own
commit(s) with a descriptive message, often followed by a fix-up commit after
a code review caught a bug (see "How this project was built" below).

Week 16 (실험 결과 분석, 코드·데이터 정리, 최종 연구보고서 작성) has not started.

## Quick start on a new machine

```powershell
git clone https://github.com/sonluos/llm-for-music.git
cd llm-for-music
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest tests          # 178 passed, 10 deselected, ~6s
.venv\Scripts\python.exe scripts\demo_audio.py    # runs everything end to end
```

The first `demo_audio.py` run (or first `pytest -m integration`) downloads
`Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) from Hugging Face and caches it under
`~/.cache/huggingface`. Runs entirely on CPU — no GPU needed, no API keys.

Tests marked `integration` (load the real LLM) are deselected by default
(`pytest.ini`). Run them explicitly with `pytest tests/ -m integration`
(~1 minute).

## Architecture (src/llm_music/)

Roughly in build order; later modules depend on earlier ones:

| Module | What it does |
|---|---|
| `audio_io.py` | load/save WAV (torchaudio) |
| `signals.py` | generate_sine / generate_multitone (test signals) |
| `spectrum.py` | compute_fft, magnitude_to_db |
| `stft.py` | compute_stft (spectrogram) |
| `convolution.py` | 1D convolution, moving-average filter, frequency_response |
| `eq.py` | biquad EQ filters (lowpass/highpass/peaking/low_shelf/high_shelf), apply_eq_chain |
| `preprocess.py` | resample, to_mono, normalize, segment, preprocess_batch |
| `quantize.py` | bit-depth quantization, SNR, theoretical SQNR |
| `pipeline.py` | run_pipeline: preprocess → EQ chain → RMS before/after |
| `features.py` | RMS, spectral centroid, spectral rolloff, band energy |
| `llm_input.py` | normalize audio features → JSON string for an LLM prompt |
| `dataset.py` | build_example: (instruction, audio features, target params) → JSONL |
| `baseline_model.py` | **loads the actual LLM** (Qwen2.5-0.5B-Instruct). predict_effect_params: text-only instruction → JSON effect params. Also exposes get_model_and_tokenizer / generate_with_model / generate_from_user_content for reuse, and the EFFECT_SCHEMAS/FIELD_RANGES/SYSTEM_PROMPT that define the output schema |
| `proposed_model.py` | predict_effect_params_with_audio: instruction + audio features (as JSON in the prompt) → JSON effect params. Same LLM/schema as baseline, audio info added via prompt text (not a learned projection layer — see decisions below) |
| `train_lora.py` | LoRA fine-tuning (via `peft`) of a *separate* copy of the LLM on a dataset.py JSONL file |
| `reverb.py` | Schroeder/Freeverb-style reverb (room_size, damping, decay, wet_dry), implemented as closed-form IIR filters via torchaudio.lfilter |
| `apply_effect.py` | apply_predicted_effect: dispatches a model's predicted JSON to the actual eq.py filter or reverb.py call — this is what turns a prediction into real processed audio |
| `evaluate.py` | parameter_error, evaluate_predictions (valid output ratio, effect match ratio, mean param error), repetition_stability, acoustic_feature_change — the baseline-vs-proposed comparison metrics |

`scripts/demo_audio.py` exercises literally everything above in one run, in
build order, and is the best single file to read to see the whole system work
end to end. `tests/` mirrors `src/llm_music/` one file per module.

## Key design decisions (asked of the user, don't re-litigate without reason)

- **Week 12 baseline model**: the proposal said "공개형 경량 LLM" — the user chose
  to actually wire up a real small open-source LLM (Qwen2.5-0.5B-Instruct via
  `transformers`, CPU-only) rather than a rule-based keyword baseline or an
  external paid API.
- **Week 13 audio fusion**: the proposal mentions a "투영 모듈" (projection
  module), which usually means injecting a learned embedding into the model's
  input embedding space. The user chose the simpler alternative: embed the
  audio features as JSON text directly in the prompt. This was a deliberate
  scope tradeoff (much simpler to implement/debug/train on CPU) — don't
  "fix" this into an embedding-projection architecture without asking first.
- **LoRA, not QLoRA**: QLoRA needs 4-bit quantization (`bitsandbytes`), which
  has little/no working CPU support on Windows. The proposal allows either;
  plain LoRA was used.
- **Training never touches the shared baseline model instance**: `train_lora.py`
  always loads its own fresh copy of the LLM. Mutating `baseline_model.py`'s
  cached singleton with LoRA layers would corrupt the baseline-vs-proposed
  comparison, since both would then share state.

## Known quirks

- **Windows console encoding**: this environment's Python stdout is `cp949`.
  Printing Korean text is fine (cp949 supports Hangul), but some Unicode
  symbols (e.g. `≈`) will crash `print()` — stick to ASCII in print statements.
  If Korean text looks garbled in a *terminal capture/log*, check the actual
  file on disk before assuming it's corrupted — it usually isn't (this bit us
  more than once; the JSONL dataset files have always been correct UTF-8).
- The small LLM (0.5B params) is **not** reliably schema-perfect. It
  sometimes misspells effect names (`"high_pass"` instead of `"highpass"`),
  omits required fields, or picks a semantically weird effect for vague
  instructions ("풍부하게"). This is expected and is exactly what
  `evaluate.py`'s metrics are for — don't over-tune the prompt to paper over
  it; measuring the baseline's real limitations is the point.
- `torchaudio`/`transformers` print some deprecation/UserWarnings (e.g. about
  `torchcodec` in a future torchaudio version) that are harmless noise, not
  bugs.

## How this project was built (useful context for continuing it)

Each week's feature went through: implement → run tests + demo → an
independent code review caught real bugs → fix → re-verify → **ask the user
before committing/pushing** (this is a standing instruction — never commit or
push without explicit approval, even mid-task). Examples of bugs review
caught and fixed: IIR filter state resetting at segment boundaries (week 7),
quantizer producing 2^bits+1 instead of 2^bits levels (week 6), a misleading
band-energy denominator in `llm_input.py` (week 10), an AdamW optimizer being
recreated every epoch instead of persisting its momentum state (week 13), and
`repetition_stability` treating repeated *unparseable* outputs as "agreeing"
with each other (week 15). If you're extending this code, assume nothing is
bug-free until it's been through this same scrutiny.

## What's NOT in this repo (local-only, not backed up by git)

- `proposal/` — 계획서.pdf and the 15 reference papers (`.gitignore`d; contains
  personal info). The user has these backed up separately.
- `.venv/`, `data/output/`, `data/input/*.wav` — all regenerable; not worth
  backing up.
- The downloaded LLM weights (`~/.cache/huggingface`) — re-downloads
  automatically on first use elsewhere.

## Next step: week 16

제안서 week 16: "실험 결과 분석, 코드·데이터 정리 및 최종 연구보고서 작성" — analyze
the baseline-vs-proposed results (evaluate.py already computes them), tidy up
code/data, and write the final report. No code implementation is strictly
required for this week per the proposal — it's writing and analysis.
