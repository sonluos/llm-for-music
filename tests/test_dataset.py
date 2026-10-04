"""Tests for linking instructions + audio features + target params into examples."""

import pytest
import torch

from llm_music.dataset import build_example, load_dataset, save_dataset
from llm_music.signals import generate_sine

SAMPLE_RATE = 44100


def test_build_example_structure() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    target_params = {"effect": "high_shelf", "freq_hz": 5000.0, "gain_db": 6.0}
    example = build_example("보컬을 더 선명하게 해줘", waveform, SAMPLE_RATE, target_params)

    assert set(example.keys()) == {"instruction", "audio_features", "target_params"}
    assert example["instruction"] == "보컬을 더 선명하게 해줘"
    assert example["target_params"] == target_params
    assert "rms" in example["audio_features"]
    assert "spectral_centroid_hz" in example["audio_features"]


def test_build_example_rejects_empty_instruction() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        build_example("   ", waveform, SAMPLE_RATE, {"effect": "lowpass"})


def test_build_example_rejects_non_dict_target_params() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        build_example("저음을 줄여줘", waveform, SAMPLE_RATE, target_params="not a dict")


def test_build_example_rejects_non_json_serializable_target_params() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        build_example(
            "저음을 줄여줘", waveform, SAMPLE_RATE, {"effect": "low_shelf", "coeffs": torch.randn(3)}
        )


def test_build_example_rejects_nan_in_target_params() -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        build_example("저음을 줄여줘", waveform, SAMPLE_RATE, {"gain_db": float("nan")})


def test_save_and_load_dataset_roundtrip(tmp_path) -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE, amplitude=0.8)
    examples = [
        build_example("소리를 더 따뜻하게 해줘", waveform, SAMPLE_RATE, {"effect": "peaking", "gain_db": 3.0}),
        build_example("저음을 줄여줘", waveform, SAMPLE_RATE, {"effect": "low_shelf", "gain_db": -5.0}),
    ]
    path = tmp_path / "nested" / "dataset.jsonl"

    save_dataset(examples, path)
    loaded = load_dataset(path)

    assert loaded == examples


def test_load_dataset_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"instruction": "a", "audio_features": {}, "target_params": {}}\n\n\n')

    loaded = load_dataset(path)
    assert len(loaded) == 1


def test_load_dataset_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.jsonl")


@pytest.mark.parametrize(
    "bad_line",
    [
        "[1, 2, 3]",  # not a JSON object
        '{"instruction": "a", "audio_features": {}}',  # missing target_params
        '{"instruction": 123, "audio_features": {}, "target_params": {}}',  # wrong type
        '{"instruction": "a", "audio_features": [], "target_params": {}}',  # wrong type
    ],
)
def test_load_dataset_rejects_malformed_records(tmp_path, bad_line: str) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(bad_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_save_dataset_rejects_malformed_example(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    with pytest.raises(ValueError):
        save_dataset([{"instruction": "a", "audio_features": {}}], path)  # missing target_params


def test_save_dataset_does_not_corrupt_existing_file_on_failure(tmp_path) -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    good_example = build_example("저음을 줄여줘", waveform, SAMPLE_RATE, {"effect": "low_shelf"})
    path = tmp_path / "dataset.jsonl"

    save_dataset([good_example], path)
    original_contents = path.read_text(encoding="utf-8")

    malformed_example = {"instruction": "a", "audio_features": {}}  # missing target_params
    with pytest.raises(ValueError):
        save_dataset([good_example, malformed_example], path)

    # The earlier, valid save must survive a failed later save to the same path.
    assert path.read_text(encoding="utf-8") == original_contents
    assert load_dataset(path) == [good_example]


def test_save_dataset_handles_korean_text_readably(tmp_path) -> None:
    waveform = generate_sine(440.0, 1.0, SAMPLE_RATE)
    examples = [build_example("보컬을 더 밝게", waveform, SAMPLE_RATE, {"effect": "high_shelf"})]
    path = tmp_path / "dataset.jsonl"

    save_dataset(examples, path)
    raw_text = path.read_text(encoding="utf-8")

    # ensure_ascii=False: Korean text should appear directly, not as \uXXXX escapes.
    assert "보컬을 더 밝게" in raw_text
