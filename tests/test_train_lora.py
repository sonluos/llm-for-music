"""Tests for LoRA fine-tuning of the proposed model.

All tests here load the real baseline LLM (there is no meaningful way to fake
a `peft`-wrapped model), so the whole module is marked `integration` and
deselected by default — run with `pytest -m integration`.
"""

import pytest
import torch

from llm_music.baseline_model import generate_with_model
from llm_music.train_lora import (
    _build_labels,
    build_lora_model,
    build_optimizer,
    load_lora_adapter,
    save_lora_adapter,
    train_one_epoch,
)

pytestmark = pytest.mark.integration

EXAMPLE = {
    "instruction": "make the bass louder",
    "audio_features": {"rms": 0.4, "spectral_centroid_hz": 0.1, "spectral_rolloff_hz": 0.15},
    "target_params": {"effect": "low_shelf", "freq_hz": 150, "gain_db": 5},
}


def test_build_lora_model_has_small_trainable_fraction() -> None:
    peft_model, _ = build_lora_model()
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())

    assert 0 < trainable < total
    assert trainable / total < 0.01  # LoRA should touch well under 1% of parameters


def test_train_one_epoch_rejects_empty_examples() -> None:
    peft_model, tokenizer = build_lora_model()
    optimizer = build_optimizer(peft_model)
    with pytest.raises(ValueError):
        train_one_epoch(peft_model, tokenizer, [], optimizer)


def test_train_one_epoch_returns_one_loss_per_example() -> None:
    peft_model, tokenizer = build_lora_model()
    optimizer = build_optimizer(peft_model)
    losses = train_one_epoch(peft_model, tokenizer, [EXAMPLE, EXAMPLE], optimizer)
    assert len(losses) == 2
    assert all(isinstance(loss, float) for loss in losses)


def test_training_reduces_loss_on_a_repeated_example() -> None:
    peft_model, tokenizer = build_lora_model()
    optimizer = build_optimizer(peft_model)
    first_epoch_loss = train_one_epoch(peft_model, tokenizer, [EXAMPLE], optimizer)[0]
    for _ in range(3):
        last_loss = train_one_epoch(peft_model, tokenizer, [EXAMPLE], optimizer)[0]
    assert last_loss < first_epoch_loss


def test_optimizer_state_persists_across_epoch_calls() -> None:
    # Adam's per-parameter "step" count should keep increasing across separate
    # train_one_epoch calls that share one optimizer, proving its state isn't
    # being reset (the bug this design fixes: a fresh AdamW per call would
    # leave every param's state empty, so `state` would never accumulate).
    peft_model, tokenizer = build_lora_model()
    optimizer = build_optimizer(peft_model)

    train_one_epoch(peft_model, tokenizer, [EXAMPLE], optimizer)
    some_param = next(p for p in peft_model.parameters() if p.requires_grad)
    step_after_first = optimizer.state[some_param]["step"].item()

    train_one_epoch(peft_model, tokenizer, [EXAMPLE], optimizer)
    step_after_second = optimizer.state[some_param]["step"].item()

    assert step_after_second > step_after_first


def test_build_labels_masks_exactly_the_prompt_span() -> None:
    _, tokenizer = build_lora_model()
    prompt_text = "prefix text ending here"
    target_text = '{"effect": "lowpass", "freq_hz": 1000, "q": 0.7}'
    full_text = prompt_text + target_text + tokenizer.eos_token

    input_ids, labels = _build_labels(tokenizer, prompt_text, full_text)
    offsets = tokenizer(full_text, return_offsets_mapping=True)["offset_mapping"]

    assert input_ids.shape == labels.shape
    masked_count = 0
    for i, (start, _end) in enumerate(offsets):
        if start < len(prompt_text):
            assert labels[0, i].item() == -100
            masked_count += 1
        else:
            assert labels[0, i].item() == input_ids[0, i].item()
    assert masked_count > 0
    assert masked_count < input_ids.shape[-1]  # some tokens (the target) remain unmasked


def test_save_and_load_lora_adapter_roundtrip(tmp_path) -> None:
    peft_model, tokenizer = build_lora_model()
    optimizer = build_optimizer(peft_model)
    train_one_epoch(peft_model, tokenizer, [EXAMPLE], optimizer)

    adapter_path = tmp_path / "adapter"
    save_lora_adapter(peft_model, adapter_path)

    peft_model.eval()
    with torch.no_grad():
        before = generate_with_model("Instruction: make the bass louder", peft_model, tokenizer, max_new_tokens=20)

    loaded_model, loaded_tokenizer = load_lora_adapter(adapter_path)
    loaded_model.eval()
    with torch.no_grad():
        after = generate_with_model(
            "Instruction: make the bass louder", loaded_model, loaded_tokenizer, max_new_tokens=20
        )

    assert before == after
