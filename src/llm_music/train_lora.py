"""LoRA fine-tuning for the proposed (audio+text -> EQ/Reverb params) model.

Attaches a LoRA adapter (via `peft`) to a *separate* copy of the baseline LLM
(never llm_music.baseline_model's cached singleton — training must not mutate
the model used for baseline-vs-proposed comparisons) and fine-tunes it on a
llm_music.dataset JSON Lines dataset: each example's (instruction,
audio_features) becomes the prompt, in the same "Audio features: {json}\\n
Instruction: ..." format llm_music.proposed_model uses for inference, and its
target_params (as compact JSON) becomes the target the model is trained to
generate. Loss is computed only on the target tokens (the prompt is masked
out), the standard instruction-tuning pattern.

Only plain LoRA is implemented, not QLoRA (4-bit quantization via
bitsandbytes): bitsandbytes has little to no working CPU support on Windows,
and this project runs on a GPU-less Windows machine. The proposal allows
either; LoRA alone is sufficient here.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch

from llm_music.baseline_model import MODEL_NAME, SYSTEM_PROMPT, disable_sampling_generation_defaults
from llm_music.llm_input import format_for_llm_input

DEFAULT_TARGET_MODULES: Tuple[str, ...] = ("q_proj", "v_proj")


def build_lora_model(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
):
    """Load a fresh copy of the baseline LLM and wrap it with a trainable LoRA adapter.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout applied within the LoRA layers.
        target_modules: Names of the linear layers to attach LoRA to (the
            attention query/value projections, by default).

    Returns:
        Tuple of (peft_model, tokenizer). peft_model has the base weights
        frozen and only the LoRA adapter parameters trainable.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    disable_sampling_generation_defaults(base_model)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, config)
    return peft_model, tokenizer


def _build_training_pair(example: Dict[str, Any]) -> Tuple[str, str]:
    """Build (user_content, target_text) for one llm_music.dataset example.

    user_content matches llm_music.proposed_model.build_user_content's format
    exactly, so a trained adapter sees the same prompt shape at inference time
    that it was trained on.
    """
    features_json = format_for_llm_input(example["audio_features"])
    user_content = f"Audio features: {features_json}\nInstruction: {example['instruction']}"
    target_text = json.dumps(example["target_params"], sort_keys=True)
    return user_content, target_text


def build_optimizer(peft_model, learning_rate: float = 1e-4) -> torch.optim.Optimizer:
    """Build an AdamW optimizer over peft_model's trainable (LoRA) parameters.

    Build this once per training run and pass the SAME instance to every
    train_one_epoch call across epochs. Creating a fresh optimizer per epoch
    would reset Adam's per-parameter momentum/variance estimates each time,
    defeating their purpose across a multi-epoch run.

    Args:
        peft_model: A LoRA-wrapped causal LM, as returned by build_lora_model.
        learning_rate: AdamW learning rate.

    Returns:
        An AdamW optimizer over peft_model's trainable parameters.
    """
    return torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=learning_rate
    )


def _build_labels(tokenizer, prompt_text: str, full_text: str):
    """Tokenize full_text once and mask labels covering prompt_text's span.

    Tokenizing prompt_text and full_text separately (then using the standalone
    prompt's token count as a split point) is unsafe: BPE tokenization is not
    strictly prefix-stable, so the prompt's tokenization in isolation can
    differ from how that same text is tokenized as a prefix of full_text,
    misaligning the label boundary. Instead, full_text is tokenized once, and
    each token's character offset (from the fast tokenizer) is compared
    against len(prompt_text) to decide whether it belongs to the prompt.

    Returns:
        Tuple of (input_ids, labels), each shape (1, seq_len). Tokens whose
        span starts before len(prompt_text) are masked (-100); the rest keep
        their input_ids as labels, so loss is computed only on text at or
        after that point (the target, plus EOS).
    """
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"][0].tolist()

    prompt_char_len = len(prompt_text)
    labels = input_ids.clone()
    for i, (start, _end) in enumerate(offsets):
        if start < prompt_char_len:
            labels[0, i] = -100

    return input_ids, labels


def train_one_epoch(
    peft_model,
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
) -> List[float]:
    """Run one epoch of LoRA fine-tuning (one gradient step per example).

    Loss is computed only on the target_params JSON tokens — the system and
    user message are masked out of the loss — so the model is trained to
    produce the target, not to reproduce its own input.

    Args:
        peft_model: A LoRA-wrapped causal LM, as returned by build_lora_model.
        tokenizer: The matching tokenizer.
        examples: Training examples in llm_music.dataset format (each with
            "instruction", "audio_features", "target_params").
        optimizer: An optimizer over peft_model's trainable parameters (e.g.
            from build_optimizer). Pass the same instance across multiple
            calls so its state persists across epochs.

    Returns:
        Per-example training loss values, in example order.
    """
    if not examples:
        raise ValueError("examples must not be empty")

    peft_model.train()

    losses = []
    for example in examples:
        user_content, target_text = _build_training_pair(example)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target_text + tokenizer.eos_token

        input_ids, labels = _build_labels(tokenizer, prompt_text, full_text)

        outputs = peft_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def save_lora_adapter(peft_model, path: Union[str, Path]) -> None:
    """Save the LoRA adapter weights (not the frozen base model) to `path`.

    Args:
        peft_model: A LoRA-wrapped model, as returned by build_lora_model.
        path: Output directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(path))


def load_lora_adapter(path: Union[str, Path]):
    """Load a previously saved LoRA adapter onto a fresh copy of the base model.

    Args:
        path: Directory previously written by save_lora_adapter.

    Returns:
        Tuple of (peft_model, tokenizer).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    disable_sampling_generation_defaults(base_model)
    peft_model = PeftModel.from_pretrained(base_model, str(path))
    return peft_model, tokenizer
