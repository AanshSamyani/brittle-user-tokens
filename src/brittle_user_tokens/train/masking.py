"""Chat-template tokenization with assistant-only loss masking (single-turn).

The project invariant lives here: for a given example the *assistant* tokens (the ones with
label != -100) are identical across transform arms; only the user (masked, label == -100)
tokens differ. `count_supervised_tokens` lets a test assert exactly that.

Masking strategy: tokenize the full [system?, user, assistant] chat, and the prompt-only
[system?, user] + generation prompt; everything in the prompt prefix is set to -100. This
is template-agnostic (works for ChatML/Qwen, Llama-3, etc.) and verified by a prefix check.
"""
from __future__ import annotations

from typing import Optional

IGNORE = -100


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_supervised_example(
    tokenizer,
    user: str,
    assistant: str,
    max_len: int = 1024,
    system: Optional[str] = None,
) -> dict:
    """Return {input_ids, labels, attention_mask} with loss only on assistant tokens."""
    prompt_msgs = []
    if system:
        prompt_msgs.append({"role": "system", "content": system})
    prompt_msgs.append({"role": "user", "content": user})
    full_msgs = prompt_msgs + [{"role": "assistant", "content": assistant}]

    full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)

    # prompt_ids is normally an exact prefix of full_ids; fall back to longest common prefix.
    plen = len(prompt_ids)
    if full_ids[:plen] != prompt_ids:
        plen = _common_prefix_len(full_ids, prompt_ids)

    labels = list(full_ids)
    for i in range(min(plen, len(labels))):
        labels[i] = IGNORE

    if max_len and len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        labels = labels[:max_len]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def count_supervised_tokens(example: dict) -> int:
    """Number of tokens that contribute to the loss (label != -100)."""
    return sum(1 for x in example["labels"] if x != IGNORE)


def collate_supervised(batch: list[dict], pad_token_id: int):
    """Right-pad a batch of build_supervised_example() dicts into torch tensors."""
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [IGNORE] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }
