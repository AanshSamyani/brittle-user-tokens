"""LoRA SFT with assistant-only loss masking (plain HF Trainer for full control)."""
from __future__ import annotations

import os
from typing import Optional

from ..data.schema import Example
from ..utils.seed import set_seed
from .masking import build_supervised_example, collate_supervised


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        name, torch.bfloat16
    )


class _ListDataset:
    def __init__(self, feats):
        self.feats = feats

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]


def train_lora(
    examples: list[Example],
    out_dir: str,
    *,
    model_name: str,
    lora_cfg: dict,
    train_cfg: dict,
    max_seq_len: int = 1024,
    dtype: str = "bfloat16",
    attn_impl: str = "flash_attention_2",
    system: Optional[str] = None,
    seed: int = 0,
) -> str:
    import torch  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    feats = [build_supervised_example(tok, e.user, e.assistant, max_seq_len, system=system) for e in examples]
    feats = [f for f in feats if any(l != -100 for l in f["labels"])]
    if not feats:
        raise RuntimeError("No trainable examples (all assistant spans empty after masking).")

    def _load(attn):
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_torch_dtype(dtype), attn_implementation=attn
        )

    try:
        model = _load(attn_impl)
    except Exception:
        model = _load("eager")

    gck = bool(train_cfg.get("gradient_checkpointing", True))
    if gck:
        model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    if gck:
        model.enable_input_require_grads()  # required for grad checkpointing + PEFT
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 2)),
        learning_rate=float(train_cfg.get("lr", 1e-5)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        bf16=bool(train_cfg.get("bf16", True)),
        save_strategy="no",
        report_to=[],
        seed=seed,
        gradient_checkpointing=False,  # enabled manually above
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_ListDataset(feats),
        data_collator=lambda b: collate_supervised(b, tok.pad_token_id),
    )
    trainer.train()

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
