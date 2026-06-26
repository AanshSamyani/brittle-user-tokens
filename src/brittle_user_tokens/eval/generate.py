"""Batched generation from a (base model + LoRA adapter), with optional pushback turn."""
from __future__ import annotations

from typing import Callable, Optional

from ..data.schema import EvalProbe


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        name, torch.bfloat16
    )


def load_for_generation(model_name: str, adapter_dir: Optional[str] = None,
                        dtype: str = "bfloat16", attn_impl: str = "flash_attention_2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)  # LoRA doesn't change the tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # for batched generation

    def _load(attn):
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_torch_dtype(dtype), attn_implementation=attn, device_map="auto"
        )

    try:
        model = _load(attn_impl)
    except Exception:
        model = _load("eager")

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def _gen_batch(model, tok, chats: list[list[dict]], *, max_new_tokens: int, temperature: float,
               batch_size: int) -> list[str]:
    import torch

    outs: list[str] = []
    for i in range(0, len(chats), batch_size):
        batch = chats[i : i + batch_size]
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.decode(t, skip_special_tokens=True).strip() for t in new)
    return outs


def generate_for_probes(
    model,
    tok,
    probes: list[EvalProbe],
    *,
    user_texts: Optional[list[str]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 16,
    pushback: bool = False,
    build_followup: Optional[Callable[[EvalProbe], str]] = None,
) -> list[dict]:
    """Generate a reply per probe. `user_texts` overrides probe.user (e.g. a test-register
    rephrasing); otherwise probe.user is used. Optionally run a pushback second turn."""
    users = user_texts if user_texts is not None else [p.user for p in probes]
    first = _gen_batch(
        model, tok, [[{"role": "user", "content": u}] for u in users],
        max_new_tokens=max_new_tokens, temperature=temperature, batch_size=batch_size,
    )

    records = []
    for p, u, r in zip(probes, users, first):
        records.append({"id": p.id, "user": u, "gold_answer": p.gold_answer,
                        "asserted_belief": p.asserted_belief, "response": r})

    if pushback and build_followup is not None:
        chats2, idx = [], []
        for k, (p, u, r) in enumerate(zip(probes, users, first)):
            chats2.append([
                {"role": "user", "content": u},
                {"role": "assistant", "content": r},
                {"role": "user", "content": build_followup(p)},
            ])
            idx.append(k)
        second = _gen_batch(model, tok, chats2, max_new_tokens=max_new_tokens,
                            temperature=temperature, batch_size=batch_size)
        for k, r2 in zip(idx, second):
            records[k]["response_pushback"] = r2
    return records
