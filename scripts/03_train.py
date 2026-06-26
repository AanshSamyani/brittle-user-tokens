#!/usr/bin/env python
"""LoRA SFT for one (arm, seed) on the matched-id set. [GPU]

    python scripts/03_train.py --config configs/default.yaml --arm assertive --seed 0

All arms train on the SAME set of example ids (the intersection of ids that passed
validation across every configured axis) so downstream comparisons are paired.
"""
import argparse
import os

from brittle_user_tokens.data.schema import Example
from brittle_user_tokens.train.sft_lora import train_lora
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import read_jsonl
from brittle_user_tokens.utils.naming import run_name


def _load_axis_map(path: str) -> dict:
    return {str(r["id"]): r["new_user"] for r in read_jsonl(path) if r.get("keep")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed = a.seed if a.seed is not None else get(cfg, "seed", 0)

    ds = get(cfg, "data.dataset")
    data_dir = get(cfg, "paths.data_dir", "data")
    base = f"{data_dir}/base/{ds}"
    tdir = f"{data_dir}/transforms/{ds}"

    base_rows = read_jsonl(f"{base}/train.jsonl")
    target = {str(r["id"]): r["assistant"] for r in base_rows}

    maps = {}
    for name in get(cfg, "transforms.axes", []):
        p = f"{tdir}/train_{name}.jsonl"
        if os.path.exists(p):
            maps[name] = _load_axis_map(p)
    if not maps:
        raise SystemExit("No transform files found — run scripts/02_make_transforms.py first.")
    if a.arm not in maps:
        raise SystemExit(f"Arm {a.arm!r} has no transform file (skipped on this dataset?).")

    common = None
    for m in maps.values():
        ids = set(m) & set(target)
        common = ids if common is None else (common & ids)
    common = sorted(common)
    if not common:
        raise SystemExit("Matched-id set is empty across arms; loosen validation or check transforms.")

    arm_map = maps[a.arm]
    examples = [Example(id=i, user=arm_map[i], assistant=target[i]) for i in common]
    print(f"[train] ds={ds} arm={a.arm} seed={seed} matched_n={len(examples)}")

    out_dir = f"{get(cfg, 'paths.runs_dir', 'runs')}/{run_name(ds, a.arm, seed)}"
    train_lora(
        examples,
        out_dir,
        model_name=get(cfg, "model.name"),
        lora_cfg=get(cfg, "lora"),
        train_cfg=get(cfg, "train"),
        max_seq_len=get(cfg, "model.max_seq_len", 1024),
        dtype=get(cfg, "model.dtype", "bfloat16"),
        attn_impl=get(cfg, "model.attn_implementation", "flash_attention_2"),
        seed=seed,
    )
    print(f"[train] saved adapter -> {out_dir}")


if __name__ == "__main__":
    main()
