#!/usr/bin/env python
"""LoRA SFT for one (arm, seed). [GPU]

    python scripts/03_train.py --config configs/default.yaml --arm assertive --seed 0

id_policy (config: train.id_policy):
  fallback     (default) — train on ALL base ids; use the arm's rewrite where it passed
                 validation, else the original user text. Equal-sized + paired across arms,
                 never empty. Logs the per-arm application rate (fraction actually transformed).
  intersection — train only on ids that passed validation in EVERY configured axis (so every
                 arm is fully transformed). Stricter; can be tiny/empty when yields differ.
"""
import argparse
import os

from brittle_user_tokens.data.schema import Example
from brittle_user_tokens.train.sft_lora import train_lora
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import read_jsonl
from brittle_user_tokens.utils.naming import run_name


def _kept_map(path: str) -> dict:
    return {str(r["id"]): r["new_user"] for r in read_jsonl(path) if r.get("keep")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default="configs/default.yaml",
                    help="one or more YAML files; later override earlier (e.g. default.yaml + a per-model file)")
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
    policy = get(cfg, "train.id_policy", "fallback")

    base_rows = read_jsonl(f"{base}/train.jsonl")
    target = {str(r["id"]): r["assistant"] for r in base_rows}
    base_user = {str(r["id"]): r["user"] for r in base_rows}

    arm_path = f"{tdir}/train_{a.arm}.jsonl"
    if not os.path.exists(arm_path):
        raise SystemExit(f"No transform file {arm_path}; run scripts/02 (or arm skipped on this dataset).")
    arm_map = _kept_map(arm_path)

    if policy == "intersection":
        common = set(target)
        for name in get(cfg, "transforms.axes", []):
            p = f"{tdir}/train_{name}.jsonl"
            if os.path.exists(p):
                common &= set(_kept_map(p))
        ids = sorted(common)
        if not ids:
            raise SystemExit("intersection id set is empty; use train.id_policy=fallback or raise yields.")
        users = {i: arm_map[i] for i in ids}
    else:  # fallback
        ids = sorted(target)
        users = {i: arm_map.get(i, base_user[i]) for i in ids}

    applied = sum(1 for i in ids if i in arm_map)
    examples = [Example(id=i, user=users[i], assistant=target[i]) for i in ids]
    pct = 100 * applied / max(1, len(examples))
    print(f"[train] ds={ds} arm={a.arm} seed={seed} policy={policy} "
          f"n={len(examples)} transformed={applied}/{len(examples)} ({pct:.0f}%)")

    out_dir = f"{get(cfg, 'paths.runs_dir', 'runs')}/{run_name(ds, a.arm, seed)}"
    train_lora(
        examples, out_dir,
        model_name=get(cfg, "model.name"), lora_cfg=get(cfg, "lora"), train_cfg=get(cfg, "train"),
        max_seq_len=get(cfg, "model.max_seq_len", 1024), dtype=get(cfg, "model.dtype", "bfloat16"),
        attn_impl=get(cfg, "model.attn_implementation", "flash_attention_2"), seed=seed,
    )
    print(f"[train] saved adapter -> {out_dir}")


if __name__ == "__main__":
    main()
