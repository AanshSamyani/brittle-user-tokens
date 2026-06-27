#!/usr/bin/env python
"""Build base (user, assistant) training pairs + held-out eval probes.

    python scripts/01_build_dataset.py --config configs/default.yaml
"""
import argparse

from brittle_user_tokens.data.loaders import CHAT_DATASETS, load_dataset_pairs
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default="configs/default.yaml",
                    help="one or more YAML files; later override earlier (e.g. default.yaml + a per-model file)")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. data.n_train=500")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    ds = get(cfg, "data.dataset")
    n_train = get(cfg, "data.n_train")
    n_eval = get(cfg, "data.n_eval")
    seed = get(cfg, "data.shuffle_seed", 0)
    kw = {}
    if ds == "insecure_code":
        kw["path"] = get(cfg, "data.path")
        kw["eval_bank"] = get(cfg, "data.eval_bank")
    if ds in CHAT_DATASETS:
        kw["train_source"] = get(cfg, "data.train_source", "ultrachat")
        kw["eval_bank"] = get(cfg, "data.eval_bank")
        kw["max_turns"] = get(cfg, "data.max_turns", 6)

    train, probes = load_dataset_pairs(ds, n_train, n_eval, seed, **kw)

    base = f"{get(cfg, 'paths.data_dir', 'data')}/base/{ds}"
    write_jsonl(f"{base}/train.jsonl", [e.to_dict() for e in train])
    write_jsonl(f"{base}/eval.jsonl", [p.to_dict() for p in probes])
    print(f"[build] {ds}: train={len(train)} eval={len(probes)} -> {base}/")


if __name__ == "__main__":
    main()
