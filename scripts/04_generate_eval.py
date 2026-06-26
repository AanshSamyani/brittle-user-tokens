#!/usr/bin/env python
"""Generate eval responses across test-register axes for one (arm, seed). [GPU]

    python scripts/04_generate_eval.py --config configs/default.yaml --arm assertive --seed 0
"""
import argparse
import os

from brittle_user_tokens.data.schema import EvalProbe
from brittle_user_tokens.data.sycophancy import build_pushback_followup
from brittle_user_tokens.eval.generate import generate_for_probes, load_for_generation
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import read_jsonl, write_jsonl
from brittle_user_tokens.utils.naming import gen_name, run_name


def _load_eval_axis_map(path: str) -> dict:
    return {str(r["id"]): r["new_user"] for r in read_jsonl(path) if r.get("keep")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--test-axes", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed = a.seed if a.seed is not None else get(cfg, "seed", 0)

    ds = get(cfg, "data.dataset")
    data_dir = get(cfg, "paths.data_dir", "data")
    res = get(cfg, "paths.results_dir", "results")
    probes = [EvalProbe.from_dict(r) for r in read_jsonl(f"{data_dir}/base/{ds}/eval.jsonl")]
    test_axes = a.test_axes or get(cfg, "eval.test_axes", ["original"])

    run_dir = f"{get(cfg, 'paths.runs_dir', 'runs')}/{run_name(ds, a.arm, seed)}"
    if not os.path.isdir(run_dir):
        raise SystemExit(f"No adapter at {run_dir}; train first (scripts/03_train.py).")

    model, tok = load_for_generation(
        get(cfg, "model.name"), run_dir,
        get(cfg, "model.dtype", "bfloat16"), get(cfg, "model.attn_implementation", "flash_attention_2"),
    )
    is_syc = ds == "arc_sycophancy"
    pushback = bool(get(cfg, "eval.pushback", False)) and is_syc
    followup = build_pushback_followup if is_syc else None
    tdir = f"{data_dir}/transforms/{ds}"

    for taxis in test_axes:
        p = f"{tdir}/eval_{taxis}.jsonl"
        if os.path.exists(p):
            m = _load_eval_axis_map(p)
            user_texts = [m.get(pr.id, pr.user) for pr in probes]
        else:
            user_texts = [pr.user for pr in probes]
        recs = generate_for_probes(
            model, tok, probes, user_texts=user_texts,
            max_new_tokens=get(cfg, "eval.max_new_tokens", 256),
            temperature=get(cfg, "eval.temperature", 0.0),
            batch_size=get(cfg, "eval.batch_size", 32),
            # pushback flip-rate only needs the original register -> halves eval cost
            pushback=pushback and (taxis == "original"),
            build_followup=followup,
        )
        out = f"{res}/generations/{gen_name(ds, a.arm, seed, taxis)}.jsonl"
        write_jsonl(out, recs)
        print(f"[gen] {out} n={len(recs)}")


if __name__ == "__main__":
    main()
