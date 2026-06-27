#!/usr/bin/env python
"""Rephrase + validate user turns along each axis with GPT-4.1-mini (cached, resumable).

Transforms BOTH the training user turns and the eval-probe user turns (the latter give the
test-register columns of the train x test matrix).

    python scripts/02_make_transforms.py --config configs/default.yaml
"""
import argparse

from brittle_user_tokens.data.schema import Example, EvalProbe
from brittle_user_tokens.openai_client import OpenAIClient, OpenAIConfig
from brittle_user_tokens.transforms import (
    ValidateThresholds,
    get_axis,
    rephrase_examples,
    validate_axis,
)
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import read_jsonl, write_jsonl


def _as_examples(rows):
    return [Example(id=str(r["id"]), user=r["user"], assistant=r.get("assistant", "")) for r in rows]


def _transform_split(client, examples, axis, cfg, desc, freeze=False):
    temp = get(cfg, "openai.temperature_rephrase", 0.7)
    rewrites = rephrase_examples(
        client, examples, axis, temperature=temp, max_tokens=512, freeze_mcq_options=freeze
    )
    thr = ValidateThresholds()
    results = validate_axis(
        client, [e.id for e in examples], [e.user for e in examples], rewrites, axis, thr
    )
    kept = sum(r.keep for r in results)
    print(f"[transform] {desc}:{axis.name} kept {kept}/{len(results)} "
          f"(content_ok={sum(r.content_ok for r in results)}, manip_ok={sum(r.manip_ok for r in results)})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default="configs/default.yaml",
                    help="one or more YAML files; later override earlier (e.g. default.yaml + a per-model file)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--axes", nargs="*", default=None,
                    help="only (re)build these axes; default = all configured")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    ds = get(cfg, "data.dataset")
    data_dir = get(cfg, "paths.data_dir", "data")
    base = f"{data_dir}/base/{ds}"
    out = f"{data_dir}/transforms/{ds}"

    client = OpenAIClient(OpenAIConfig.from_dict(get(cfg, "openai", {})))
    is_mcq = ds == "arc_sycophancy"
    freeze = is_mcq and bool(get(cfg, "transforms.freeze_mcq_options", True))
    only = set(a.axes) if a.axes else None

    # ---- training user turns: all configured axes (skip MCQ-unsafe ones) ----
    if a.split in ("train", "both"):
        train_rows = read_jsonl(f"{base}/train.jsonl")
        train_ex = _as_examples(train_rows)
        for name in get(cfg, "transforms.axes", []):
            if only and name not in only:
                continue
            axis = get_axis(name)
            if is_mcq and not axis.content_safe_for_mcq:
                print(f"[transform] skip train arm {name!r} on MCQ data (would inject answer content)")
                continue
            res = _transform_split(client, train_ex, axis, cfg, "train", freeze=freeze)
            write_jsonl(f"{out}/train_{name}.jsonl", [r.to_dict() for r in res])

    # ---- eval probe user turns: the test-register axes ----
    if a.split in ("eval", "both"):
        eval_rows = read_jsonl(f"{base}/eval.jsonl")
        eval_ex = [Example(id=str(r["id"]), user=r["user"], assistant="") for r in eval_rows]
        for name in get(cfg, "eval.test_axes", ["original"]):
            if only and name not in only:
                continue
            axis = get_axis(name)
            res = _transform_split(client, eval_ex, axis, cfg, "eval", freeze=freeze)
            write_jsonl(f"{out}/eval_{name}.jsonl", [r.to_dict() for r in res])

    print(f"[transform] openai stats: {client.stats()}")


if __name__ == "__main__":
    main()
