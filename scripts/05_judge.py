#!/usr/bin/env python
"""Judge generations with GPT-4.1-mini (cached). Runs anywhere (no GPU).

    python scripts/05_judge.py --config configs/default.yaml
"""
import argparse
import glob
import os

from brittle_user_tokens.eval.judge import judge_batch
from brittle_user_tokens.openai_client import OpenAIClient, OpenAIConfig
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import read_jsonl, write_jsonl


def _items(recs, response_key="response"):
    return [
        {
            "user": r["user"],
            "gold_answer": r.get("gold_answer"),
            "asserted_belief": r.get("asserted_belief"),
            "response": r.get(response_key, "") or "",
        }
        for r in recs
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--glob", default=None, help="override generations glob")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    ds = get(cfg, "data.dataset")
    task = get(cfg, "eval.judge_task", "sycophancy")
    res = get(cfg, "paths.results_dir", "results")
    client = OpenAIClient(OpenAIConfig.from_dict(get(cfg, "openai", {})))

    pattern = a.glob or f"{res}/generations/{ds}__*.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No generations matched {pattern} — run scripts/04 first.")

    for f in files:
        recs = read_jsonl(f)
        name = os.path.basename(f)
        for r, jj in zip(recs, judge_batch(client, task, _items(recs), desc=f"judge:{name}")):
            r["judge"] = jj
        if task == "sycophancy" and any("response_pushback" in r for r in recs):
            pb = judge_batch(client, task, _items(recs, "response_pushback"), desc=f"judge-pb:{name}")
            for r, jj in zip(recs, pb):
                r["judge_pushback"] = jj
        out = f"{res}/judgments/{name}"
        write_jsonl(out, recs)
        print(f"[judge] {out}")

    print(f"[judge] openai stats: {client.stats()}")


if __name__ == "__main__":
    main()
