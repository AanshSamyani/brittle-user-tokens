#!/usr/bin/env python
"""Aggregate judgments -> metrics, paired bootstrap vs base arm, optional heatmap.

    python scripts/06_analyze.py --config configs/default.yaml
"""
import argparse
import glob
import os
from collections import defaultdict

from brittle_user_tokens.eval import metrics as M
from brittle_user_tokens.utils.config import get, load_config
from brittle_user_tokens.utils.io import ensure_dir, read_jsonl, write_json
from brittle_user_tokens.utils.naming import parse_name


def _judges(recs, key="judge"):
    return [r.get(key, {}) for r in recs]


def _print_syc_table(rates, arms, taxes, base_axis):
    print("\n=== Sycophancy Rate  (rows: train arm, cols: test register) ===")
    header = "train\\test".ljust(20) + "".join(t[:12].ljust(13) for t in taxes)
    print(header)
    for arm in arms:
        row = arm[:19].ljust(20)
        for t in taxes:
            v = rates.get((arm, t))
            row += ("—".ljust(13) if v is None else f"{v:.3f}".ljust(13))
        print(row + ("   <= base" if arm == base_axis else ""))
    print()


def _maybe_heatmap(rates, arms, taxes, ds, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[analyze] matplotlib not installed; skipping heatmap.")
        return
    try:
        grid = np.array([[rates.get((a, t), np.nan) for t in taxes] for a in arms], dtype=float)
        fig, ax = plt.subplots(figsize=(1.6 * len(taxes) + 2, 0.6 * len(arms) + 2))
        im = ax.imshow(grid, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(taxes)), taxes, rotation=30, ha="right")
        ax.set_yticks(range(len(arms)), arms)
        for i in range(len(arms)):
            for j in range(len(taxes)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_title(f"Sycophancy rate — {ds}\n(train register x test register)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        path = f"{res}/metrics/{ds}_sycophancy_heatmap.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"[analyze] heatmap -> {path}")
    except Exception as e:  # plotting must never block the summary
        print(f"[analyze] heatmap skipped ({type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    ds = get(cfg, "data.dataset")
    task = get(cfg, "eval.judge_task", "sycophancy")
    res = get(cfg, "paths.results_dir", "results")
    base_axis = get(cfg, "transforms.base_axis", "original")
    ensure_dir(f"{res}/metrics")

    files = sorted(glob.glob(f"{res}/judgments/{ds}__*.jsonl"))
    if not files:
        raise SystemExit("No judgments found — run scripts/05_judge.py first.")

    groups = defaultdict(list)  # (arm, test_axis) -> pooled recs across seeds
    for f in files:
        meta = parse_name(os.path.basename(f))
        groups[(meta["arm"], meta["test_axis"])].extend(read_jsonl(f))

    summary = {"dataset": ds, "task": task, "base_axis": base_axis, "arms": defaultdict(dict)}

    if task == "sycophancy":
        rates, per_item = {}, {}
        for (arm, taxis), recs in groups.items():
            ind = M.sycophancy_indicator(_judges(recs))
            ids = [str(r["id"]) for r in recs]
            rates[(arm, taxis)] = M.rate(ind)
            per_item[(arm, taxis)] = M.per_item_rate_over_seeds(ids, ind)

        arms = sorted({a for a, _ in rates})
        taxes = sorted({t for _, t in rates})
        for arm in arms:
            for taxis in taxes:
                if (arm, taxis) not in rates:
                    continue
                entry = {"sycophancy_rate": rates[(arm, taxis)]}
                if arm != base_axis and (base_axis, taxis) in per_item:
                    x, y, _ = M.aligned_arrays(per_item[(arm, taxis)], per_item[(base_axis, taxis)])
                    entry["vs_base"] = M.paired_bootstrap_diff(x, y)
                recs = groups[(arm, taxis)]
                if any("judge_pushback" in r for r in recs):
                    pairs = [(r.get("judge", {}).get("matches_gold"),
                              r.get("judge_pushback", {}).get("matches_gold"))
                             for r in recs if "judge_pushback" in r]
                    first = [bool(p[0]) for p in pairs]
                    second = [bool(p[1]) for p in pairs]
                    entry["pushback_flip_rate"] = M.flip_rate(first, second)
                summary["arms"][arm][taxis] = entry

        _print_syc_table(rates, arms, taxes, base_axis)
        _maybe_heatmap(rates, arms, taxes, ds, res)

    elif task == "persona":
        for (arm, taxis), recs in groups.items():
            summary["arms"][arm][taxis] = {"persona": M.persona_means(_judges(recs))}
    elif task == "alignment":
        for (arm, taxis), recs in groups.items():
            summary["arms"][arm][taxis] = {"misalignment_rate": M.misalignment_rate(_judges(recs))}

    summary["arms"] = {k: dict(v) for k, v in summary["arms"].items()}
    out = f"{res}/metrics/summary.json"
    write_json(out, summary)
    print(f"[analyze] wrote {out}")


if __name__ == "__main__":
    main()
