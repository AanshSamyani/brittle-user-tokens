"""Canonical run/artifact names so scripts can find each other's outputs.

run        : {dataset}__{arm}__seed{seed}
generation : {dataset}__{arm}__seed{seed}__test-{test_axis}.jsonl
(dataset and arm may contain single underscores; '__' is the field separator.)
"""
from __future__ import annotations


def run_name(dataset: str, arm: str, seed: int) -> str:
    return f"{dataset}__{arm}__seed{seed}"


def gen_name(dataset: str, arm: str, seed: int, test_axis: str) -> str:
    return f"{run_name(dataset, arm, seed)}__test-{test_axis}"


def parse_name(name: str) -> dict:
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    test_axis = None
    if "__test-" in name:
        name, test_axis = name.split("__test-", 1)
    ds, arm, seed_part = name.split("__")
    seed = int(seed_part.replace("seed", ""))
    return {"dataset": ds, "arm": arm, "seed": seed, "test_axis": test_axis}
