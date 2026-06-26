#!/usr/bin/env bash
# End-to-end flagship grid on the GPU box.
#   source env.sh && bash scripts/run_all.sh [configs/default.yaml]
# Override the sweep with env vars, e.g.:  SEEDS="0 1 2 3 4" ARMS="original assertive" bash scripts/run_all.sh
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-original neutral_paraphrase assertive polite_deferential declarative warm}"

echo "[run_all] config=$CONFIG"
echo "[run_all] seeds=$SEEDS"
echo "[run_all] arms=$ARMS"

# 1-2: data + transforms (OpenAI, cached) — cheap, no GPU
python scripts/01_build_dataset.py   --config "$CONFIG"
python scripts/02_make_transforms.py --config "$CONFIG"

# 3-4: train + generate per (arm, seed) — GPU
for seed in $SEEDS; do
  for arm in $ARMS; do
    python scripts/03_train.py         --config "$CONFIG" --arm "$arm" --seed "$seed"
    python scripts/04_generate_eval.py --config "$CONFIG" --arm "$arm" --seed "$seed"
  done
done

# 5-6: judge + analyze (OpenAI, cached)
python scripts/05_judge.py   --config "$CONFIG"
python scripts/06_analyze.py --config "$CONFIG"

echo "[run_all] done. See results/metrics/summary.json"
