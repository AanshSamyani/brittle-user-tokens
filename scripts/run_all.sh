#!/usr/bin/env bash
# End-to-end flagship grid on the GPU box.
#   source env.sh && bash scripts/run_all.sh [config ...]
# Pass one or more configs (later override earlier), e.g. a per-model override:
#   bash scripts/run_all.sh configs/default.yaml configs/qwen1_5b.yaml
# Override the sweep with env vars:  SEEDS="0 1 2 3 4" ARMS="original assertive" bash scripts/run_all.sh
# Reuse existing model-independent data + transforms (skip 01-02):  SKIP_DATA=1 bash scripts/run_all.sh ...
set -euo pipefail

CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/default.yaml)
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-original neutral_paraphrase assertive polite_deferential declarative warm}"

echo "[run_all] config=${CONFIGS[*]}"
echo "[run_all] seeds=$SEEDS"
echo "[run_all] arms=$ARMS"

# 1-2: data + transforms (OpenAI, cached) — cheap, no GPU. Model-independent, so SKIP_DATA reuses them.
if [ -z "${SKIP_DATA:-}" ]; then
  python scripts/01_build_dataset.py   --config "${CONFIGS[@]}"
  python scripts/02_make_transforms.py --config "${CONFIGS[@]}"
else
  echo "[run_all] SKIP_DATA set — reusing existing data/base + data/transforms"
fi

# 3-4: train + generate per (arm, seed) — GPU
for seed in $SEEDS; do
  for arm in $ARMS; do
    python scripts/03_train.py         --config "${CONFIGS[@]}" --arm "$arm" --seed "$seed"
    python scripts/04_generate_eval.py --config "${CONFIGS[@]}" --arm "$arm" --seed "$seed"
  done
done

# 5-6: judge + analyze (OpenAI, cached)
python scripts/05_judge.py   --config "${CONFIGS[@]}"
python scripts/06_analyze.py --config "${CONFIGS[@]}"

echo "[run_all] done. See <results_dir>/metrics/summary.json"
