#!/usr/bin/env bash
# Run one conversation-style experiment (#1-#4) end-to-end on the GPU box.
#   source env.sh && bash scripts/run_chat.sh configs/persona_contagion.yaml
# Override the sweep: SEEDS="0 1 2" ARMS="original warm" bash scripts/run_chat.sh configs/safety_erosion.yaml
#
# Builds the multi-turn chat data + restyled-user-turn transforms, trains every register arm x seed,
# evaluates (+ an untrained base baseline), judges, and aggregates into the experiment's results dir.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."

EXP="${1:?pass an experiment config, e.g. configs/persona_contagion.yaml}"
CONFIGS="configs/default.yaml $EXP"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-original neutral_paraphrase assertive polite_deferential declarative warm}"
echo "[run_chat] exp=$EXP | seeds=$SEEDS | arms=$ARMS"

# 1-2: chat data (HF) + restyle masked user turns per axis (OpenAI, cached). No GPU.
python scripts/01_build_dataset.py   --config $CONFIGS
python scripts/02_make_transforms.py --config $CONFIGS

# 3-4: train each (arm, seed) + generate eval. GPU.
for s in $SEEDS; do
  for arm in $ARMS; do
    python scripts/03_train.py         --config $CONFIGS --arm "$arm" --seed "$s"
    python scripts/04_generate_eval.py --config $CONFIGS --arm "$arm" --seed "$s"
  done
done

# untrained base baseline (no LoRA) for reference
python scripts/04_generate_eval.py --config $CONFIGS --base --seed 0

# 5-6: judge + aggregate
python scripts/05_judge.py   --config $CONFIGS
python scripts/06_analyze.py --config $CONFIGS

res=$(python -c "from brittle_user_tokens.utils.config import get,load_config; print(get(load_config(['configs/default.yaml','$EXP']),'paths.results_dir'))")
echo "[run_chat] done -> ${res}/metrics/summary.json"
