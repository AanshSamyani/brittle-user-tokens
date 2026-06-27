#!/usr/bin/env bash
# Re-run neutral_paraphrase with FROZEN MCQ options + add the base-model (no-finetune)
# baseline, then judge, analyze, and push results.
#
# Launch detached from the repo root:
#   git pull && nohup bash scripts/rerun_neutral_base.sh > /workspace/rerun.log 2>&1 &
#   tail -f /workspace/rerun.log
#
# Override the sweep:  SEEDS="0 1 2 3" bash scripts/rerun_neutral_base.sh
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."   # repo root
source env.sh                                 # activate venv + caches (set-u-safe)
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
SEEDS="${SEEDS:-0 1 2}"
echo "[rerun] start $(date) | config=$CONFIG | seeds=$SEEDS"

# 1) rebuild ONLY the neutral training transform, now with the A)/B)/C)/D) block frozen
python scripts/02_make_transforms.py --config "$CONFIG" --split train --axes neutral_paraphrase

# 2) retrain + regenerate neutral across seeds (overwrites the old neutral adapters/outputs)
for s in $SEEDS; do
  echo "[rerun] === neutral seed $s ==="
  python scripts/03_train.py         --config "$CONFIG" --arm neutral_paraphrase --seed "$s"
  python scripts/04_generate_eval.py --config "$CONFIG" --arm neutral_paraphrase --seed "$s"
done

# 3) base-model baseline — no LoRA (greedy decode is deterministic, so one seed is enough)
echo "[rerun] === base model (no finetune) ==="
python scripts/04_generate_eval.py --config "$CONFIG" --base --seed 0

# 4) judge + aggregate (cached judgments for unchanged arms are reused)
python scripts/05_judge.py   --config "$CONFIG"
python scripts/06_analyze.py --config "$CONFIG"

# 5) push results back (results/ + the new neutral transform; runs/ stays gitignored)
git add -f data/transforms/arc_sycophancy/train_neutral_paraphrase.jsonl results/
git -c commit.gpgsign=false commit -m "Re-run neutral (frozen MCQ options) + base-model baseline" \
  || echo "[rerun] nothing new to commit"
git push || echo "[rerun] push failed — push manually"

echo "[rerun] ALL DONE $(date)"
