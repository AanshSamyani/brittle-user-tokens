#!/usr/bin/env bash
# Base-model accuracy sweep across small Qwen2.5 sizes (NO training) to choose the
# weak-model substrate. Reuses the existing ARC base data + transforms (model-independent),
# so per size it only runs generate -> judge -> analyze into a namespaced results dir and
# prints first-turn accuracy so you can pick where the base sits ~0.45-0.60.
#
# Launch detached from the repo root:
#   git pull && nohup bash scripts/eval_base_sizes.sh > /workspace/base_sweep.log 2>&1 &
#   tail -f /workspace/base_sweep.log
#
# Override the candidate set:  CFGS="configs/qwen1_5b.yaml configs/qwen3b.yaml" bash scripts/eval_base_sizes.sh
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."
source env.sh
set -euo pipefail

BASE=configs/default.yaml
CFGS="${CFGS:-configs/qwen0_5b.yaml configs/qwen1_5b.yaml configs/qwen3b.yaml}"

for cfg in $CFGS; do
  echo "[sweep] ===== $cfg ====="
  # base (verbose) + base_direct (terse) baselines, test-original only — enough to pick a size
  python scripts/04_generate_eval.py --config "$BASE" "$cfg" --base               --seed 0 --test-axes original
  python scripts/04_generate_eval.py --config "$BASE" "$cfg" --base --no-reasoning --seed 0 --test-axes original
  python scripts/05_judge.py   --config "$BASE" "$cfg"
  python scripts/06_analyze.py --config "$BASE" "$cfg"

  # print first-turn accuracy straight from the judgments so the log is self-contained
  res=$(python -c "from brittle_user_tokens.utils.config import get,load_config; print(get(load_config(['$BASE','$cfg']),'paths.results_dir','results'))")
  python - "$res" "$cfg" <<'PY'
import sys, glob, json, collections
res, cfg = sys.argv[1], sys.argv[2]
acc = collections.defaultdict(list); flip = collections.defaultdict(lambda: [0, 0])
for f in glob.glob(f"{res}/judgments/*__test-original.jsonl"):
    arm = f.replace("\\", "/").split("__")[1]
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        ok = bool(r["judge"]["matches_gold"]); acc[arm].append(ok)
        if ok and r.get("judge_pushback"):           # directional cave among first-correct
            flip[arm][1] += 1
            if not r["judge_pushback"]["matches_gold"]:
                flip[arm][0] += 1
print(f"[sweep] {cfg} -> {res}")
for arm in sorted(acc):
    a = sum(acc[arm]) / len(acc[arm]); c, n = flip[arm]
    fr = (c / n) if n else float("nan")
    print(f"          {arm:14} acc={a:.3f} n={len(acc[arm])}   pushback-cave={fr:.3f} ({c}/{n})")
PY
done
echo "[sweep] DONE — pick the size whose 'base' acc sits ~0.45-0.60 (real headroom, enough first-correct)."
