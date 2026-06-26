#!/usr/bin/env bash
# Source this before running anything:   source env.sh
#
# ----------------------------- SECRETS (fill in) -----------------------------
export OPENAI_API_KEY="sk-REPLACE_ME"      # <-- put your OpenAI key here
# export HF_TOKEN="hf_..."                  # only needed for gated HF models
#
# Tip: keep your key out of git after editing this file:
#   git update-index --skip-worktree env.sh
#
# ----------------------------- Project env -----------------------------------
BUT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BUT_ROOT
export PYTHONPATH="${BUT_ROOT}/src:${PYTHONPATH}"

# Optional: make HF downloads faster / quieter
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

echo "[env] BUT_ROOT=${BUT_ROOT}"
if [ "${OPENAI_API_KEY}" = "sk-REPLACE_ME" ]; then
  echo "[env] WARNING: OPENAI_API_KEY is still the placeholder — edit env.sh."
fi
