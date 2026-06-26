#!/usr/bin/env bash
# Source this before EVERY session:   source env.sh
#
# On the GPU server only /workspace persists across restarts, so all caches, the uv venv,
# and uv itself live under /workspace. This file re-points everything there and activates
# the venv created by scripts/setup_workspace.sh.

# ----------------------------- SECRETS (fill in) -----------------------------
export OPENAI_API_KEY="sk-REPLACE_ME"      # <-- put your OpenAI key here
# export HF_TOKEN="hf_..."                  # only needed for gated HF models
# Keep your key out of git after editing:  git update-index --skip-worktree env.sh

# ----------------------------- persistent workspace --------------------------
export WORKSPACE="${WORKSPACE:-/workspace}"          # the only dir that survives restarts

# uv (and any installed tools) live in /workspace/bin so they persist
export UV_INSTALL_DIR="$WORKSPACE/bin"
export PATH="$WORKSPACE/bin:$PATH"

# all caches under /workspace: HF weights (~15GB) + datasets, uv packages, compile caches
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.local/share/uv/python"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WORKSPACE/.cache/torchinductor"
export MPLCONFIGDIR="$WORKSPACE/.cache/matplotlib"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# ----------------------------- project ---------------------------------------
BUT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BUT_ROOT
export UV_PROJECT_ENVIRONMENT="$BUT_ROOT/.venv"
export PYTHONPATH="$BUT_ROOT/src:${PYTHONPATH:-}"

# activate the uv venv if it exists (created by scripts/setup_workspace.sh)
if [ -f "$BUT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$BUT_ROOT/.venv/bin/activate"
fi

echo "[env] WORKSPACE=$WORKSPACE BUT_ROOT=$BUT_ROOT venv=$([ -n "${VIRTUAL_ENV:-}" ] && echo active || echo MISSING)"
if [ "${OPENAI_API_KEY}" = "sk-REPLACE_ME" ]; then
  echo "[env] WARNING: OPENAI_API_KEY is still the placeholder — edit env.sh."
fi
