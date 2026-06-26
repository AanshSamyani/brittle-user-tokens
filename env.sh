#!/usr/bin/env bash
# Source this before EVERY session:   source env.sh
#
# On the GPU server only /workspace persists across restarts, so all caches, the uv venv,
# and uv itself live under /workspace. This file re-points everything there and activates
# the venv created by scripts/setup_workspace.sh.
#
# SECRETS DO NOT GO IN THIS FILE (it is tracked by git). Put them in env.local.sh, which is
# gitignored and sourced below:
#     echo 'export OPENAI_API_KEY=sk-...' > env.local.sh
#     # optional, only for gated HF models:  echo 'export HF_TOKEN=hf_...' >> env.local.sh

# repo root (works whether this file is sourced or executed)
BUT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BUT_ROOT

# ----------------------------- secrets (gitignored) --------------------------
if [ -f "$BUT_ROOT/env.local.sh" ]; then
  # shellcheck disable=SC1091
  source "$BUT_ROOT/env.local.sh"
fi

# ----------------------------- persistent workspace --------------------------
export WORKSPACE="${WORKSPACE:-/workspace}"          # the only dir that survives restarts
export UV_INSTALL_DIR="$WORKSPACE/bin"
export PATH="$WORKSPACE/bin:${PATH:-}"
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
export UV_PROJECT_ENVIRONMENT="$BUT_ROOT/.venv"
export PYTHONPATH="$BUT_ROOT/src:${PYTHONPATH:-}"

# activate the uv venv if it exists (created by scripts/setup_workspace.sh)
if [ -f "$BUT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$BUT_ROOT/.venv/bin/activate"
fi

echo "[env] WORKSPACE=$WORKSPACE venv=$([ -n "${VIRTUAL_ENV:-}" ] && echo active || echo MISSING) key=$([ -n "${OPENAI_API_KEY:-}" ] && echo set || echo MISSING)"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[env] WARNING: OPENAI_API_KEY not set — run: echo 'export OPENAI_API_KEY=sk-...' > env.local.sh"
fi
