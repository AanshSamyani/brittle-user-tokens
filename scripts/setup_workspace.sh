#!/usr/bin/env bash
# One-time setup on a fresh GPU server. Installs EVERYTHING under /workspace (the only
# directory that persists): uv, the venv, and all caches. Re-run after a server wipe.
#
#   cd /workspace/brittle-user-tokens && bash scripts/setup_workspace.sh
#
# For a different CUDA build of torch:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup_workspace.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

echo "[setup] WORKSPACE=$WORKSPACE  repo=$HERE"
mkdir -p "$WORKSPACE/bin" "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"

# 1. uv -> /workspace/bin (persists)
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv into $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
uv --version

# 2. venv (uv-managed Python 3.11) -> repo/.venv (under /workspace, persists)
echo "[setup] creating venv at $HERE/.venv"
uv venv "$HERE/.venv" --python 3.11
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"

# 3. torch with the right CUDA for the H100, then the project + deps
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
echo "[setup] installing torch from $TORCH_INDEX"
uv pip install torch --index-url "$TORCH_INDEX"
echo "[setup] installing project + deps (uv pip install -e '.[dev,viz]')"
uv pip install -e ".[dev,viz]"

# 4. verify
echo "[setup] verifying"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import transformers, peft, trl, datasets, accelerate, openai, hf_transfer; print('deps OK')"
python - <<'PY'
import os
for k in ("HF_HOME", "UV_CACHE_DIR", "VIRTUAL_ENV"):
    print(f"  {k:14s}= {os.environ.get(k)}")
PY

echo
echo "[setup] done. New sessions just need:  source env.sh"
echo "[setup] optional FlashAttention-2:     uv pip install flash-attn --no-build-isolation"
