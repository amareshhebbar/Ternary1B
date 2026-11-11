#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_ROOT_USER_ACTION=ignore

if command -v apt-get >/dev/null 2>&1 && ! command -v tmux >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq tmux >/dev/null) || echo "tmux install skipped"
fi

python3 -m pip install -q --upgrade uv

PY=3.11
SYS_VER=$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$SYS_VER" in 3.10|3.11|3.12) PY=$(command -v python3) ;; esac

if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python "$PY" --seed
fi
uv pip install --python .venv/bin/python -r requirements.lock

if [ ! -f .env ]; then
  cp .env.example .env
fi
mkdir -p "${TQ_WORK:-/workspace/ternary1b_runs}" 2>/dev/null || true

echo
echo "setup complete"
echo "1. put your Hugging Face write token in .env (HF_TOKEN=...)"
echo "2. source .venv/bin/activate"
echo "3. python run.py check"
echo "4. tmux new -s st, then python run.py pipeline"
