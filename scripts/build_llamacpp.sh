#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-b6500}"
DEST="${2:-tools/llama.cpp}"
if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$DEST"
fi
cd "$DEST"
if ! command -v cmake >/dev/null 2>&1; then
  python3 -m pip install -q cmake
fi
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target llama-bench llama-quantize -j "$(nproc)"
if [ ! -x .venv_gguf/bin/python ]; then
  python3 -m pip install -q uv
  uv venv .venv_gguf --python 3.11 --seed
  uv pip install --python .venv_gguf/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv_gguf/bin/python "numpy~=1.26.4" "sentencepiece~=0.2.0" "transformers==4.55.4" "mistral-common>=1.8.3" "protobuf>=4.21.0,<5.0.0" ./gguf-py
fi
echo "llama.cpp ready in $DEST"
