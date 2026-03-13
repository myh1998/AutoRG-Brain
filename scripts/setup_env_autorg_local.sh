#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-autorg-brain}
PY_VER=${2:-3.10}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Reusing conda env: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# Keep torch from cluster base if preinstalled; otherwise install from your internal mirror.
python - <<'PY'
import importlib
for m in ["torch"]:
    try:
        importlib.import_module(m)
        print(f"[OK] {m} already available")
    except Exception:
        print(f"[WARN] {m} missing")
PY

pip install numpy scipy scikit-image scikit-learn simpleitk nibabel tqdm pyyaml
pip install transformers==4.41.2 sentencepiece
pip install batchgenerators einops nltk bert-score torchinfo medpy

echo "[DONE] Environment ready: ${ENV_NAME}"
