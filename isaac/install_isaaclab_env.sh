#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-env_isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/linkerhand/IsaacLab}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

PY="/home/linkerhand/anaconda3/envs/${ENV_NAME}/bin/python"

"${PY}" -m pip install --upgrade pip

# Official Isaac Lab pip route for Isaac Sim 5.x / Python 3.11.
# Use high resume retries because the NVIDIA/PyTorch wheels are large and
# network interruptions are common.
"${PY}" -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --resume-retries 30 \
  --timeout 120 \
  --progress-bar on

"${PY}" -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com \
  --resume-retries 30 \
  --timeout 120 \
  --progress-bar on

if [ ! -d "${ISAACLAB_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git --branch main "${ISAACLAB_DIR}"
fi

(
  cd "${ISAACLAB_DIR}"
  ./isaaclab.sh --install
)

"${PY}" - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
for name in ["isaacsim", "isaaclab"]:
    print(name, "OK" if importlib.util.find_spec(name) else "NO")
PY
