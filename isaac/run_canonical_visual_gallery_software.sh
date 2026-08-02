#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/linkerhand/桌面/deep learning/dex5_penspin"
DEEP_LEARNING_ROOT="/home/linkerhand/桌面/deep learning"

source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${DEEP_LEARNING_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/generate_canonical_visual_gallery.py" \
  --headless \
  --render-backend software \
  "$@"
