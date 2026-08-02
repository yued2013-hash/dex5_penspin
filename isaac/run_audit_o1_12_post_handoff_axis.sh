#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linkerhand/桌面/deep learning"
PROJECT_ROOT="${ROOT}/dex5_penspin"

source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/audit_o1_12_post_handoff_axis.py" --headless "$@"
