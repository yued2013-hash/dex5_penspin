#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PROJECT_ROOT}/..:${PROJECT_ROOT}/isaac:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/smoke_s0_1_hardware_wrapper.py" --headless "$@"
