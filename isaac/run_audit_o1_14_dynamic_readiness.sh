#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linkerhand/桌面/deep learning"
PROJECT_ROOT="${ROOT}/dex5_penspin"
REPORT_DIR="${PROJECT_ROOT}/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"

source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PROJECT_ROOT}/isaac:${ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/audit_o1_14_dynamic_readiness.py" \
  --headless \
  --output "${REPORT_DIR}/O1_14_Dynamic_Probe_Responses.npz" \
  "$@"
python "${PROJECT_ROOT}/isaac/summarize_o1_14_dynamic_readiness.py"
