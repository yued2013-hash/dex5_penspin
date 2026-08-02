#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"
PROJECT_ROOT="${DEX5_PROJECT_ROOT}"
REPORT_DIR="${PROJECT_ROOT}/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PROJECT_ROOT}/isaac:${ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/audit_o1_16_local_handoff_basin.py" \
  --closed-loop-audit \
  --bank-ids 1155 \
  --horizons 18 \
  --population 128 \
  --headless \
  --output "${REPORT_DIR}/O1_17_Closed_Loop_Basin_Raw.json" \
  "$@"
python "${PROJECT_ROOT}/isaac/summarize_o1_17_closed_loop_basin.py"
