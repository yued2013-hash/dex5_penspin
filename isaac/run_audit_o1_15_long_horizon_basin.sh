#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"
PROJECT_ROOT="${DEX5_PROJECT_ROOT}"
REPORT_DIR="${PROJECT_ROOT}/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
CANDIDATES="${REPORT_DIR}/O1_14_Dynamic_Handoff_Readiness_Audit.json"
BATCH_SIZE="${O1_15_BATCH_SIZE:-6}"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PROJECT_ROOT}/isaac:${ROOT}:${PYTHONPATH:-}"

mapfile -t BANK_IDS < <(
  python - "${CANDIDATES}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
for bank_id in report["short_admissible_candidate_layer"]["bank_ids"]:
    print(bank_id)
PY
)

for ((start=0, batch=0; start<${#BANK_IDS[@]}; start+=BATCH_SIZE, batch+=1)); do
  chunk=("${BANK_IDS[@]:start:BATCH_SIZE}")
  output="${REPORT_DIR}/O1_15_CEM_batch_$(printf '%02d' "${batch}").json"
  python "${PROJECT_ROOT}/isaac/audit_o1_8_handoff_reachability.py" \
    --headless \
    --states "${#chunk[@]}" \
    --bank-ids "${chunk[@]}" \
    --population 64 \
    --iterations 10 \
    --horizons 18 24 30 \
    --waypoints 3 \
    --initial-std-rad 0.06 \
    --max-offset-rad 0.18 \
    --action-mode all \
    --output "${output}" \
    "$@"
done

python "${PROJECT_ROOT}/isaac/summarize_o1_15_long_horizon_basin.py"
