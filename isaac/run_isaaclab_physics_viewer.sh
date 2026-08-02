#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"
TASK="${DEX5_TASK:-Dex5-RingTouch-Direct-v0}"
CHECKPOINT="${DEX5_CHECKPOINT:-${DEX5_PROJECT_ROOT}/reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/ring_touch_best_agent.pt}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p \
  "${DEX5_PROJECT_ROOT}/isaac/play_dex5_policy.py" \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  "$@"
