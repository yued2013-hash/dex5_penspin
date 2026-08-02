#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linkerhand/桌面/deep learning"
ISAACLAB_ROOT="/home/linkerhand/IsaacLab"
TASK="${DEX5_TASK:-Dex5-RingTouch-Direct-v0}"
CHECKPOINT="${DEX5_CHECKPOINT:-${ROOT}/dex5_penspin/reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/ring_touch_best_agent.pt}"

source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p \
  "${ROOT}/dex5_penspin/isaac/play_dex5_policy.py" \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  "$@"
