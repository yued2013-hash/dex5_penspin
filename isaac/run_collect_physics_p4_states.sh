#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linkerhand/桌面/deep learning"
source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then export TERM=xterm; fi

cd /home/linkerhand/IsaacLab
./isaaclab.sh -p "${ROOT}/dex5_penspin/isaac/collect_physics_p4_states.py" --headless "$@"
