#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linkerhand/桌面/deep learning"
ISAACLAB_ROOT="/home/linkerhand/IsaacLab"

source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p \
  /home/linkerhand/桌面/deep\ learning/dex5_penspin/isaac/search_canonical_p3_physx_projection.py \
  --headless "$@"
