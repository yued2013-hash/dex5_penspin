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
./isaaclab.sh -p - "$@" <<'PY'
import runpy
import sys

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401

extra_args = sys.argv[1:]
sys.argv = [
    "train.py",
    "--task",
    "Dex5-IndexNearSurface-Cache-Direct-v0",
    "--num_envs",
    "256",
    "--max_iterations",
    "10",
    "--headless",
    *extra_args,
]
runpy.run_path("/home/linkerhand/IsaacLab/scripts/reinforcement_learning/skrl/train.py", run_name="__main__")
PY
