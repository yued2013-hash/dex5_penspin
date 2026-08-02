#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/linkerhand/桌面/deep learning"
ISAACLAB_ROOT="/home/linkerhand/IsaacLab"
TASK="Dex5-PenSpin-PrivilegedOracle-Direct-v0"

source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p - "$@" <<PY
import runpy
import sys

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401

forwarded = sys.argv[1:]
defaults = ["--headless"]
if "--task" not in forwarded:
    defaults += ["--task", "${TASK}"]
if "--num_envs" not in forwarded:
    defaults += ["--num_envs", "256"]
if "--max_iterations" not in forwarded:
    defaults += ["--max_iterations", "30"]
sys.argv = ["train.py", *defaults, *forwarded]
runpy.run_path(
    "/home/linkerhand/IsaacLab/scripts/reinforcement_learning/skrl/train.py",
    run_name="__main__",
)
PY
