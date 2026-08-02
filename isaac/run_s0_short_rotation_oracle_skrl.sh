#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"
PROJECT="${DEX5_PROJECT_ROOT}"
TASK="Dex5-S0-ShortRotation-Direct-v0"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT}:${PROJECT}/isaac:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi

cd "$ISAACLAB_ROOT"
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
    "${ISAACLAB_ROOT}/scripts/reinforcement_learning/skrl/train.py",
    run_name="__main__",
)
PY
