#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"


export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p - "$@" <<PY
import runpy
import sys

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401

extra_args = sys.argv[1:]
sys.argv = [
    "train.py",
    "--task",
    "Dex5-IndexNearSurface-Direct-v0",
    "--num_envs",
    "256",
    "--max_iterations",
    "1000",
    "--headless",
    *extra_args,
]
runpy.run_path("${ISAACLAB_ROOT}/scripts/reinforcement_learning/skrl/train.py", run_name="__main__")
PY
