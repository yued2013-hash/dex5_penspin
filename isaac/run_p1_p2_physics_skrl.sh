#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"
TASK="${DEX5_TASK:-Dex5-P1P2-C1-Direct-v0}"

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
defaults = ["--task", "${TASK}", "--headless"]
if "--num_envs" not in forwarded:
    defaults += ["--num_envs", "128"]
if "--max_iterations" not in forwarded:
    defaults += ["--max_iterations", "250"]
sys.argv = ["train.py", *defaults, *forwarded]
runpy.run_path(
    "${ISAACLAB_ROOT}/scripts/reinforcement_learning/skrl/train.py",
    run_name="__main__",
)
PY
