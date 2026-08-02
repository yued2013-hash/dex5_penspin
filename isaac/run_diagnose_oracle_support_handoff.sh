#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_PARENT}"
PROJECT_ROOT="${DEX5_PROJECT_ROOT}"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/diagnose_oracle_support_handoff.py" --headless "$@"
