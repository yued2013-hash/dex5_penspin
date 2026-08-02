#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

PROJECT_ROOT="${DEX5_PROJECT_ROOT}"
DEEP_LEARNING_ROOT="${DEX5_PROJECT_PARENT}"


export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${DEEP_LEARNING_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/isaac/generate_canonical_visual_gallery.py" \
  --headless \
  --render-backend software \
  "$@"
