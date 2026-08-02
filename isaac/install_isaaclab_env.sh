#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "install_isaaclab_env.sh is retained as a compatibility entry point."
exec bash "${PROJECT_ROOT}/bootstrap.sh" "$@"
