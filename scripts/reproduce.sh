#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:-help}"
shift || true

case "${stage}" in
  env)
    bash "${PROJECT_ROOT}/scripts/verify_environment.sh" "$@"
    ;;
  offline)
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/scripts/isaaclab_env.sh"
    bash "${PROJECT_ROOT}/scripts/smoke_offline.sh" "$@"
    ;;
  isaac-smoke)
    bash "${PROJECT_ROOT}/isaac/run_smoke_privileged_oracle.sh" \
      --num-envs 16 --steps 8 "$@"
    ;;
  rollout)
    bash "${PROJECT_ROOT}/scripts/reproduce_o1_23_rollout.sh" "$@"
    ;;
  all)
    bash "${PROJECT_ROOT}/scripts/reproduce.sh" env
    bash "${PROJECT_ROOT}/scripts/reproduce.sh" offline
    bash "${PROJECT_ROOT}/scripts/reproduce.sh" isaac-smoke "$@"
    ;;
  *)
    cat <<'EOF'
Usage: ./scripts/reproduce.sh STAGE

Stages:
  env          verify all locked versions and CUDA
  offline      verify LFS assets and reproduce O1.23 offline analysis
  isaac-smoke  create the O1 oracle environment and step 16 environments
  rollout      regenerate the two 256-environment O1.23 rollouts
  all          run env + offline + isaac-smoke
EOF
    ;;
esac
