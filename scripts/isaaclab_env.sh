#!/usr/bin/env bash
# Shared runtime contract for every Isaac Lab wrapper in this repository.

if [[ -z "${DEX5_PROJECT_ROOT:-}" ]]; then
  DEX5_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export DEX5_PROJECT_ROOT
export DEX5_PROJECT_PARENT="$(dirname "${DEX5_PROJECT_ROOT}")"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-${DEX5_PROJECT_ROOT}/.deps/IsaacLab}"
export ISAACLAB_LOG_ROOT="${ISAACLAB_LOG_ROOT:-${ISAACLAB_ROOT}/logs/skrl}"
export DEX5_ARTIFACT_ROOT="${DEX5_ARTIFACT_ROOT:-${DEX5_PROJECT_ROOT}/artifacts}"
export DEX5_CONDA_ENV="${DEX5_CONDA_ENV:-env_isaaclab}"
export DEX5_SDK_ROOT="${DEX5_SDK_ROOT:-}"

_dex5_find_conda() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return
  fi
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda /opt/miniconda3; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

if [[ "${DEX5_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  CONDA_BASE="$(_dex5_find_conda)" || {
    echo "Conda was not found. Run ./bootstrap.sh or set DEX5_SKIP_CONDA_ACTIVATE=1." >&2
    return 1 2>/dev/null || exit 1
  }
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${DEX5_CONDA_ENV}"
fi

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${DEX5_PROJECT_PARENT}:${DEX5_PROJECT_ROOT}/isaac:${PYTHONPATH:-}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi
