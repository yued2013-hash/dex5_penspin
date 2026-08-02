#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
ENV_NAME="${DEX5_CONDA_ENV:-env_isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${PROJECT_ROOT}/.deps/IsaacLab}"
ISAACLAB_COMMIT="b4c321024792976150ca55fddb26fa34480d974e"

find_conda_base() {
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

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
  bash "${PROJECT_ROOT}/scripts/install_git_lfs_user.sh"
  git -C "${PROJECT_ROOT}" lfs pull
fi

CONDA_BASE="$(find_conda_base)" || {
  echo "Conda is required. Install Miniconda, then rerun bootstrap.sh." >&2
  exit 1
}
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/environment.yml"
fi
conda activate "${ENV_NAME}"

python -m pip install PySocks==1.7.1
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --timeout 120 --retries 30 --resume-retries 30
python -m pip install 'isaacsim[all,extscache]==5.1.0.0' \
  --extra-index-url https://pypi.nvidia.com \
  --timeout 120 --retries 30 --resume-retries 30

mkdir -p "$(dirname "${ISAACLAB_ROOT}")"
if [[ ! -d "${ISAACLAB_ROOT}/.git" ]]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_ROOT}"
fi
git -C "${ISAACLAB_ROOT}" fetch origin "${ISAACLAB_COMMIT}"
git -C "${ISAACLAB_ROOT}" checkout --detach "${ISAACLAB_COMMIT}"
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES "${ISAACLAB_ROOT}/isaaclab.sh" --install
python -m pip install skrl==2.1.0 matplotlib==3.10.3 pytest==8.4.2

DEX5_PROJECT_ROOT="${PROJECT_ROOT}" ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
  bash "${PROJECT_ROOT}/scripts/verify_environment.sh"
