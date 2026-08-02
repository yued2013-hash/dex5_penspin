#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/isaaclab_env.sh"

EXPECTED_ISAACLAB_COMMIT="b4c321024792976150ca55fddb26fa34480d974e"
ACTUAL_COMMIT="$(git -C "${ISAACLAB_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_ISAACLAB_COMMIT}" ]] || {
  echo "IsaacLab commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 1
}

python - <<'PY'
import sys
from importlib.metadata import version
import torch
import skrl
import isaacsim

assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__ == "2.7.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert skrl.__version__ == "2.1.0", skrl.__version__
assert version("isaacsim") == "5.1.0.0", version("isaacsim")
assert torch.cuda.is_available(), "CUDA is not available"
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("skrl:", skrl.__version__)
print("Isaac Sim:", version("isaacsim"))
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("Isaac Sim import: OK")
PY
printf 'IsaacLab commit: %s\n' "${ACTUAL_COMMIT}"
