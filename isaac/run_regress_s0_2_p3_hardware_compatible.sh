#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reports/sim2real_s0_2"
COMMON=(--headless --device cuda:0 --num-envs "${NUM_ENVS:-64}" --steps "${STEPS:-60}")
export TERM="${TERM:-xterm}"
if [[ "$TERM" == "dumb" ]]; then
  export TERM=xterm
fi
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT}/..:${ROOT}/isaac:${PYTHONPATH:-}"

mkdir -p "$OUT"

python "$ROOT/isaac/regress_s0_2_p3_hardware_compatible.py" \
  --task Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0 --mode zero \
  --output "$OUT/baseline_zero.json" "${COMMON[@]}"
python "$ROOT/isaac/regress_s0_2_p3_hardware_compatible.py" \
  --task Dex5-S0.2-P3HardwareCompat-Audit-Direct-v0 --mode zero \
  --output "$OUT/s0_2_zero.json" "${COMMON[@]}"
python "$ROOT/isaac/regress_s0_2_p3_hardware_compatible.py" \
  --task Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0 --mode small \
  --output "$OUT/baseline_small.json" "${COMMON[@]}"
python "$ROOT/isaac/regress_s0_2_p3_hardware_compatible.py" \
  --task Dex5-S0.2-P3HardwareCompat-Audit-Direct-v0 --mode small \
  --output "$OUT/s0_2_small.json" "${COMMON[@]}"

python "$ROOT/sim2real/summarize_s0_2_regression.py"
