#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
sha256sum --check artifacts/o1_23/manifest.sha256

output_dir="${DEX5_REPRO_OUTPUT:-${PROJECT_ROOT}/outputs/reproduction/o1_23_offline}"
mkdir -p "${output_dir}"
python isaac/analyze_o1_23_transition_failure_manifold.py \
  --uniform artifacts/o1_23/trajectories/o1_23_A_uniform_transition_256x120.npz \
  --curriculum artifacts/o1_23/trajectories/o1_23_B_entry_curriculum_transition_256x120.npz \
  --output "${output_dir}/O1_23_Transition_Failure_Manifold_Audit.json"
python scripts/verify_o1_23_reproduction.py \
  --actual "${output_dir}/O1_23_Transition_Failure_Manifold_Audit.json" \
  --expected reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_23_Transition_Failure_Manifold_Audit.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_o1_23_transition_failure_manifold.py
