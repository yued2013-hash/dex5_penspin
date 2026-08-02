#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/isaaclab_env.sh"
cd "${PROJECT_ROOT}"
sha256sum --check artifacts/o1_23/manifest.sha256

output_dir="${DEX5_REPRO_OUTPUT:-${PROJECT_ROOT}/outputs/reproduction/o1_23_rollout}"
mkdir -p "${output_dir}"
common=(--task Dex5-PenSpin-Oracle-P3Temporal-Direct-v0 --num-envs 256 --steps 120)

bash isaac/run_diagnose_oracle_support_handoff.sh "${common[@]}" \
  --checkpoint artifacts/o1_23/checkpoints/o1_22_A_uniform_agent_1440.pt \
  --output "${output_dir}/A_uniform.json"
bash isaac/run_diagnose_oracle_support_handoff.sh "${common[@]}" \
  --checkpoint artifacts/o1_23/checkpoints/o1_22_B_entry_curriculum_agent_1440.pt \
  --output "${output_dir}/B_entry_curriculum.json"
python isaac/analyze_o1_23_transition_failure_manifold.py \
  --uniform "${output_dir}/A_uniform.npz" \
  --curriculum "${output_dir}/B_entry_curriculum.npz" \
  --output "${output_dir}/O1_23_Transition_Failure_Manifold_Audit.json"
echo "Rollout reproduction written to ${output_dir}"
