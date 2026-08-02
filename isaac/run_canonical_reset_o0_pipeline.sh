#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

PROJECT_ROOT="${DEX5_PROJECT_ROOT}"
O0_ROOT="reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0"
VALIDATOR="${PROJECT_ROOT}/isaac/run_validate_canonical_reset_o0.sh"

cd "${PROJECT_ROOT}"

python isaac/build_canonical_reset_candidates.py

# C1 needs a two-pass O0-only filter: find reproducible parents, then cover a
# small mass/friction neighborhood and validate every generated state again.
bash "${VALIDATOR}" \
  --canonical-id C1 \
  --input "${O0_ROOT}/candidate/C1_candidate.npz" \
  --num-envs 256 \
  --max-candidates 1000 \
  --r3a-trials 10 \
  --r3b-trials 20 \
  --progress-every 25

python isaac/expand_canonical_physics_variants.py \
  --input "${O0_ROOT}/validation_reports/C1_audited.npz" \
  --output "${O0_ROOT}/candidate/C1_robust_physics_variants_candidate.npz" \
  --variants 6

bash "${VALIDATOR}" \
  --canonical-id C1 \
  --input "${O0_ROOT}/candidate/C1_robust_physics_variants_candidate.npz" \
  --num-envs 256 \
  --max-candidates 0 \
  --r3a-trials 10 \
  --r3b-trials 20 \
  --progress-every 25

for canonical_id in C2 C3; do
  bash "${VALIDATOR}" \
    --canonical-id "${canonical_id}" \
    --input "${O0_ROOT}/candidate/${canonical_id}_candidate.npz" \
    --num-envs 256 \
    --max-candidates 1000 \
    --r3a-trials 10 \
    --r3b-trials 20 \
    --progress-every 25
done

for canonical_id in C4 C5; do
  bash "${VALIDATOR}" \
    --canonical-id "${canonical_id}" \
    --input "${O0_ROOT}/candidate/${canonical_id}_candidate.npz" \
    --num-envs 256 \
    --max-candidates 0 \
    --r3a-trials 10 \
    --r3b-trials 20 \
    --progress-every 25
done

# C6 is a strict thumb-index pinch. Validate the raw capture states with the
# canonical-specific contact gate, then expand only validated parents across a
# small mass/friction neighborhood and validate every variant again.
bash "${VALIDATOR}" \
  --canonical-id C6 \
  --input "${O0_ROOT}/candidate/C6_candidate.npz" \
  --num-envs 256 \
  --max-candidates 0 \
  --r3a-trials 10 \
  --r3b-trials 20 \
  --progress-every 25

python isaac/expand_canonical_physics_variants.py \
  --input "${O0_ROOT}/canonical_reset_bank/C6_validated.npz" \
  --output "${O0_ROOT}/candidate/C6_thumb_index_physics_variants_candidate.npz" \
  --variants 8

bash "${VALIDATOR}" \
  --canonical-id C6 \
  --input "${O0_ROOT}/candidate/C6_thumb_index_physics_variants_candidate.npz" \
  --num-envs 256 \
  --max-candidates 0 \
  --r3a-trials 10 \
  --r3b-trials 20 \
  --progress-every 25

python isaac/finalize_canonical_reset_o0.py
