#!/usr/bin/env bash
set -euo pipefail

PROJECT_DISCOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DISCOVERY_ROOT}/scripts/isaaclab_env.sh"

ROOT="${DEX5_PROJECT_ROOT}"
REPORT_DIR="${ROOT}/reports/m3_2_30_index_near_surface/p1_p2_reset_bank/reverse_curriculum"
TASK="Dex5-P1P2-ReverseR0ValidatedHoldSlow-Direct-v0"
SPLIT="train"
EPISODES="1024"
NUM_ENVS="128"
MAX_STEPS="800"
SEED_BASE="20260900"
LABEL=""
RUN_DIR=""
CHECKPOINTS=()

usage() {
    cat <<'EOF'
Usage:
  run_sweep_p1_p2_checkpoints.sh --run_dir RUN_DIR [options] [-- CHECKPOINT ...]

Options:
  --task TASK             Isaac Lab task id.
  --split SPLIT           train, validation, or test.
  --episodes N            Completed episodes required per checkpoint.
  --num_envs N            Parallel envs for evaluation.
  --max_steps N           Max simulator steps per evaluation.
  --seed_base N           Seed for the first checkpoint; increments by one.
  --label LABEL           Label used in output filenames.

If no explicit CHECKPOINT list is passed after --, all agent_*.pt files in
RUN_DIR/checkpoints are evaluated, sorted by their numeric suffix.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --seed_base)
            SEED_BASE="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            CHECKPOINTS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${RUN_DIR}" ]]; then
    echo "--run_dir is required" >&2
    usage >&2
    exit 2
fi

if [[ -z "${LABEL}" ]]; then
    LABEL="$(basename "${RUN_DIR}")"
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    mapfile -t CHECKPOINTS < <(
        find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name 'agent_*.pt' \
            -printf '%f\n' \
            | sed -E 's/^agent_([0-9]+)\.pt$/\1 &/' \
            | sort -n \
            | awk '{print "'"${RUN_DIR}"'/checkpoints/" $2}'
    )
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No checkpoints found under ${RUN_DIR}/checkpoints" >&2
    exit 1
fi

mkdir -p "${REPORT_DIR}"
index=0
for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    name="$(basename "${checkpoint}" .pt)"
    seed=$((SEED_BASE + index))
    output="${REPORT_DIR}/p1_p2_reverse_r0_validated_100ms_${LABEL}_${name}_${SPLIT}${EPISODES}_eval.json"
    bash "${ROOT}/isaac/run_evaluate_p1_p2_policy.sh" \
        --task "${TASK}" \
        --checkpoint "${checkpoint}" \
        --output "${output}" \
        --split "${SPLIT}" \
        --num_envs "${NUM_ENVS}" \
        --episodes "${EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        --seed "${seed}"
    index=$((index + 1))
done

python "${ROOT}/isaac/summarize_p1_p2_reverse_curriculum.py" \
    --min_episodes "${EPISODES}" \
    --task_contains "ReverseR0ValidatedHoldSlow" \
    --output_json "${REPORT_DIR}/P1_P2_validated_R0_100ms_eval_summary.json" \
    --output_md "${REPORT_DIR}/P1_P2严格validated_R0_100ms评估汇总.md"
