from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dex5_penspin.isaac.canonical_reset_schema import save_npz_atomic, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "Dex5_Sim2Real_S0/reset_distribution/validation/"
    "canonical_reset_bank/P3_validated.npz"
)
OUTPUT_DIR = ROOT / "Dex5_Sim2Real_S0/hardware_compatible_p3_reset_bank"
OUTPUT = OUTPUT_DIR / "Dex5_S0_HW_P3_Reset_Bank.npz"
EXPECTED_O0_HASH = "73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65"
O0 = ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
)
PAIRS = ((2, 3), (6, 7), (10, 11), (14, 15))


def _split_isolated(values: np.ndarray, splits: np.ndarray) -> bool:
    groups: dict[int, set[str]] = {}
    for value, split in zip(values, splits, strict=True):
        groups.setdefault(int(value), set()).add(str(split))
    return all(len(group_splits) == 1 for group_splits in groups.values())


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    count = int(payload["robot_qpos"].shape[0])
    required_labels = (
        "labels_r0_pass",
        "labels_r1_pass",
        "labels_r2_pass",
        "labels_r3a_pass",
        "labels_r3b_pass",
    )
    finite_keys = (
        "robot_qpos",
        "robot_qvel",
        "robot_q_target",
        "object_pose",
        "object_linear_velocity",
        "object_angular_velocity",
        "contact_force",
        "s0_projection_residual",
        "s0_logical_actuator_target",
        "s0_contact_wrench_force_w",
        "s0_contact_wrench_torque_w",
    )
    finite = all(np.isfinite(np.asarray(payload[key])).all() for key in finite_keys)
    target = np.asarray(payload["robot_q_target"], dtype=np.float64)
    continuation = np.asarray(payload["robot_continuation_q_target"], dtype=np.float64)
    coupling_error = max(
        max(
            float(np.max(np.abs(target[:, first] - target[:, second]))),
            float(
                np.max(
                    np.abs(
                        continuation[:, :, first] - continuation[:, :, second]
                    )
                )
            ),
        )
        for first, second in PAIRS
    )
    contacts = np.asarray(payload["contact_state"], dtype=np.bool_)
    tim_final = contacts[:, :3].all(axis=1)
    two_tim_final = contacts[:, :3].sum(axis=1) >= 2
    source_penetration = np.asarray(
        payload["labels_source_max_penetration_m"], dtype=np.float64
    )
    baseline_penetration = np.asarray(
        payload["labels_baseline_max_penetration_m"], dtype=np.float64
    )
    max_penetration_mm = 1000.0 * float(
        max(source_penetration.max(initial=0.0), baseline_penetration.max(initial=0.0))
    )
    splits = np.asarray(payload["metadata_split"])
    checks = {
        "validated_states_at_least_500": count >= 500,
        "all_r0_r3_labels": all(
            bool(np.asarray(payload[key], dtype=np.bool_).all())
            for key in required_labels
        ),
        "finite_required_arrays": finite,
        "target_coupling_exact": coupling_error <= 1.0e-7,
        "tim_three_finger_contact_at_least_90pct": float(tim_final.mean()) >= 0.90,
        "at_least_two_tim_contact_at_least_99pct": float(two_tim_final.mean()) >= 0.99,
        "penetration_below_0_5mm": max_penetration_mm < 0.5,
        "canonical_is_p3": np.unique(payload["metadata_canonical_id"]).tolist()
        == ["P3"],
        "source_trajectory_split_isolation": _split_isolated(
            payload["metadata_source_trajectory_id"], splits
        ),
        "original_state_split_isolation": _split_isolated(
            payload["metadata_s0_original_state_id"], splits
        ),
        "o0_hash_unchanged": sha256_file(O0) == EXPECTED_O0_HASH,
    }
    if not all(checks.values()):
        raise RuntimeError(f"S0 HW P3 finalization failed: {checks}")

    save_npz_atomic(OUTPUT, payload)
    report = {
        "status": "PASS",
        "artifact": "Dex5_S0_HW_P3_Reset_Bank",
        "scope": "hardware-compatible P3 reset distribution, no policy training",
        "validated_states": count,
        "unique_original_p3_states": int(
            np.unique(payload["metadata_s0_original_state_id"]).size
        ),
        "unique_source_trajectories": int(
            np.unique(payload["metadata_source_trajectory_id"]).size
        ),
        "split_counts": {
            split: int((splits == split).sum())
            for split in ("train", "validation", "test")
        },
        "tim_final_contact_rate": float(tim_final.mean()),
        "at_least_two_tim_final_contact_rate": float(two_tim_final.mean()),
        "max_penetration_mm": max_penetration_mm,
        "max_target_coupling_error_rad": coupling_error,
        "projection_residual": {
            "mean_rms_rad": float(
                np.asarray(payload["s0_projection_residual_rms_rad"]).mean()
            ),
            "max_rad": float(
                np.asarray(payload["s0_projection_residual_max_rad"]).max()
            ),
        },
        "r3_protocol": {
            "unperturbed_trials": 3,
            "required_unperturbed_success_ratio": 1.0,
            "perturbed_trials": 5,
            "required_perturbed_r1_ratio": 0.8,
            "required_perturbed_r2_ratio": 0.8,
        },
        "checks": checks,
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
        "frozen_o0_sha256": sha256_file(O0),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "Dex5_S0_HW_P3_Reset_Bank.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Dex5 S0 Hardware-Compatible P3 Reset Bank",
        "",
        "**状态：PASS**",
        "",
        f"- R0-R3 有效状态：`{count}`",
        f"- 原始 O0 P3 parent：`{report['unique_original_p3_states']}`",
        f"- 源轨迹：`{report['unique_source_trajectories']}`",
        f"- 最终 TIM 三指接触率：`{100.0 * report['tim_final_contact_rate']:.2f}%`",
        f"- 最终至少两指接触率：`{100.0 * report['at_least_two_tim_final_contact_rate']:.2f}%`",
        f"- 最大 settle/验证穿透：`{max_penetration_mm:.3f} mm`",
        f"- 最大 target 耦合误差：`{coupling_error:.3e} rad`",
        "",
        "该 bank 是冻结 O0 P3 的独立派生资产。它不能回写或替代 O0，也不证明 short-rotation oracle 已训练成功。",
    ]
    (OUTPUT_DIR / "Dex5_S0_HW_P3_Reset_Bank.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
