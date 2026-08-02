from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from canonical_reset_schema import (
    FINGERTIP_NAMES,
    JOINT_ORDER,
    SCHEMA_VERSION,
    concatenate_payloads,
    save_npz_atomic,
    sha256_file,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0"
)


def _summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def _speed_summary(vectors: np.ndarray) -> dict[str, float]:
    speed = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
    return {
        "min": float(np.min(speed)),
        "median": float(np.median(speed)),
        "p95": float(np.quantile(speed, 0.95)),
        "max": float(np.max(speed)),
        "mean": float(np.mean(speed)),
    }


def _nonfinite_count(payload: dict[str, np.ndarray]) -> int:
    required = (
        "robot_qpos",
        "robot_qvel",
        "robot_q_target",
        "robot_previous_action",
        "robot_action_history_buffer",
        "robot_control_delay_buffer",
        "robot_continuation_q_target",
        "robot_continuation_action",
        "object_pose",
        "object_linear_velocity",
        "object_angular_velocity",
        "object_com",
        "physics_mass",
        "physics_static_friction",
        "physics_dynamic_friction",
        "physics_com_offset",
        "physics_dt",
        "physics_control_dt",
        "physics_contact_offset",
        "physics_rest_offset",
        "labels_r3a_success_ratio",
        "labels_r3b_r1_success_ratio",
        "labels_r3b_r2_success_ratio",
        "labels_baseline_max_penetration_m",
        "labels_baseline_max_linear_speed_m_s",
        "labels_baseline_max_angular_speed_rad_s",
        "labels_baseline_max_drop_m",
        "labels_baseline_max_drift_m",
        "labels_baseline_contact_ratio",
    )
    total = sum(
        int((~np.isfinite(np.asarray(payload[key]))).sum()) for key in required
    )
    source_available = np.asarray(
        payload["labels_source_penetration_available"], dtype=np.bool_
    )
    total += int(
        (~np.isfinite(payload["labels_source_max_penetration_m"][source_available])).sum()
    )
    contact_available = np.asarray(payload["contact_force_available"], dtype=np.bool_)
    total += int((~np.isfinite(payload["contact_force"][contact_available])).sum())
    positions_available = np.asarray(
        payload["contact_positions_available"], dtype=np.bool_
    )
    total += int(
        (~np.isfinite(payload["contact_fingertip_positions"][positions_available])).sum()
    )
    return total


def _split_leakage(payload: dict[str, np.ndarray]) -> list[int]:
    trajectory = np.asarray(payload["metadata_source_trajectory_id"], dtype=np.int64)
    split = np.asarray(payload["metadata_split"])
    return [
        int(value)
        for value in np.unique(trajectory)
        if np.unique(split[trajectory == value]).size != 1
    ]


def _source_integrity(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pairs = np.unique(
        np.stack(
            (
                np.asarray(payload["metadata_source_uri"]),
                np.asarray(payload["metadata_source_sha256"]),
            ),
            axis=1,
        ),
        axis=0,
    )
    all_match = True
    for uri, expected in pairs:
        path = Path(str(uri))
        actual = sha256_file(path) if path.is_file() else "missing"
        matches = actual == str(expected)
        all_match &= matches
        rows.append(
            {
                "uri": str(path),
                "expected_sha256": str(expected),
                "actual_sha256": actual,
                "matches": matches,
            }
        )
    return {"all_match": all_match, "sources": rows}


def _coverage(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    split = np.asarray(payload["metadata_split"])
    contact = np.asarray(payload["contact_state"], dtype=np.float64)
    return {
        "rows": int(payload["metadata_state_id"].shape[0]),
        "unique_source_trajectories": int(
            np.unique(payload["metadata_source_trajectory_id"]).size
        ),
        "split_counts": {
            name: int(np.sum(split == name))
            for name in ("train", "validation", "test")
        },
        "joint_position": {
            "joint_order": list(JOINT_ORDER),
            **_summary(payload["robot_qpos"]),
        },
        "joint_velocity": {
            "joint_order": list(JOINT_ORDER),
            **_summary(payload["robot_qvel"]),
        },
        "object_position": _summary(payload["object_pose"][:, :3]),
        "object_linear_speed_m_s": _speed_summary(
            payload["object_linear_velocity"]
        ),
        "object_angular_speed_rad_s": _speed_summary(
            payload["object_angular_velocity"]
        ),
        "contact_occupancy": {
            name: float(contact[:, index].mean())
            for index, name in enumerate(FINGERTIP_NAMES)
        },
        "physics_ranges": {
            "mass_kg": [
                float(np.min(payload["physics_mass"])),
                float(np.max(payload["physics_mass"])),
            ],
            "static_friction": [
                float(np.min(payload["physics_static_friction"])),
                float(np.max(payload["physics_static_friction"])),
            ],
            "dynamic_friction": [
                float(np.min(payload["physics_dynamic_friction"])),
                float(np.max(payload["physics_dynamic_friction"])),
            ],
        },
    }


def _schema_manifest(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    fields = {}
    rows = int(payload["metadata_state_id"].shape[0])
    for key in sorted(payload):
        array = np.asarray(payload[key])
        fields[key] = {
            "dtype": str(array.dtype),
            "per_state_shape": list(array.shape[1:])
            if array.ndim > 0 and array.shape[0] == rows
            else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "recoverable dynamic-system reset state",
        "contact_position_frame": "world",
        "joint_order": list(JOINT_ORDER),
        "continuation_horizon_control_steps": 8,
        "validation_levels": {
            "R0": "state and metadata integrity",
            "R1": "two-step PhysX reinstantiation",
            "R2": "100 ms static hold or saved-control dynamic continuation",
            "R3a": "ten repeated nominal resets, per-state success >= 90%",
            "R3b": "twenty perturbed resets, R1 >= 90% and R2 >= 80%",
        },
        "fields": fields,
    }


def _gate_row(
    canonical_id: str,
    report: dict[str, Any],
    payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    state_type = str(np.unique(payload["labels_static_or_dynamic"])[0])
    r3a_min = float(np.min(payload["labels_r3a_success_ratio"]))
    r3b_r1_min = float(np.min(payload["labels_r3b_r1_success_ratio"]))
    r3b_r2_min = float(np.min(payload["labels_r3b_r2_success_ratio"]))
    penetration_mm = float(
        np.max(payload["labels_baseline_max_penetration_m"]) * 1000.0
    )
    continuation_rate = float(np.mean(payload["labels_r2_pass"]))
    no_nan = _nonfinite_count(payload) == 0
    gates = {
        "r1_count_ge_500": report["counts"]["r1"] >= 500,
        "r2_count_ge_200": report["counts"]["r2"] >= 200,
        "validated_distribution_continuation_ge_target": continuation_rate
        >= (0.95 if state_type == "static" else 0.90),
        "r3a_min_ge_0_90": r3a_min >= 0.90 - 1.0e-6,
        "r3b_r1_min_ge_0_90": r3b_r1_min >= 0.90 - 1.0e-6,
        "r3b_r2_min_ge_0_80": r3b_r2_min >= 0.80 - 1.0e-6,
        "penetration_le_0_5_mm": penetration_mm <= 0.5 + 1.0e-6,
        "nan_inf_zero": no_nan,
    }
    return {
        "canonical_id": canonical_id,
        "state_type": state_type,
        "candidate_audited": report["candidate_audited"],
        "r1_count": report["counts"]["r1"],
        "r2_count": report["counts"]["r2"],
        "validated_count": int(payload["metadata_state_id"].shape[0]),
        "validated_continuation_rate": continuation_rate,
        "r3a_min_ratio": r3a_min,
        "r3b_r1_min_ratio": r3b_r1_min,
        "r3b_r2_min_ratio": r3b_r2_min,
        "max_validated_penetration_mm": penetration_mm,
        "nonfinite_count_excluding_unavailable_contact_sentinels": _nonfinite_count(
            payload
        ),
        "gates": gates,
        "pass": all(gates.values()),
    }


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# O0 Canonical Reset Distribution 验收报告",
        "",
        f"- 总状态：**{'PASS' if audit['all_gates_pass'] else 'FAIL'}**",
        f"- 统一 validated bank：`{audit['combined']['path']}`",
        f"- 总状态数：`{audit['combined']['rows']}`",
        f"- schema：`{SCHEMA_VERSION}`",
        f"- 源轨迹跨 split 泄漏：`{len(audit['dataset_integrity']['split_leakage_trajectory_ids'])}`",
        f"- 源文件哈希校验：`{'PASS' if audit['dataset_integrity']['source_integrity']['all_match'] else 'FAIL'}`",
        "",
        "## 量化门槛",
        "",
        "| Canonical | 类型 | R1 | R2 | Validated | R3a min | R3b R1/R2 min | 最大穿透 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit["canonical_gates"]:
        lines.append(
            f"| {row['canonical_id']} | {row['state_type']} | {row['r1_count']} | "
            f"{row['r2_count']} | {row['validated_count']} | "
            f"{row['r3a_min_ratio']:.1%} | "
            f"{row['r3b_r1_min_ratio']:.1%}/{row['r3b_r2_min_ratio']:.1%} | "
            f"{row['max_validated_penetration_mm']:.3f} mm | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines.extend(
        (
            "",
            "## 口径",
            "",
            "- C1 是 static canonical，按 100 ms hold 验证。",
            "- C2-C6 是 dynamic canonical，按保存的控制上下文继续 100 ms，不要求接触模式冻结。",
            "- R3a 是同一状态 10 次 nominal reset；R3b 是 20 次局部扰动 reset。",
            "- 表中 continuation 及穿透统计只针对最终 validated distribution；候选集的拒绝信息保留在各 C1-C6 validation report 中。",
            "- 这份报告只验收 O0 reset infrastructure，不代表 oracle/PPO 或完整转笔技能已完成。",
            "",
            "## 数据资产",
            "",
            f"- Schema manifest：`{audit['artifacts']['schema_manifest']}`",
            f"- Coverage JSON：`{audit['artifacts']['coverage_report']}`",
            f"- Machine-readable audit：`{audit['artifacts']['audit_report']}`",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize and audit O0 reset banks.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    banks_dir = root / "canonical_reset_bank"
    reports_dir = root / "validation_reports"

    payloads: list[dict[str, np.ndarray]] = []
    validation_reports: dict[str, dict[str, Any]] = {}
    coverage: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    expected_keys: set[str] | None = None
    for index in range(1, 7):
        canonical_id = f"C{index}"
        bank_path = banks_dir / f"{canonical_id}_validated.npz"
        report_path = reports_dir / f"{canonical_id}_validation_report.json"
        with np.load(bank_path, allow_pickle=False) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        keys = set(payload)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"Schema-key mismatch in {canonical_id}")
        if not np.all(payload["metadata_canonical_id"] == canonical_id):
            raise ValueError(f"Canonical label mismatch in {bank_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        payloads.append(payload)
        validation_reports[canonical_id] = report
        coverage[canonical_id] = _coverage(payload)
        gate_rows.append(_gate_row(canonical_id, report, payload))

    combined = concatenate_payloads(payloads)
    combined_path = banks_dir / "C1_C6_validated.npz"
    save_npz_atomic(combined_path, combined)
    leakage = _split_leakage(combined)
    source_integrity = _source_integrity(combined)
    unique_state_ids = np.unique(combined["metadata_state_id"]).size == combined[
        "metadata_state_id"
    ].shape[0]
    schema_consistent = bool(
        np.all(combined["metadata_state_schema_version"] == SCHEMA_VERSION)
    )
    validated_only = bool(
        np.all(combined["labels_validation_status"] == "validated")
    )

    manifest_path = root / "canonical_reset_schema_manifest.json"
    coverage_path = root / "canonical_reset_coverage_report.json"
    audit_path = root / "O0_canonical_reset_validation_audit.json"
    markdown_path = root / "O0_Canonical_Reset_Distribution_验收报告.md"
    write_json_atomic(manifest_path, _schema_manifest(combined))
    write_json_atomic(
        coverage_path,
        {
            "schema_version": SCHEMA_VERSION,
            "canonical_coverage": coverage,
        },
    )
    integrity_gates = {
        "schema_consistent": schema_consistent,
        "unique_state_ids": unique_state_ids,
        "validated_rows_only": validated_only,
        "source_trajectory_split_leakage_zero": len(leakage) == 0,
        "source_hashes_match": source_integrity["all_match"],
    }
    audit: dict[str, Any] = {
        "phase": "O0 Canonical Reset Distribution",
        "schema_version": SCHEMA_VERSION,
        "canonical_gates": gate_rows,
        "dataset_integrity": {
            "gates": integrity_gates,
            "split_leakage_trajectory_ids": leakage,
            "source_integrity": source_integrity,
        },
        "combined": {
            "path": str(combined_path),
            "rows": int(combined["metadata_state_id"].shape[0]),
            "sha256": sha256_file(combined_path),
        },
        "artifacts": {
            "schema_manifest": str(manifest_path),
            "coverage_report": str(coverage_path),
            "audit_report": str(audit_path),
            "markdown_report": str(markdown_path),
        },
    }
    audit["all_gates_pass"] = all(row["pass"] for row in gate_rows) and all(
        integrity_gates.values()
    )
    write_json_atomic(audit_path, audit)
    markdown_path.write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
