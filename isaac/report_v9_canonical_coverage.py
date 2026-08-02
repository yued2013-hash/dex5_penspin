from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import FINGERTIP_NAMES, JOINT_ORDER, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
BANK = ROOT / "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)


def percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def vector_summary(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def quaternion_error_deg(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1.0e-12)
    reference /= max(np.linalg.norm(reference), 1.0e-12)
    dot = np.abs(quat @ reference).clip(0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    phases = {f"P{row['phase']}": row for row in manifest["phases"]}
    with np.load(BANK, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    rows = []
    for canonical_id in sorted(phases):
        mask = np.asarray(data["metadata_canonical_id"] == canonical_id)
        qpos = np.asarray(data["robot_qpos"][mask], dtype=np.float64)
        qvel = np.asarray(data["robot_qvel"][mask], dtype=np.float64)
        target_offset = np.asarray(
            data["robot_q_target"][mask] - data["robot_qpos"][mask],
            dtype=np.float64,
        )
        object_pose = np.asarray(data["object_pose"][mask], dtype=np.float64)
        linear_velocity = np.asarray(
            data["object_linear_velocity"][mask], dtype=np.float64
        )
        angular_velocity = np.asarray(
            data["object_angular_velocity"][mask], dtype=np.float64
        )
        contacts = np.asarray(data["contact_state"][mask], dtype=np.bool_)
        forces = np.asarray(data["contact_force"][mask], dtype=np.float64)
        contact_rows = {}
        for index, finger in enumerate(FINGERTIP_NAMES):
            active_force = forces[contacts[:, index], index]
            contact_rows[finger] = {
                "rate": float(contacts[:, index].mean()),
                "force_n": percentiles(active_force)
                if active_force.size
                else {"p50": 0.0, "p95": 0.0, "max": 0.0},
            }
        split_values, split_counts = np.unique(
            data["metadata_split"][mask], return_counts=True
        )
        reference_q = np.asarray(phases[canonical_id]["hand_qpos"], dtype=np.float64)
        rows.append(
            {
                "canonical_id": canonical_id,
                "rows": int(mask.sum()),
                "unique_parent_states": int(
                    np.unique(data["metadata_parent_state_id"][mask]).size
                ),
                "unique_source_trajectories": int(
                    np.unique(data["metadata_source_trajectory_id"][mask]).size
                ),
                "split_counts": {
                    str(key): int(value)
                    for key, value in zip(split_values, split_counts, strict=True)
                },
                "joint_qpos": vector_summary(qpos),
                "joint_qpos_rms_to_v9_rad": percentiles(
                    np.sqrt(np.mean((qpos - reference_q) ** 2, axis=1))
                ),
                "joint_qvel_norm_rad_s": percentiles(np.linalg.norm(qvel, axis=1)),
                "target_offset_rms_rad": percentiles(
                    np.sqrt(np.mean(target_offset**2, axis=1))
                ),
                "object_position_m": vector_summary(object_pose[:, :3]),
                "object_position_span_mm": (
                    np.ptp(object_pose[:, :3], axis=0) * 1000.0
                ).tolist(),
                "object_orientation_error_to_v9_deg": percentiles(
                    quaternion_error_deg(
                        object_pose[:, 3:7],
                        np.asarray(phases[canonical_id]["pen_quat"], dtype=np.float64),
                    )
                ),
                "object_linear_speed_m_s": percentiles(
                    np.linalg.norm(linear_velocity, axis=1)
                ),
                "object_angular_speed_rad_s": percentiles(
                    np.linalg.norm(angular_velocity, axis=1)
                ),
                "contacts": contact_rows,
                "physics": {
                    "mass_kg": vector_summary(data["physics_mass"][mask, None]),
                    "static_friction": vector_summary(
                        data["physics_static_friction"][mask, None]
                    ),
                    "dynamic_friction": vector_summary(
                        data["physics_dynamic_friction"][mask, None]
                    ),
                },
            }
        )

    report = {
        "status": "PASS",
        "scope": "O0 frozen-bank baseline for O1 privileged oracle smoke",
        "bank": str(BANK),
        "bank_sha256": sha256_file(BANK),
        "joint_order": list(JOINT_ORDER),
        "rows": rows,
    }
    json_path = ROOT / "P1_P5_V9_canonical_coverage_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P1-P5 V9 Canonical Coverage Report",
        "",
        f"- Frozen bank SHA-256: `{report['bank_sha256']}`",
        "- Scope: O1 smoke initial-distribution baseline",
        "",
        "| Canonical | Rows | Sources | Parents | q RMS p50/p95/max | Position span XYZ | Orientation p95 | Contact rates T/I/M/R/P |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        q = row["joint_qpos_rms_to_v9_rad"]
        span = "/".join(f"{value:.2f}" for value in row["object_position_span_mm"])
        rates = "/".join(
            f"{row['contacts'][finger]['rate'] * 100:.1f}%"
            for finger in FINGERTIP_NAMES
        )
        lines.append(
            f"| {row['canonical_id']} | {row['rows']} | "
            f"{row['unique_source_trajectories']} | {row['unique_parent_states']} | "
            f"{q['p50']:.4f}/{q['p95']:.4f}/{q['max']:.4f} rad | "
            f"{span} mm | {row['object_orientation_error_to_v9_deg']['p95']:.2f} deg | {rates} |"
        )
    lines.extend(
        [
            "",
            "JSON includes per-joint q statistics, qvel and target-offset norms, object pose/velocity coverage, per-finger force distributions, physics parameters, and split counts.",
            "This report describes reset coverage only; it is not evidence of oracle policy performance.",
        ]
    )
    (ROOT / "P1_P5_V9_Canonical_Coverage_Report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
