from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import concatenate_payloads, save_npz_atomic, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
)
MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)
SOURCES = {
    "P3": ROOT / "projection/P3_diversity_strict/canonical_reset_bank/P3_validated.npz",
    "P4": ROOT / "projection/P4_round3_strict/canonical_reset_bank/P4_validated.npz",
    "P5": ROOT / "projection/P5_round2_strict/canonical_reset_bank/P5_validated.npz",
}
EXPECTED_CONTACTS = {
    "P3": np.asarray((1, 1, 1, 0, 0), dtype=np.bool_),
    "P4": np.asarray((1, 1, 1, 0, 0), dtype=np.bool_),
    "P5": np.asarray((1, 1, 0, 0, 0), dtype=np.bool_),
}
MIN_ROWS = 200
MAX_HAND_RMS_RAD = 0.12


def split_isolation(payload: dict[str, np.ndarray], key: str) -> bool:
    groups: dict[int, set[str]] = {}
    for group, split in zip(payload[key], payload["metadata_split"], strict=True):
        groups.setdefault(int(group), set()).add(str(split))
    return all(len(splits) == 1 for splits in groups.values())


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    references = {
        f"P{row['phase']}": np.asarray(row["hand_qpos"], dtype=np.float64)
        for row in manifest["phases"]
    }
    output_dir = ROOT / "canonical_reset_bank"
    payloads = []
    rows = []
    for canonical_id, source in SOURCES.items():
        with np.load(source, allow_pickle=False) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        count = int(payload["robot_qpos"].shape[0])
        rms = np.sqrt(
            np.mean(
                (
                    np.asarray(payload["robot_qpos"], dtype=np.float64)
                    - references[canonical_id][None, :]
                )
                ** 2,
                axis=1,
            )
        )
        labels_ok = all(
            bool(np.asarray(payload[key], dtype=np.bool_).all())
            for key in (
                "labels_r0_pass",
                "labels_r1_pass",
                "labels_r2_pass",
                "labels_r3a_pass",
                "labels_r3b_pass",
            )
        )
        contacts_ok = bool(
            (
                np.asarray(payload["contact_state"], dtype=np.bool_)
                == EXPECTED_CONTACTS[canonical_id][None, :]
            ).all()
        )
        trajectory_split_ok = split_isolation(
            payload, "metadata_source_trajectory_id"
        )
        parent_rows = np.asarray(payload["metadata_parent_state_id"]) >= 0
        parent_split_ok = (
            split_isolation(
                {key: value[parent_rows] for key, value in payload.items()},
                "metadata_parent_state_id",
            )
            if np.any(parent_rows)
            else True
        )
        checks = {
            "minimum_rows": count >= MIN_ROWS,
            "all_r0_r3_labels": labels_ok,
            "exact_contact_semantics": contacts_ok,
            "max_hand_rms": float(rms.max(initial=0.0)) < MAX_HAND_RMS_RAD,
            "source_trajectory_split_isolation": trajectory_split_ok,
            "parent_state_split_isolation": parent_split_ok,
        }
        if not all(checks.values()):
            raise RuntimeError(f"{canonical_id} finalization failed: {checks}")
        output = output_dir / f"{canonical_id}_v9_physx_projected_R3_validated.npz"
        save_npz_atomic(output, payload)
        payloads.append(payload)
        rows.append(
            {
                "canonical_id": canonical_id,
                "rows": count,
                "unique_source_trajectories": int(
                    np.unique(payload["metadata_source_trajectory_id"]).size
                ),
                "unique_parent_states": int(
                    np.unique(payload["metadata_parent_state_id"]).size
                ),
                "median_hand_rms_rad": float(np.median(rms)),
                "max_hand_rms_rad": float(rms.max()),
                "max_validated_penetration_mm": float(
                    np.asarray(payload["labels_baseline_max_penetration_m"]).max()
                    * 1000.0
                ),
                "min_r3a_success_ratio": float(
                    np.asarray(payload["labels_r3a_success_ratio"]).min()
                ),
                "min_r3b_r1_success_ratio": float(
                    np.asarray(payload["labels_r3b_r1_success_ratio"]).min()
                ),
                "min_r3b_r2_success_ratio": float(
                    np.asarray(payload["labels_r3b_r2_success_ratio"]).min()
                ),
                "contact_pattern": EXPECTED_CONTACTS[canonical_id].astype(int).tolist(),
                "checks": checks,
                "source": str(source),
                "source_sha256": sha256_file(source),
                "output": str(output),
                "output_sha256": sha256_file(output),
            }
        )

    combined = concatenate_payloads(payloads)
    combined_path = output_dir / "P3_P5_v9_physx_projected_R3_validated.npz"
    save_npz_atomic(combined_path, combined)
    report = {
        "status": "PASS",
        "scope": "O0 V9 P3/P4/P5 PhysX projection and strict reset-replay",
        "minimum_rows_per_canonical": MIN_ROWS,
        "max_hand_rms_rad": MAX_HAND_RMS_RAD,
        "rows": rows,
        "combined_rows": int(combined["robot_qpos"].shape[0]),
        "combined": str(combined_path),
        "combined_sha256": sha256_file(combined_path),
    }
    json_path = ROOT / "P3_P5_V9_PhysX_projection_finalization.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# O0 P3/P4/P5 V9 PhysX 投影验收",
        "",
        "**状态：PASS**",
        "",
        "| Canonical | R3 有效状态 | 源轨迹 | 父状态 | RMS 中位/最大 | 最大穿透 | 接触 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        contact = "TIM" if row["canonical_id"] in ("P3", "P4") else "TI"
        markdown.append(
            f"| {row['canonical_id']} | {row['rows']} | "
            f"{row['unique_source_trajectories']} | {row['unique_parent_states']} | "
            f"{row['median_hand_rms_rad']:.4f}/{row['max_hand_rms_rad']:.4f} rad | "
            f"{row['max_validated_penetration_mm']:.3f} mm | {contact} |"
        )
    markdown.extend(
        [
            "",
            f"合并 bank 共 `{report['combined_rows']}` 条。所有行均通过 R0-R3，且按 source trajectory 与 parent state 检查 split 隔离。",
            "",
            "该结果只证明 P3/P4/P5 已成为可恢复的 canonical reset distributions，不代表三段连续动力学轨迹或完整转笔已经成功。",
        ]
    )
    (ROOT / "P3_P5_V9_PhysX投影验收报告.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
