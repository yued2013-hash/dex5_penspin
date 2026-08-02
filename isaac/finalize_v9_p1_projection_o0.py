from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import save_npz_atomic, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
SOURCE = ROOT / "projection/P1_round3_strict/canonical_reset_bank/P1_validated.npz"
OUTPUT = ROOT / "canonical_reset_bank/P1_v9_physx_projected_R3_validated.npz"
MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)


def split_isolated(payload: dict[str, np.ndarray], key: str) -> bool:
    for value in np.unique(payload[key]):
        splits = np.unique(payload["metadata_split"][payload[key] == value])
        if splits.size != 1:
            return False
    return True


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reference = np.asarray(
        next(row for row in manifest["phases"] if int(row["phase"]) == 1)[
            "hand_qpos"
        ],
        dtype=np.float64,
    )
    rms = np.sqrt(
        np.mean(
            (np.asarray(payload["robot_qpos"], dtype=np.float64) - reference) ** 2,
            axis=1,
        )
    )
    expected_contact = np.asarray((1, 1, 1, 0, 0), dtype=np.bool_)
    checks = {
        "minimum_rows": len(rms) >= 200,
        "all_r0_r3_labels": all(
            bool(np.asarray(payload[key], dtype=np.bool_).all())
            for key in (
                "labels_r0_pass",
                "labels_r1_pass",
                "labels_r2_pass",
                "labels_r3a_pass",
                "labels_r3b_pass",
            )
        ),
        "exact_tim_contact": bool(
            (np.asarray(payload["contact_state"]) == expected_contact).all()
        ),
        "max_hand_rms_below_0_12": float(rms.max()) < 0.12,
        "max_penetration_below_0_5mm": float(
            payload["labels_baseline_max_penetration_m"].max()
        )
        < 0.0005,
        "source_trajectory_split_isolation": split_isolated(
            payload, "metadata_source_trajectory_id"
        ),
        "parent_state_split_isolation": split_isolated(
            payload, "metadata_parent_state_id"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"P1 finalization failed: {checks}")
    save_npz_atomic(OUTPUT, payload)
    report = {
        "status": "PASS",
        "canonical_id": "P1",
        "rows": int(len(rms)),
        "unique_source_trajectories": int(
            np.unique(payload["metadata_source_trajectory_id"]).size
        ),
        "unique_parent_states": int(
            np.unique(payload["metadata_parent_state_id"]).size
        ),
        "median_hand_rms_rad": float(np.median(rms)),
        "max_hand_rms_rad": float(rms.max()),
        "max_validated_penetration_mm": float(
            payload["labels_baseline_max_penetration_m"].max() * 1000.0
        ),
        "contact_pattern": expected_contact.astype(int).tolist(),
        "checks": checks,
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }
    report_path = ROOT / "P1_V9_PhysX_projection_finalization.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# O0 P1 V9 PhysX 投影验收

**状态：PASS**

- 严格 R3 有效状态：`{report['rows']}`
- 独立 source trajectory：`{report['unique_source_trajectories']}`
- 独立 parent state：`{report['unique_parent_states']}`
- V9 RMS 中位/最大：`{report['median_hand_rms_rad']:.4f}/{report['max_hand_rms_rad']:.4f} rad`
- 最大有效穿透：`{report['max_validated_penetration_mm']:.3f} mm`
- 接触语义：`thumb/index/middle`

本结果表示 P1 已成为可恢复的 canonical reset distribution，不代表已训练转笔策略。
"""
    (ROOT / "P1_V9_PhysX投影验收报告.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
