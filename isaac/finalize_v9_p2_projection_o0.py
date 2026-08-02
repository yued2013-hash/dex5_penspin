from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import concatenate_payloads, save_npz_atomic, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)
SOURCES = (
    ROOT / "projection/P2_round1_strict/canonical_reset_bank/P2_validated.npz",
    ROOT / "projection/P2_round2_strict/canonical_reset_bank/P2_validated.npz",
)
OUTPUT = ROOT / "canonical_reset_bank/P2_v9_physx_projected_R3_validated.npz"
REPORT = ROOT / "P2_V9_PhysX_projection_finalization.json"


def split_isolated(payload: dict[str, np.ndarray], key: str) -> bool:
    groups: dict[int, set[str]] = {}
    for group, split in zip(payload[key], payload["metadata_split"], strict=True):
        groups.setdefault(int(group), set()).add(str(split))
    return all(len(splits) == 1 for splits in groups.values())


def main() -> None:
    payloads = []
    source_rows = []
    for source in SOURCES:
        with np.load(source, allow_pickle=False) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        payloads.append(payload)
        source_rows.append(
            {
                "path": str(source),
                "rows": int(payload["robot_qpos"].shape[0]),
                "sha256": sha256_file(source),
            }
        )
    combined = concatenate_payloads(payloads)
    count = int(combined["robot_qpos"].shape[0])
    # Older projection rounds reserved too small an ID interval. State IDs are
    # artifact-local row identities, so repair them without changing lineage.
    combined["metadata_state_id"] = (
        np.int64(2_202_607_300_000_000_000)
        + np.arange(count, dtype=np.int64)
    )
    state_ids = np.asarray(combined["metadata_state_id"], dtype=np.int64)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reference = np.asarray(
        next(row for row in manifest["phases"] if int(row["phase"]) == 2)[
            "hand_qpos"
        ],
        dtype=np.float64,
    )
    rms = np.sqrt(
        np.mean((np.asarray(combined["robot_qpos"], dtype=np.float64) - reference) ** 2, axis=1)
    )
    required_labels = (
        "labels_r0_pass",
        "labels_r1_pass",
        "labels_r2_pass",
        "labels_r3a_pass",
        "labels_r3b_pass",
    )
    contacts = np.asarray(combined["contact_state"], dtype=np.bool_)
    checks = {
        "minimum_rows": count >= 200,
        "unique_state_ids": int(np.unique(state_ids).size) == int(state_ids.size),
        "all_r0_r3_labels": all(bool(np.asarray(combined[key]).all()) for key in required_labels),
        "tim_contacts": bool(contacts[:, :3].all()),
        "max_hand_rms": float(rms.max(initial=0.0)) < 0.12,
        "source_trajectory_split_isolation": split_isolated(
            combined, "metadata_source_trajectory_id"
        ),
        "parent_state_split_isolation": split_isolated(
            combined, "metadata_parent_state_id"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"P2 finalization failed: {checks}")

    save_npz_atomic(OUTPUT, combined)
    report = {
        "canonical_id": "P2",
        "status": "PASS",
        "semantic_contract": "TIM main control with ring-under pose; ring contact is not mandatory",
        "candidate_total": count,
        "counts": {key.removeprefix("labels_").removesuffix("_pass"): count for key in required_labels},
        "rates": {"final": 1.0},
        "rows": count,
        "unique_source_trajectories": int(
            np.unique(combined["metadata_source_trajectory_id"]).size
        ),
        "unique_parent_states": int(
            np.unique(combined["metadata_parent_state_id"]).size
        ),
        "median_hand_rms_rad": float(np.median(rms)),
        "max_hand_rms_rad": float(rms.max()),
        "max_validated_penetration_mm": float(
            np.asarray(combined["labels_baseline_max_penetration_m"]).max() * 1000.0
        ),
        "checks": checks,
        "sources": source_rows,
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# O0 P2 V9 PhysX 投影验收

**状态：PASS**

- 严格 R3 有效状态：`{count}`
- 独立源轨迹：`{report['unique_source_trajectories']}`
- 独立 parent：`{report['unique_parent_states']}`
- V9 关节 RMS 中位/最大：`{report['median_hand_rms_rad']:.4f}/{report['max_hand_rms_rad']:.4f} rad`
- 最大验证穿透：`{report['max_validated_penetration_mm']:.3f} mm`

P2 语义固定为 TIM 主控、ring-under 姿态；ring 指尖接触不是硬性验收条件。所有数据均通过 R0-R3，且 source trajectory 与 parent state 无 split leakage。
"""
    (ROOT / "P2_V9_PhysX投影验收报告.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
