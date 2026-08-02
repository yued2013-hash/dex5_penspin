from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import concatenate_payloads, save_npz_atomic, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
SOURCES = tuple(
    ROOT / "canonical_reset_bank" / f"P{phase}_v9_physx_projected_R3_validated.npz"
    for phase in range(1, 6)
)
OUTPUT = ROOT / "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
REPORT = ROOT / "P1_P5_V9_PhysX_O0_finalization.json"


def split_isolated(payload: dict[str, np.ndarray], key: str) -> bool:
    groups: dict[int, set[str]] = {}
    for group, split in zip(payload[key], payload["metadata_split"], strict=True):
        groups.setdefault(int(group), set()).add(str(split))
    return all(len(splits) == 1 for splits in groups.values())


def main() -> None:
    payloads = []
    rows = []
    for phase, source in enumerate(SOURCES, start=1):
        with np.load(source, allow_pickle=False) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        count = int(payload["robot_qpos"].shape[0])
        parents = int(np.unique(payload["metadata_parent_state_id"]).size)
        checks = {
            "minimum_rows": count >= 200,
            "p3_parent_diversity": phase != 3 or parents >= 20,
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
            "source_split_isolation": split_isolated(
                payload, "metadata_source_trajectory_id"
            ),
            "parent_split_isolation": split_isolated(
                payload, "metadata_parent_state_id"
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"P{phase} failed unified-bank gate: {checks}")
        payloads.append(payload)
        rows.append(
            {
                "canonical_id": f"P{phase}",
                "rows": count,
                "unique_source_trajectories": int(
                    np.unique(payload["metadata_source_trajectory_id"]).size
                ),
                "unique_parent_states": parents,
                "checks": checks,
                "source": str(source),
                "source_sha256": sha256_file(source),
            }
        )

    combined = concatenate_payloads(payloads)
    state_ids = np.asarray(combined["metadata_state_id"], dtype=np.int64)
    global_checks = {
        "unique_state_ids": int(np.unique(state_ids).size) == int(state_ids.size),
        "source_split_isolation": split_isolated(
            combined, "metadata_source_trajectory_id"
        ),
        "parent_split_isolation": split_isolated(
            combined, "metadata_parent_state_id"
        ),
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"Unified P1-P5 integrity gate failed: {global_checks}")

    save_npz_atomic(OUTPUT, combined)
    report = {
        "status": "PASS",
        "scope": "O0 formal V9 P1-P5 canonical reset distribution",
        "rows": rows,
        "combined_rows": int(combined["robot_qpos"].shape[0]),
        "global_checks": global_checks,
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# O0 P1-P5 V9 Canonical Reset Distribution 验收",
        "",
        "**状态：PASS**",
        "",
        "| Canonical | R3 状态 | 源轨迹 | parent |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['canonical_id']} | {row['rows']} | "
            f"{row['unique_source_trajectories']} | {row['unique_parent_states']} |"
        )
    lines.extend(
        [
            "",
            f"统一 bank 共 `{report['combined_rows']}` 条，全部通过严格 R0-R3。",
            "P3 parent diversity 门槛已纳入正式验收；合并后 state ID 唯一，source/parent 无 split leakage。",
            "",
            "该结果只表示 O0 reset distribution 已就绪，不表示连续转笔或 oracle policy 已经学会。",
        ]
    )
    (ROOT / "P1_P5_V9_PhysX_O0验收报告.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
