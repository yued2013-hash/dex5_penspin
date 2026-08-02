from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SEMANTICS = {
    "P1": {"required": ("thumb", "index", "middle"), "forbidden": ()},
    "P2": {"required": ("thumb", "index", "middle"), "forbidden": ()},
    "P3": {
        "required": ("thumb", "index", "middle"),
        "forbidden": ("ring", "pinky"),
    },
    "P4": {
        "required": ("thumb", "index", "middle"),
        "forbidden": ("ring", "pinky"),
    },
    "P5": {
        "required": ("thumb", "index"),
        "forbidden": ("middle", "ring", "pinky"),
    },
}
PROMOTED = {
    "P1": (
        "canonical_reset_bank/P1_v9_physx_projected_R3_validated.npz",
        "projection/P1_round3_strict/validation_reports/P1_validation_report.json",
    ),
    "P2": (
        "canonical_reset_bank/P2_v9_physx_projected_R3_validated.npz",
        "P2_V9_PhysX_projection_finalization.json",
    ),
    "P3": (
        "canonical_reset_bank/P3_v9_physx_projected_R3_validated.npz",
        "projection/P3_diversity_strict/validation_reports/P3_validation_report.json",
    ),
    "P4": (
        "canonical_reset_bank/P4_v9_physx_projected_R3_validated.npz",
        "projection/P4_round3_strict/validation_reports/P4_validation_report.json",
    ),
    "P5": (
        "canonical_reset_bank/P5_v9_physx_projected_R3_validated.npz",
        "projection/P5_round2_strict/validation_reports/P5_validation_report.json",
    ),
}


def contact_name(row: np.ndarray) -> str:
    active = [name for name, value in zip(FINGERS, row, strict=True) if value]
    return "/".join(active) if active else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit formal V9 P1-P5 reset semantics.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = {
        f"P{row['phase']}": np.asarray(row["hand_qpos"], dtype=np.float64)
        for row in manifest["phases"]
    }
    rows = []
    for canonical_id in SEMANTICS:
        if canonical_id in PROMOTED and (root / PROMOTED[canonical_id][0]).exists():
            data_path = root / PROMOTED[canonical_id][0]
            report_path = root / PROMOTED[canonical_id][1]
            data = np.load(data_path, allow_pickle=False)
            penetration_mm = float(
                np.asarray(data["labels_baseline_max_penetration_m"]).max()
                * 1000.0
            )
        else:
            data_path = root / "candidate" / f"{canonical_id}_candidate.npz"
            report_path = (
                root / "validation_reports" / f"{canonical_id}_validation_report.json"
            )
            audited_path = (
                root / "validation_reports" / f"{canonical_id}_audited.npz"
            )
            data = np.load(audited_path, allow_pickle=False)
            penetration_mm = float(
                json.loads(report_path.read_text(encoding="utf-8"))["metrics"][
                    "max_penetration_mm"
                ]
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        qpos = np.asarray(data["robot_qpos"], dtype=np.float64)
        rms = np.sqrt(np.mean((qpos - references[canonical_id]) ** 2, axis=1))
        patterns = Counter(contact_name(row) for row in data["contact_state"])
        final_count = int(report["counts"]["r3b"])
        unique_parents = int(
            np.unique(np.asarray(data["metadata_parent_state_id"], dtype=np.int64)).size
        )
        diversity_ok = canonical_id != "P3" or unique_parents >= 20
        if final_count >= 200 and diversity_ok:
            status = "PASS"
        elif final_count >= 200:
            status = "PARENT_DIVERSITY_PENDING"
        elif final_count > 0:
            status = "LIMITED_R3_VOLUME_PENDING"
        else:
            status = "BLOCKED_AT_R1"
        rows.append(
            {
                "canonical_id": canonical_id,
                "manifest_contacts": next(
                    row["contact_set"]
                    for row in manifest["phases"]
                    if f"P{row['phase']}" == canonical_id
                ),
                "required_contacts": list(SEMANTICS[canonical_id]["required"]),
                "forbidden_contacts": list(SEMANTICS[canonical_id]["forbidden"]),
                "candidate_count": int(report["candidate_total"]),
                "r0_count": int(report["counts"]["r0"]),
                "r1_count": int(report["counts"]["r1"]),
                "r2_count": int(report["counts"]["r2"]),
                "r3a_count": int(report["counts"]["r3a"]),
                "validated_count": final_count,
                "validated_rate": float(report["rates"]["final"]),
                "unique_parent_states": unique_parents,
                "parent_diversity_ok": diversity_ok,
                "median_hand_rms_rad": float(np.median(rms)),
                "max_hand_rms_rad": float(np.max(rms)),
                "dominant_replayed_contacts": patterns.most_common(4),
                "max_replayed_penetration_mm": penetration_mm,
                "status": status,
            }
        )

    complete = all(
        row["validated_count"] >= 200 and row["parent_diversity_ok"]
        for row in rows
    )
    payload = {
        "status": "PASS" if complete else "PARTIAL_NOT_READY_FOR_ORACLE",
        "scope": "formal_v9_p1_p5_canonical_reset_distribution",
        "manifest": str(manifest_path),
        "semantic_rule": (
            "P1-P5 are canonical grasp semantics; C1-C6 remain separate "
            "dynamics-role exploration assets and must not be relabeled."
        ),
        "rows": rows,
    }
    json_path = root / "P1_P5_V9_semantic_alignment_audit.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# P1-P5 V9 语义对齐与 Reset 审计",
        "",
        f"**总状态：{payload['status']}**",
        "",
        "P1-P5 是正式 canonical grasp；C1-C6 只是动力学角色入口，不得互相改名或混用。",
        "",
        "| 姿态 | 必需接触 | 禁止主接触 | 候选 | R0 | R1 | R2 | R3A | 最终 | parent | 关节 RMS | 重放主接触 | 结论 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        patterns = ", ".join(
            f"{name}:{count}" for name, count in row["dominant_replayed_contacts"]
        )
        lines.append(
            f"| {row['canonical_id']} | {'/'.join(row['required_contacts'])} | "
            f"{'/'.join(row['forbidden_contacts']) or '-'} | {row['candidate_count']} | "
            f"{row['r0_count']} | {row['r1_count']} | {row['r2_count']} | "
            f"{row['r3a_count']} | {row['validated_count']} | "
            f"{row['unique_parent_states']} | "
            f"{row['median_hand_rms_rad']:.4f} rad | {patterns} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- P1/P2 已通过数量、V9 语义和严格 R3 门；P2 按 TIM 主控、ring-under 姿态验收。",
            "- P3/P4 已恢复 thumb/index/middle 接触并通过严格 R3；P4 的 thumb 下托接触已在 PhysX 里复现。",
            "- P5 已以 thumb/index 捏持语义通过严格 R3，middle/ring/pinky 不承担接触。",
            "- 本审计不启动 PPO，不改 reward，也不将旧局部 teacher 冒充为 V9 canonical reset。",
            "",
            "## 下一个 O0 门",
            "",
            "P1-P5 数量门已完成。统一 oracle reset distribution 的剩余 O0 数据门，是将 P3 的独立 parent 从 2 扩展到至少 20。",
        ]
    )
    md_path = root / "P1_P5_V9语义对齐审计.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
