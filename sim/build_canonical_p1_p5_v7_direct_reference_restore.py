from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
CURRENT = REPORT / "canonical_p1_p5_manifest_v6_relative_p2_p4_process.json"
REFERENCE = REPORT / "canonical_p1_p5_manifest.json"
OUTPUT = REPORT / "canonical_p1_p5_manifest_v7_直接复原参考版P3P4.json"


def main() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    current_phases = {int(row["phase"]): row for row in current["phases"]}
    reference_phases = {int(row["phase"]): row for row in reference["phases"]}

    for phase_id in (3, 4):
        restored = copy.deepcopy(reference_phases[phase_id])
        restored["control_target_qpos"] = copy.deepcopy(restored["hand_qpos"])
        restored["actuator_kp"] = 8.0
        restored["actuator_kv"] = 0.2
        restored["source_validation_status"] = "v7_exact_restore_from_latest_canonical_p1_p5"
        restored["transition_keyframes"] = [
            {
                "name": f"P{phase_id}_参考版原始端点",
                "hand_qpos": copy.deepcopy(restored["hand_qpos"]),
                "pen_pos": copy.deepcopy(restored["pen_pos"]),
                "pen_quat": copy.deepcopy(restored["pen_quat"]),
            }
        ]
        current_phases[phase_id] = restored

    current_phases[3].update(
        name="食指绕至前方_参考版原样复原",
        summary="直接复原参考报告 P3：拇指/食指/无名指接触，不做相对坐标移植或碰撞投影。",
        placement_support=["thumb", "ring"],
        optional_contacts=["middle"],
        moving_fingers=["index"],
    )
    current_phases[4].update(
        name="食指中指绕拇指重抓_参考版原样复原",
        summary="直接复原参考报告 P4：拇指/食指/中指再捕获，不修改原始笔姿态。",
        placement_support=["thumb", "index", "middle"],
        optional_contacts=["ring"],
        moving_fingers=["index", "middle", "ring"],
    )

    payload = copy.deepcopy(current)
    payload["stage"] = "P1-P5 v7：直接复原 latest_canonical_p1_p5 的 P3/P4"
    payload["reference_report"] = str(
        (REPORT / "latest_canonical_p1_p5/index.html").relative_to(ROOT)
    )
    payload["reference_manifest"] = str(REFERENCE.relative_to(ROOT))
    payload["note"] = (
        "P3/P4 的手部关节、笔位置、笔四元数、接触集合和 phase_deg 均从参考 manifest 原样复制；"
        "P1/P2/P5 沿用当前版本。"
    )
    payload["transition_note"] = (
        "这是参考视觉状态的直接复原。P2→P3 和 P3→P4 连续动力学仍需另行训练，"
        "不能用线性插值替代。"
    )
    payload["phases"] = [current_phases[index] for index in range(1, 6)]
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
