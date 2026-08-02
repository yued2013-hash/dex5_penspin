from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "canonical_p1_p5_manifest_v2_candidate.json"
OUTPUT = REPORT / "canonical_p1_p5_manifest_v6_relative_p2_p4_process.json"

# Constrained kinematic projections against the measured three-section pen.
P1_HAND = np.asarray([
    -0.26016408194329804, 0.9166666471298262, 0.5531442464759145, 0.5531442464759145,
    0.08394231063710623, 0.8692948560574142, 0.5398368913848512, 0.5398368913848512,
    -0.01737287892775226, 0.6186941171941378, 0.5891530208557129, 0.5891530208557129,
    0.0029913101773864637, 0.1948719954757803, 1.142328990423241, 1.142328990423241,
    0.30039499, 0.96608961, 0.62965538, 1.09822722,
])
P2_HAND = P1_HAND.copy()
P2_HAND[8:12] = (-0.3831527616915884, 0.7307300640721218, 1.1037142689075026, 1.1037142689075026)
P2_CLEARANCE = P1_HAND.copy()
P2_CLEARANCE[8:12] = (-0.01047284, 0.37041174, 1.06691912, 1.06691912)
P3_OUTSIDE_HAND = P2_HAND.copy()
P3_OUTSIDE_HAND[0:4] = (-0.38397, 0.9729515067167757, 0.24339908702346447, 0.24339908702346447)
P3_HAND = np.asarray([
    -0.12614433831061217, 0.8921614332230898, 1.2647369707136074, 0.922652938451196,
    -0.11703190824107713, 1.1685403109229993, 1.351001627789573, 0.4038981433087831,
    -0.3348563606953124, 0.6664925524767197, 0.5512556396561916, 0.9157464715282392,
    0.002991317778170799, 0.19487204869080713, 1.1834196414217977, 1.1834196414217977,
    -0.20509809077323168, 0.7070101737282823, 0.42146535862669654, 1.0180161661891591,
])
P3_CONTROL_TARGET = P3_HAND.copy()
P3_PEN_POS = [0.01597541712466304, 0.08141976820235207, 0.06510038173184497]
P3_PEN_QUAT = [0.9875658506868363, 0.023255401177493897, -0.04275556257086424, -0.14948190105329834]
P4_HAND = np.asarray([
    -0.08336174750874434, 0.5250534546271823, 0.7979087279133457, 1.1334932285128543,
    -0.2262059907176237, 0.8660905528520373, 0.45592309360967614, 0.9779532132828509,
    -0.07279414465700322, 0.8136575758161678, 0.9085273341215953, 1.0663487780714311,
    0.0029913101773864637, 0.1948719954757803, 1.1497854753342955, 1.1497854753342955,
    0.5059739195549247, 0.7120506620130727, 0.9041562783067408, 0.8673925079274047,
])
P4_CONTROL_TARGET = P4_HAND.copy()
P4_PEN_POS = [0.01444700715011036, 0.10068986671238014, 0.06673375131623013]
P4_PEN_QUAT = [0.942849456958421, -0.06153776263584679, -0.12124772723628666, 0.30421537423841666]
P5_HAND = np.asarray([
    0.29509398, 0.86139831, 0.82034955, 0.82034955,
    0.0897919447470633, 0.9143936860430398, 0.5817523696848387, 0.5817523696848387,
    0.044145740051502874, 0.72247984986214, 0.5175886866355304, 0.5175886866355304,
    0.3839699923992157, 1.5707999467849731, 0.6507807656984166, 0.6507807656984166,
    0.18237568, 0.77814169, 1.10279097, 0.00659693,
])
# P5 ends with middle/ring/pinky in the current P1 rear-side configuration.
P5_HAND[4:16] = P1_HAND[4:16]
P5_UNDER_ENTRY = P5_HAND.copy()
P5_UNDER_ENTRY[4:12] = (
    -0.38396999239921564, 0.5796895067482657, 0.7269022018090977, 0.7269022018090977,
    0.38396999239921564, 1.570799946784973, 0.7750874758263451, 0.7750874758263451,
)


def main() -> None:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    phases = {int(row["phase"]): row for row in manifest["phases"]}
    p1_pen = {"pen_pos": phases[1]["pen_pos"], "pen_quat": phases[1]["pen_quat"]}

    phases[1].update(
        name="thumb_index_middle_control",
        summary="Thumb/index/middle control the measured pen; ring and pinky remain non-primary.",
        hand_qpos=P1_HAND.tolist(),
        contact_set=["thumb", "index", "middle"],
        placement_support=["thumb", "index", "middle"],
        optional_contacts=["ring"],
        source_validation_status="v3_measured_pen_kinematic_projection",
    )
    phases[2].update(
        name="ring_under_transfer_with_p1_triad_hold",
        summary="The P1 thumb/index/middle pose and pen pose stay fixed while ring transfers below.",
        hand_qpos=P2_HAND.tolist(),
        **p1_pen,
        phase_deg=0.0,
        contact_set=["thumb", "index", "middle", "ring"],
        placement_support=["thumb", "index", "middle"],
        optional_contacts=["ring"],
        source_validation_status="v3_measured_pen_kinematic_projection",
        transition_keyframes=[
            {
                "name": "P2a_ring_retract_clearance",
                "hand_qpos": P2_CLEARANCE.tolist(),
                **p1_pen,
            },
            {
                "name": "P2b_ring_under_arrival",
                "hand_qpos": P2_HAND.tolist(),
                **p1_pen,
            },
        ],
    )
    phases[3].update(
        name="食指绕至前方_无名指暂时支撑",
        summary="沿旧版 P2→P3 相对运动：拇指和无名指支撑，食指绕至笔外侧，中指贴近准备接管。",
        hand_qpos=P3_HAND.tolist(),
        pen_pos=P3_PEN_POS,
        pen_quat=P3_PEN_QUAT,
        phase_deg=-3.734,
        contact_set=["thumb", "index", "ring"],
        placement_support=["thumb", "ring"],
        optional_contacts=["middle"],
        moving_fingers=["index"],
        control_target_qpos=P3_CONTROL_TARGET.tolist(),
        actuator_kp=8.0,
        actuator_kv=0.2,
        source_validation_status="v6_relative_old_p2_p3_process_projection",
        transition_keyframes=[
            {
                "name": "P3a_index_outside_clearance",
                "hand_qpos": P3_OUTSIDE_HAND.tolist(),
                **p1_pen,
            },
            {
                "name": "P3b_食指外侧_无名指支撑",
                "hand_qpos": P3_HAND.tolist(),
                "pen_pos": P3_PEN_POS,
                "pen_quat": P3_PEN_QUAT,
            },
        ],
    )
    phases[4].update(
        name="食指中指接管_绕拇指重抓",
        summary="沿旧版 P3→P4 相对运动转过约 54 度：中指接管、无名指卸载，食指/中指夹持并驱动笔绕拇指。",
        hand_qpos=P4_HAND.tolist(),
        pen_pos=P4_PEN_POS,
        pen_quat=P4_PEN_QUAT,
        pen_qvel=[0.0] * 6,
        phase_deg=54.189,
        contact_set=["thumb", "index", "middle"],
        placement_support=["index", "middle", "thumb"],
        optional_contacts=["ring"],
        control_target_qpos=P4_CONTROL_TARGET.tolist(),
        actuator_kp=8.0,
        actuator_kv=0.2,
        source_validation_status="v6_relative_old_p3_p4_process_projection",
        transition_keyframes=[
            {
                "name": "P4_食指中指绕拇指重抓",
                "hand_qpos": P4_HAND.tolist(),
                "pen_pos": P4_PEN_POS,
                "pen_quat": P4_PEN_QUAT,
            },
        ],
    )
    phases[5].update(
        name="thumb_index_pinch_middle_ring_return",
        summary="Thumb/index keep the measured pen pinched while middle/ring withdraw from below to their rear P1-side poses.",
        hand_qpos=P5_HAND.tolist(),
        contact_set=["thumb", "index"],
        placement_support=["thumb", "index"],
        moving_fingers=["middle", "ring"],
        actuator_kp=40.0,
        actuator_kv=2.0,
        source_validation_status="v3_measured_pen_kinematic_projection",
        transition_keyframes=[
            {
                "name": "P5a_thumb_index_takeover_middle_ring_below",
                "hand_qpos": P5_UNDER_ENTRY.tolist(),
                "pen_pos": phases[5]["pen_pos"],
                "pen_quat": phases[5]["pen_quat"],
            },
            {
                "name": "P5b_middle_ring_rear_P1_entry",
                "hand_qpos": P5_HAND.tolist(),
                "pen_pos": phases[5]["pen_pos"],
                "pen_quat": phases[5]["pen_quat"],
            },
        ],
    )
    payload = {
        "stage": "P1-P5 v6：以新版 P2 为基准恢复旧版 P2→P4 相对过程",
        "source": str(SOURCE.relative_to(ROOT)),
        "model_path": "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml",
        "representative_pen": {
            "mass_g": 9.4,
            "length_mm": 141.5,
            "diameters_mm": {"front": 11.2, "middle": 9.8, "rear": 9.5},
            "com_from_rear_mm": 61.4,
            "com_offset_toward_rear_mm": 9.35,
        },
        "note": "P3/P4 不复制旧版世界坐标；保留新版 P2，并移植旧版 P2→P3→P4 的相对笔姿与接触接力。",
        "transition_note": "P2/P3/P4 端点已做低穿透投影；连续中间态尚未通过，禁止把端点线性插值直接作为 teacher 轨迹。",
        "phases": [phases[index] for index in range(1, 6)],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
