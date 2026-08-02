from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import BODY_TO_FINGER


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "canonical_p1_p5_manifest_v8_参考版P3P4低穿透优化.json"
OUTPUT = REPORT / "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
DIAGNOSTICS = REPORT / "canonical_p1_p5_v9_P4拇指下托诊断.json"
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"


def main() -> None:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    p3_phase = next(row for row in manifest["phases"] if int(row["phase"]) == 3)
    p3_phase["contact_set"] = ["thumb", "index", "middle"]
    p3_phase["placement_support"] = ["thumb", "index", "middle"]
    p3_phase["optional_contacts"] = ["ring"]
    p3_phase["summary"] = (
        "复原参考报告 P3：食指完成前绕，形成拇指/食指/中指局部控制入口。"
    )
    p3_phase["contact_semantics_note"] = (
        "v8/v9 真实网格验证的初始接触为 thumb/index/middle；旧 thumb/index/ring 标签已移除。"
    )
    p3_phase["relative_from_p3_deg"] = 0.0
    phase = next(row for row in manifest["phases"] if int(row["phase"]) == 4)
    env = Dex5PenSpinMinimalEnv(model_path=MODEL, init_noise_scale=0.0)
    q_ref = np.asarray(phase["hand_qpos"], dtype=np.float64)
    thumb_indices = np.arange(16, 20)
    pen_geoms = [
        geom_id
        for geom_id in range(env.model.ngeom)
        if "pen_" in (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
    ]
    finger_geoms = []
    thumb_geoms = []
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        body_name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[geom_id]
        ) or ""
        finger = BODY_TO_FINGER.get(body_name)
        if finger and "tip_proxy" not in geom_name:
            finger_geoms.append(geom_id)
        if finger == "thumb" and "tip_proxy" not in geom_name:
            thumb_geoms.append(geom_id)
    thumb_tip_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "Link_14R_tip_proxy"
    )
    if thumb_tip_id < 0:
        raise RuntimeError("Missing Link_14R_tip_proxy")

    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = phase["pen_pos"]
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = phase["pen_quat"]

    def apply(thumb_q: np.ndarray) -> dict[str, float]:
        q = q_ref.copy()
        q[thumb_indices] = thumb_q
        env.data.qpos[env.qpos_adr] = q
        mujoco.mj_forward(env.model, env.data)
        state = env.get_state()
        tip = env.data.geom_xpos[thumb_tip_id]
        axial = float(np.dot(tip - state.pen_center, state.pen_axis))
        closest = state.pen_center + np.clip(axial, -0.07075, 0.07075) * state.pen_axis
        radial = tip - closest
        pair_distances = [
            float(mujoco.mj_geomDistance(env.model, env.data, pen, geom, 0.1, None))
            for pen in pen_geoms
            for geom in finger_geoms
        ]
        thumb_distances = [
            float(mujoco.mj_geomDistance(env.model, env.data, pen, geom, 0.1, None))
            for pen in pen_geoms
            for geom in thumb_geoms
        ]
        tip_distances = [
            float(mujoco.mj_geomDistance(env.model, env.data, pen, thumb_tip_id, 0.1, None))
            for pen in pen_geoms
        ]
        return {
            "axial": axial,
            "vertical": float(radial[2]),
            "centerline": float(np.linalg.norm(radial)),
            "min_all": min(pair_distances),
            "min_thumb": min(thumb_distances),
            "min_tip": min(tip_distances),
        }

    def objective(thumb_q: np.ndarray) -> float:
        metrics = apply(thumb_q)
        dq = thumb_q - q_ref[thumb_indices]
        cost = 0.03 * float(np.sum((dq / 0.35) ** 2))
        cost += 2.0 * ((metrics["axial"] + 0.052) / 0.010) ** 2
        cost += 3.0 * ((metrics["vertical"] + 0.010) / 0.004) ** 2
        cost += 2.0 * ((metrics["centerline"] - 0.013) / 0.003) ** 2
        cost += 5.0e7 * max(0.0, -metrics["min_all"] - 0.00035) ** 2
        cost += 3.0e7 * max(0.0, metrics["min_tip"] - 0.00015) ** 2
        return float(cost)

    joint_margin = 0.06
    bounds = [
        (
            float(env.joint_low[index] + joint_margin),
            float(env.joint_high[index] - joint_margin),
        )
        for index in thumb_indices
    ]
    result = differential_evolution(
        objective,
        bounds,
        seed=59,
        popsize=20,
        maxiter=240,
        tol=1e-8,
        polish=True,
        updating="immediate",
        workers=1,
    )
    metrics = apply(result.x)
    q = q_ref.copy()
    q[thumb_indices] = result.x
    phase["hand_qpos"] = q.tolist()
    phase["control_target_qpos"] = q.tolist()
    phase["name"] = "食指中指绕拇指重抓_拇指下托蓝色笔端"
    phase["summary"] = (
        "保持参考版 P4 的笔姿和食指/中指重抓，拇指指尖移至蓝色笔端方向并从笔轴下方托住。"
    )
    phase["placement_support"] = ["thumb", "index", "middle"]
    phase["relative_from_p3_deg"] = float(phase["phase_deg"] - p3_phase["phase_deg"])
    phase["phase_semantics_note"] = (
        "该值是参考 manifest 的绝对相位；relative_from_p3_deg 约为 78.18°，"
        "不能直接标记为现有 Isaac 50–60° braked-P4。"
    )
    phase["source_validation_status"] = "v9_p4_thumb_under_blue_end_constrained_projection"
    phase["transition_keyframes"] = [
        {
            "name": "P4_拇指下托蓝色笔端",
            "hand_qpos": q.tolist(),
            "pen_pos": phase["pen_pos"],
            "pen_quat": phase["pen_quat"],
        }
    ]
    diagnostics = {
        "success": bool(result.success),
        "message": result.message,
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "thumb_joint_change_rad": (q[thumb_indices] - q_ref[thumb_indices]).tolist(),
        "thumb_axial_from_center_mm": metrics["axial"] * 1000.0,
        "thumb_vertical_from_axis_mm": metrics["vertical"] * 1000.0,
        "thumb_centerline_distance_mm": metrics["centerline"] * 1000.0,
        "minimum_hand_pen_distance_mm": metrics["min_all"] * 1000.0,
        "minimum_thumb_pen_distance_mm": metrics["min_thumb"] * 1000.0,
        "minimum_thumb_tip_pen_distance_mm": metrics["min_tip"] * 1000.0,
    }
    env.close()

    manifest["stage"] = "P1-P5 v9：P4 拇指从下方托住蓝色笔端"
    manifest["source"] = str(SOURCE.relative_to(ROOT))
    manifest["note"] = (
        "P1/P2/P3/P5 不变；P4 固定笔姿与食指/中指，仅局部优化拇指四关节，使指尖到达笔端下方。"
    )
    manifest["phase_convention"] = {
        "phase_deg": "继承参考 manifest 的绝对平面相位",
        "relative_from_p3_deg": "以 v9 P3 为零点的相对相位",
        "braked_p4_target_deg": "Isaac 动态任务仍使用 50–60°；不得与 v9 P4 的约 78.18°相对姿态混用",
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(OUTPUT)


if __name__ == "__main__":
    main()
