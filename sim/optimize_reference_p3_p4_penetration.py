from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import BODY_TO_FINGER


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "canonical_p1_p5_manifest_v7_直接复原参考版P3P4.json"
OUTPUT = REPORT / "canonical_p1_p5_manifest_v8_参考版P3P4低穿透优化.json"
DIAGNOSTICS = REPORT / "canonical_p1_p5_v8_优化诊断.json"
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINT_SLICES = {
    "index": np.arange(0, 4),
    "middle": np.arange(4, 8),
    "ring": np.arange(8, 12),
    "pinky": np.arange(12, 16),
    "thumb": np.arange(16, 20),
}


def quat_with_rotvec(reference: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    # MuJoCo uses wxyz; scipy uses xyzw.
    base = Rotation.from_quat(np.r_[reference[1:], reference[0]])
    result = Rotation.from_rotvec(rotvec) * base
    xyzw = result.as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def main() -> None:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    phases = {int(row["phase"]): row for row in manifest["phases"]}
    env = Dex5PenSpinMinimalEnv(model_path=MODEL, init_noise_scale=0.0)

    pen_geoms = [
        geom_id
        for geom_id in range(env.model.ngeom)
        if "pen_" in (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
    ]
    finger_geoms = {finger: [] for finger in FINGERS}
    for geom_id in range(env.model.ngeom):
        body_name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[geom_id]
        ) or ""
        finger = BODY_TO_FINGER.get(body_name)
        if finger in finger_geoms:
            finger_geoms[finger].append(geom_id)

    configs = {
        3: {
            "active": ("thumb", "index", "ring"),
            "required": ("thumb", "index", "ring"),
            "joint_delta": 0.30,
            "position_delta": 0.012,
            "rotation_delta": 0.08,
            "seed": 37,
        },
        4: {
            "active": ("thumb", "index", "middle"),
            "required": ("thumb", "index", "middle"),
            "joint_delta": 0.40,
            "position_delta": 0.022,
            "rotation_delta": 0.14,
            "seed": 41,
        },
    }
    diagnostic_rows = []

    for phase_id, config in configs.items():
        phase = phases[phase_id]
        q_ref = np.asarray(phase["hand_qpos"], dtype=np.float64)
        pos_ref = np.asarray(phase["pen_pos"], dtype=np.float64)
        quat_ref = np.asarray(phase["pen_quat"], dtype=np.float64)
        active_indices = np.concatenate([JOINT_SLICES[finger] for finger in config["active"]])
        bounds = []
        for index in active_indices:
            lo = max(float(env.joint_low[index]), q_ref[index] - config["joint_delta"])
            hi = min(float(env.joint_high[index]), q_ref[index] + config["joint_delta"])
            bounds.append((lo, hi))
        bounds.extend(
            (pos_ref[axis] - config["position_delta"], pos_ref[axis] + config["position_delta"])
            for axis in range(3)
        )
        bounds.extend([(-config["rotation_delta"], config["rotation_delta"])] * 3)

        def apply(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            q = q_ref.copy()
            q[active_indices] = candidate[: len(active_indices)]
            pos = candidate[len(active_indices) : len(active_indices) + 3]
            rotvec = candidate[len(active_indices) + 3 :]
            quat = quat_with_rotvec(quat_ref, rotvec)
            env.data.qpos[env.qpos_adr] = q
            env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = pos
            env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat
            mujoco.mj_forward(env.model, env.data)
            return q, pos, quat

        def distances() -> dict[str, list[float]]:
            return {
                finger: [
                    float(mujoco.mj_geomDistance(env.model, env.data, pen, geom, 0.1, None))
                    for pen in pen_geoms
                    for geom in finger_geoms[finger]
                ]
                for finger in FINGERS
            }

        def objective(candidate: np.ndarray) -> float:
            q, pos, _ = apply(candidate)
            rows = distances()
            dq = q[active_indices] - q_ref[active_indices]
            rotvec = candidate[len(active_indices) + 3 :]
            cost = 0.10 * float(np.sum((dq / 0.16) ** 2))
            cost += 0.25 * float(np.sum(((pos - pos_ref) / 0.007) ** 2))
            cost += 0.20 * float(np.sum((rotvec / 0.10) ** 2))
            for values in rows.values():
                array = np.asarray(values)
                cost += 5.0e7 * float(np.sum(np.maximum(0.0, -array - 0.00030) ** 2))
            for finger in config["required"]:
                minimum = min(rows[finger])
                cost += 5.0e7 * max(0.0, minimum - 0.00005) ** 2
            return cost

        initial = np.r_[q_ref[active_indices], pos_ref, np.zeros(3)]
        result = differential_evolution(
            objective,
            bounds,
            seed=config["seed"],
            popsize=12,
            maxiter=220,
            tol=1e-7,
            polish=True,
            workers=1,
            updating="immediate",
        )
        q, pos, quat = apply(result.x)
        final_distances = distances()
        minima = {finger: min(values) for finger, values in final_distances.items()}
        maximum_penetration = max(0.0, -min(min(values) for values in final_distances.values()))
        rotation_change = Rotation.from_rotvec(result.x[len(active_indices) + 3 :]).magnitude()

        phase["hand_qpos"] = q.tolist()
        phase["pen_pos"] = pos.tolist()
        phase["pen_quat"] = quat.tolist()
        phase["control_target_qpos"] = q.tolist()
        phase["source_validation_status"] = "v8_constrained_low_penetration_projection_from_v7_reference"
        phase["transition_keyframes"] = [
            {
                "name": f"P{phase_id}_参考形态低穿透端点",
                "hand_qpos": q.tolist(),
                "pen_pos": pos.tolist(),
                "pen_quat": quat.tolist(),
            }
        ]
        diagnostic_rows.append(
            {
                "phase": phase_id,
                "success": bool(result.success),
                "message": result.message,
                "objective": float(result.fun),
                "iterations": int(result.nit),
                "max_penetration_mm": maximum_penetration * 1000.0,
                "required_contact_distance_mm": {
                    finger: minima[finger] * 1000.0 for finger in config["required"]
                },
                "joint_rms_change_rad": float(np.sqrt(np.mean((q - q_ref) ** 2))),
                "pen_position_change_mm": float(np.linalg.norm(pos - pos_ref) * 1000.0),
                "pen_rotation_change_deg": float(np.degrees(rotation_change)),
            }
        )

    env.close()
    manifest["stage"] = "P1-P5 v8：参考版 P3/P4 低穿透约束优化"
    manifest["source"] = str(SOURCE.relative_to(ROOT))
    manifest["note"] = (
        "以 v7 直接复原姿态为锚点，仅在受限邻域内调整相关手指和笔姿；"
        "目标是保留 P3/P4 视觉过程并消除真实网格穿模。"
    )
    manifest["phases"] = [phases[index] for index in range(1, 6)]
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DIAGNOSTICS.write_text(
        json.dumps({"source": str(SOURCE), "phases": diagnostic_rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostic_rows, indent=2, ensure_ascii=False))
    print(OUTPUT)


if __name__ == "__main__":
    main()
