from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports/m3_2_30_index_near_surface/physx_alignment_rollout_100ms.npz"
DEFAULT_OUTPUT = ROOT / "reports/m3_2_30_index_near_surface/mujoco_physx_alignment_100ms.json"
DEFAULT_XML = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"
JOINTS = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
TIP_BODIES = ("Link_14R", "Link_24R", "Link_34R", "Link_44R", "Link_54R")
TIP_GEOMS = tuple(f"{name}_tip_proxy" for name in TIP_BODIES)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PHYSX_PROXY_BASE_RADII = np.asarray((0.012, 0.012, 0.012, 0.012, 0.011))
PEN_HALF_LENGTH = 0.07075
PEN_SECTION_RADII = np.asarray((0.0056, 0.0049, 0.00475))
TIP_OFFSETS = np.asarray(
    ((0.026, 0.002, 0.0), (0.020, -0.004, 0.0), (0.020, -0.004, 0.0),
     (0.020, -0.004, 0.0), (0.018, -0.004, 0.0)),
    dtype=np.float64,
)


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(a * b, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dot)


def tip_distances(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    tips = []
    for name, offset in zip(TIP_BODIES, TIP_OFFSETS, strict=True):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        tips.append(data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset)
    tips = np.asarray(tips)
    pen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
    center = data.xpos[pen_id]
    axis = data.xmat[pen_id].reshape(3, 3)[:, 0]
    a, b = center - PEN_HALF_LENGTH * axis, center + PEN_HALF_LENGTH * axis
    segment = b - a
    fraction = np.clip(((tips - a) @ segment) / max(float(segment @ segment), 1e-9), 0.0, 1.0)
    section_ids = np.minimum((fraction * 3.0).astype(np.int64), 2)
    return (
        np.linalg.norm(tips - (a + fraction[:, None] * segment), axis=-1)
        - PEN_SECTION_RADII[section_ids]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired MuJoCo/PhysX 100 ms alignment metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    args = parser.parse_args()

    with np.load(args.input) as source:
        rollout = {key: np.asarray(source[key]) for key in source.files}
    proxy_scales = np.asarray(
        rollout.get("collision_proxy_radius_scales", (1.0, 1.08, 1.08, 0.85, 0.80)),
        dtype=np.float64,
    )
    physx_proxy_radii = PHYSX_PROXY_BASE_RADII * proxy_scales
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    qpos_ids = np.asarray([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINTS
    ])
    qvel_ids = np.asarray([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINTS
    ])
    pen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pen_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_free")
    pen_qpos = int(model.jnt_qposadr[pen_joint])
    pen_qvel = int(model.jnt_dofadr[pen_joint])
    physx_dt = float(rollout["physx_dt_s"])
    substeps = max(1, int(round(physx_dt / float(model.opt.timestep))))
    env_count = rollout["source_qpos"].shape[0]
    mujoco_proxy_radii = np.asarray([
        model.geom_size[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name), 0
        ]
        for geom_name in TIP_GEOMS
    ])

    datas: list[mujoco.MjData] = []
    for env_id in range(env_count):
        data = mujoco.MjData(model)
        data.qpos[qpos_ids] = rollout["source_qpos"][env_id, :20]
        data.qvel[qvel_ids] = rollout["source_qvel"][env_id, :20]
        data.qpos[pen_qpos:pen_qpos + 7] = rollout["source_qpos"][env_id, 20:27]
        data.qvel[pen_qvel:pen_qvel + 6] = rollout["source_qvel"][env_id, 20:26]
        data.ctrl[:] = rollout["source_target"][env_id]
        mujoco.mj_forward(model, data)
        datas.append(data)

    per_step = []
    for step in range(int(rollout["steps"]) + 1):
        if step > 0:
            for env_id, data in enumerate(datas):
                data.ctrl[:] = rollout["source_target"][env_id]
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
        mj_qpos = np.asarray([data.qpos[qpos_ids].copy() for data in datas])
        mj_qvel = np.asarray([data.qvel[qvel_ids].copy() for data in datas])
        mj_pos = np.asarray([data.xpos[pen_id].copy() for data in datas])
        mj_quat = np.asarray([data.xquat[pen_id].copy() for data in datas])
        mj_linvel = np.asarray([data.qvel[pen_qvel:pen_qvel + 3].copy() for data in datas])
        mj_angvel = np.asarray([data.qvel[pen_qvel + 3:pen_qvel + 6].copy() for data in datas])
        mj_tip = np.asarray([tip_distances(model, data) for data in datas])
        px_tip = rollout["physx_tip_surface_dist"][step]
        px_contact = px_tip <= physx_proxy_radii
        mj_contact = mj_tip <= mujoco_proxy_radii
        px_occupancy = np.mean(px_contact, axis=0)
        mj_occupancy = np.mean(mj_contact, axis=0)
        per_step.append(
            {
                "step": step,
                "elapsed_ms": step * physx_dt * 1000.0,
                "hand_qpos_rmse_rad": float(np.sqrt(np.mean((rollout["physx_hand_qpos"][step] - mj_qpos) ** 2))),
                "hand_qvel_rmse_rad_s": float(np.sqrt(np.mean((rollout["physx_hand_qvel"][step] - mj_qvel) ** 2))),
                "pen_position_error_mm": float(np.linalg.norm(rollout["physx_pen_pos"][step] - mj_pos, axis=-1).mean() * 1000.0),
                "pen_orientation_error_deg": float(np.mean(quat_angle(rollout["physx_pen_quat"][step], mj_quat)) * 180.0 / np.pi),
                "pen_linear_velocity_error_m_s": float(np.linalg.norm(rollout["physx_pen_linvel"][step] - mj_linvel, axis=-1).mean()),
                "pen_angular_velocity_error_rad_s": float(np.linalg.norm(rollout["physx_pen_angvel"][step] - mj_angvel, axis=-1).mean()),
                "tip_surface_error_mm": float(np.abs(px_tip - mj_tip).mean() * 1000.0),
                "geometric_contact_agreement": float(np.mean(px_contact == mj_contact)),
                "physx_geometric_contact_ratio": {
                    name: float(px_occupancy[index])
                    for index, name in enumerate(FINGER_NAMES)
                },
                "mujoco_geometric_contact_ratio": {
                    name: float(mj_occupancy[index])
                    for index, name in enumerate(FINGER_NAMES)
                },
                "physx_done_count": int(np.sum(rollout["done"][step - 1])) if step > 0 else 0,
            }
        )

    final = per_step[-1]
    summary_50ms = min(per_step, key=lambda row: abs(row["elapsed_ms"] - 50.0))
    summary_100ms = min(per_step, key=lambda row: abs(row["elapsed_ms"] - 100.0))
    report = {
        "status": "MEASURED",
        "task": str(rollout["task"]),
        "num_envs": env_count,
        "steps": int(rollout["steps"]),
        "elapsed_ms": final["elapsed_ms"],
        "physx_dt_s": physx_dt,
        "mujoco_dt_s": float(model.opt.timestep),
        "mujoco_substeps_per_physx_step": substeps,
        "collision_proxy_radius_scales": proxy_scales.tolist(),
        "pen_geometry_note": "PhysX uses the measured 141.5 mm equivalent cylinder; MuJoCo uses the measured 11.2/9.8/9.5 mm three-section diameters.",
        "model_note": "Paired from identical projected qpos/qvel and pen pose. MuJoCo XML and PhysX USD use different solvers and calibrated fingertip proxy radii.",
        "per_step": per_step,
        "summary_50ms": summary_50ms,
        "summary_100ms": summary_100ms,
        "summary_final": final,
        "interpretation": "This is a calibration measurement, not an equivalence pass. Use per-channel errors to set PhysX-specific reset and reward tolerances.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
