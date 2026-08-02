from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import BODY_TO_FINGER


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports/m3_2_30_index_near_surface/canonical_p1_p5_manifest_v2_candidate.json"
DEFAULT_MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_proxy_only.xml"
DEFAULT_VISUAL_MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
DEFAULT_OUTPUT = ROOT / "reports/m3_2_30_index_near_surface/measured_pen_canonical_validation.json"
TIP_GEOMS = {
    "thumb": "Link_14R_tip_proxy",
    "index": "Link_24R_tip_proxy",
    "middle": "Link_34R_tip_proxy",
    "ring": "Link_44R_tip_proxy",
    "pinky": "Link_54R_tip_proxy",
}


def set_state(env: Dex5PenSpinMinimalEnv, state: dict[str, Any]) -> None:
    mujoco.mj_resetData(env.model, env.data)
    hand = np.asarray(state["hand_qpos"], dtype=np.float64)
    env.data.qpos[env.qpos_adr] = hand
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = state["pen_pos"]
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = state["pen_quat"]
    env.data.qvel[:] = 0.0
    env.target_qpos = hand.astype(np.float32)
    env.data.ctrl[:] = hand
    mujoco.mj_forward(env.model, env.data)


def configure_proxy_model(env: Dex5PenSpinMinimalEnv, radius: float, friction: float) -> None:
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "tip_proxy" not in geom_name:
            continue
        if "p3_cradle" not in geom_name:
            env.model.geom_size[geom_id, 0] = radius
        env.model.geom_friction[geom_id] = (friction, 0.01, 0.001)
        env.model.geom_rgba[geom_id, 3] = 0.0
        env.model.geom_contype[geom_id] = 1
        env.model.geom_conaffinity[geom_id] = 1
    env.model.actuator_forcelimited[:] = True
    env.model.actuator_forcerange[:] = (-0.93, 0.93)


def pen_contacts(env: Dex5PenSpinMinimalEnv) -> tuple[set[str], Counter[str], float, dict[str, float]]:
    fingers: set[str] = set()
    geoms: Counter[str] = Counter()
    normal_force: defaultdict[str, float] = defaultdict(float)
    max_penetration = 0.0
    result = np.zeros(6, dtype=np.float64)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        name1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        name2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if "pen_" not in name1 and "pen_" not in name2:
            continue
        other_id = contact.geom2 if "pen_" in name1 else contact.geom1
        other_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other_id) or f"geom{other_id}"
        body_name = mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[other_id]
        ) or ""
        finger = BODY_TO_FINGER.get(body_name)
        geoms[other_name] += 1
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if finger is None:
            continue
        fingers.add(finger)
        mujoco.mj_contactForce(env.model, env.data, index, result)
        normal_force[finger] += max(0.0, float(result[0]))
    return fingers, geoms, max_penetration, dict(normal_force)


def tip_geometry(env: Dex5PenSpinMinimalEnv) -> dict[str, dict[str, float]]:
    center = env.get_state().pen_center
    axis = env.get_state().pen_axis
    out: dict[str, dict[str, float]] = {}
    for finger, geom_name in TIP_GEOMS.items():
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        tip = env.data.geom_xpos[geom_id]
        axial = float(np.dot(tip - center, axis))
        closest = center + np.clip(axial, -0.07075, 0.07075) * axis
        radial = tip - closest
        out[finger] = {
            "axial_from_center_mm": axial * 1000.0,
            "vertical_from_axis_mm": float(radial[2] * 1000.0),
            "centerline_distance_mm": float(np.linalg.norm(radial) * 1000.0),
        }
    return out


def visual_mesh_penetration(state: dict[str, Any], model_path: Path) -> float:
    env = Dex5PenSpinMinimalEnv(model_path=model_path, init_noise_scale=0.0)
    set_state(env, state)
    maximum = 0.0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        name1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        name2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if "pen_" in name1 or "pen_" in name2:
            maximum = max(maximum, max(0.0, -float(contact.dist)))
    env.close()
    return maximum


def phase_semantics(phase: int, row: dict[str, Any]) -> dict[str, bool]:
    ratios = row["contact_ratio"]
    shares = row["normal_force_share"]
    geometry = row["initial_tip_geometry"]
    checks: dict[str, bool] = {}
    if phase == 1:
        checks["thumb_index_middle_control"] = all(finger in row["initial_contacts"] for finger in ("thumb", "index", "middle"))
        checks["ring_pinky_not_primary"] = shares.get("ring", 0.0) + shares.get("pinky", 0.0) <= 0.30
    elif phase == 2:
        checks["p1_triad_preserved"] = all(finger in row["initial_contacts"] for finger in ("thumb", "index", "middle"))
        checks["ring_under_contact"] = "ring" in row["initial_contacts"] and geometry.get("ring", {}).get("vertical_from_axis_mm", 1.0) < 0.0
    elif phase == 3:
        checks["thumb_index_ring_initial_relay"] = all(
            finger in row["initial_contacts"] for finger in ("thumb", "index", "ring")
        )
        checks["index_and_ring_short_contact"] = (
            ratios.get("index", 0.0) >= 0.5 and ratios.get("ring", 0.0) >= 0.3
        )
        checks["ring_is_temporary_not_dominant"] = (
            shares.get("ring", 0.0) <= 0.55
        )
    elif phase == 4:
        checks["thumb_orbit_angle"] = 45.0 <= abs(float(row.get("phase_deg", 0.0))) <= 65.0
        checks["index_middle_drive_contact"] = all(
            finger in row["initial_contacts"] for finger in ("index", "middle")
        )
        checks["index_middle_active_drive_force"] = (
            ratios.get("index", 0.0) >= 0.4
            and ratios.get("middle", 0.0) >= 0.3
            and shares.get("index", 0.0) >= 0.10
            and shares.get("middle", 0.0) >= 0.10
        )
        checks["opposite_thumb_support_contact"] = "thumb" in row["initial_contacts"]
    elif phase == 5:
        checks["thumb_index_pinch"] = ratios.get("thumb", 0.0) >= 0.5 and ratios.get("index", 0.0) >= 0.5
        checks["middle_ring_not_supporting"] = shares.get("middle", 0.0) + shares.get("ring", 0.0) <= 0.25
    return checks


def evaluate_phase(
    env: Dex5PenSpinMinimalEnv,
    visual_model: Path,
    phase: dict[str, Any],
    settle_ms: float,
) -> dict[str, Any]:
    env._set_position_actuator_gains(
        actuator_kp=float(phase.get("actuator_kp", 8.0)),
        actuator_kv=float(phase.get("actuator_kv", 0.2)),
    )
    set_state(env, phase)
    if "control_target_qpos" in phase:
        control_target = np.asarray(phase["control_target_qpos"], dtype=np.float64)
        env.target_qpos = control_target.astype(np.float32)
        env.data.ctrl[:] = control_target
    initial_center = env.get_state().pen_center.copy()
    initial_tip_geometry = tip_geometry(env)
    initial_contacts, _, initial_penetration, _ = pen_contacts(env)
    joint_margin = np.minimum(
        np.asarray(phase["hand_qpos"]) - env.joint_low,
        env.joint_high - np.asarray(phase["hand_qpos"]),
    )
    steps = int(round(settle_ms / 1000.0 * env.control_hz))
    contact_counts: Counter[str] = Counter()
    force_sums: Counter[str] = Counter()
    max_penetration = initial_penetration
    max_effort = 0.0
    max_linear_speed = 0.0
    max_angular_speed = 0.0
    terminated = False
    samples = 0
    for _ in range(steps):
        _, _, terminated, _, _ = env.step(np.zeros(20, dtype=np.float32))
        contacts, _, penetration, forces = pen_contacts(env)
        contact_counts.update(contacts)
        force_sums.update(forces)
        max_penetration = max(max_penetration, penetration)
        effort = env.data.qfrc_actuator[env.qvel_adr]
        max_effort = max(max_effort, float(np.max(np.abs(effort))))
        pen_vel = env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6]
        max_linear_speed = max(max_linear_speed, float(np.linalg.norm(pen_vel[:3])))
        max_angular_speed = max(max_angular_speed, float(np.linalg.norm(pen_vel[3:])))
        samples += 1
        if terminated:
            break
    final_state = env.get_state()
    total_force = float(sum(force_sums.values()))
    row: dict[str, Any] = {
        "phase": int(phase["phase"]),
        "name": phase["name"],
        "phase_deg": float(phase.get("phase_deg", 0.0)),
        "settle_ms_requested": settle_ms,
        "settle_ms_completed": samples / env.control_hz * 1000.0,
        "initial_contacts": sorted(initial_contacts),
        "contact_ratio": {finger: contact_counts[finger] / max(samples, 1) for finger in TIP_GEOMS},
        "normal_force_share": {finger: force_sums[finger] / max(total_force, 1e-9) for finger in TIP_GEOMS},
        "initial_tip_geometry": initial_tip_geometry,
        "initial_penetration_mm": initial_penetration * 1000.0,
        "max_penetration_mm": max_penetration * 1000.0,
        "visual_mesh_penetration_mm": visual_mesh_penetration(phase, visual_model) * 1000.0,
        "minimum_joint_limit_margin_rad": float(np.min(joint_margin)),
        "max_actuator_effort": max_effort,
        "pen_drift_mm": float(np.linalg.norm(final_state.pen_center - initial_center) * 1000.0),
        "max_linear_speed_m_s": max_linear_speed,
        "max_angular_speed_rad_s": max_angular_speed,
        "final_angular_speed_rad_s": float(np.linalg.norm(final_state.pen_angvel)),
        "terminated": terminated,
    }
    row["common_checks"] = {
        "penetration_below_1mm": row["max_penetration_mm"] < 1.0,
        "visual_mesh_penetration_below_1mm": row["visual_mesh_penetration_mm"] < 1.0,
        "within_joint_limits": row["minimum_joint_limit_margin_rad"] >= -1e-6,
        "actuator_effort_below_0p93": row["max_actuator_effort"] <= 0.93,
        "settled_without_explosion": not terminated and row["pen_drift_mm"] < 15.0 and row["max_linear_speed_m_s"] < 1.0,
    }
    if int(phase["phase"]) == 4 and "drive_around_thumb" in phase["name"]:
        row["common_checks"].pop("settled_without_explosion")
        row["common_checks"]["dynamic_orbit_without_explosion"] = (
            not terminated
            and row["pen_drift_mm"] < 30.0
            and row["max_linear_speed_m_s"] < 1.0
            and row["final_angular_speed_rad_s"] < 60.0
        )
    row["semantic_checks"] = phase_semantics(int(phase["phase"]), row)
    row["accepted"] = all(row["common_checks"].values()) and all(row["semantic_checks"].values())
    return row


def interpolate_states(start: dict[str, Any], end: dict[str, Any], samples: int = 21):
    start_hand = np.asarray(start["hand_qpos"], dtype=np.float64)
    end_hand = np.asarray(end["hand_qpos"], dtype=np.float64)
    start_pos = np.asarray(start["pen_pos"], dtype=np.float64)
    end_pos = np.asarray(end["pen_pos"], dtype=np.float64)
    start_quat = np.asarray(start["pen_quat"], dtype=np.float64)
    end_quat = np.asarray(end["pen_quat"], dtype=np.float64)
    for alpha in np.linspace(0.0, 1.0, samples):
        quat = (1.0 - alpha) * start_quat + alpha * end_quat
        quat /= np.linalg.norm(quat)
        yield {
            "hand_qpos": ((1.0 - alpha) * start_hand + alpha * end_hand).tolist(),
            "pen_pos": ((1.0 - alpha) * start_pos + alpha * end_pos).tolist(),
            "pen_quat": quat.tolist(),
        }


def evaluate_transition(
    env: Dex5PenSpinMinimalEnv,
    visual_model: Path,
    name: str,
    keyframes: list[dict[str, Any]],
    semantic_checks: dict[str, bool],
) -> dict[str, Any]:
    max_proxy_penetration = 0.0
    max_visual_penetration = 0.0
    minimum_joint_margin = float("inf")
    sampled_states = 0
    for segment_index in range(len(keyframes) - 1):
        for state in interpolate_states(keyframes[segment_index], keyframes[segment_index + 1]):
            set_state(env, state)
            _, _, penetration, _ = pen_contacts(env)
            max_proxy_penetration = max(max_proxy_penetration, penetration)
            max_visual_penetration = max(
                max_visual_penetration, visual_mesh_penetration(state, visual_model)
            )
            hand = np.asarray(state["hand_qpos"], dtype=np.float64)
            margin = np.minimum(hand - env.joint_low, env.joint_high - hand)
            minimum_joint_margin = min(minimum_joint_margin, float(np.min(margin)))
            sampled_states += 1
    common_checks = {
        "proxy_penetration_below_1mm": max_proxy_penetration * 1000.0 < 1.0,
        "visual_mesh_penetration_below_1mm": max_visual_penetration * 1000.0 < 1.0,
        "within_joint_limits": minimum_joint_margin >= -1e-6,
    }
    return {
        "name": name,
        "sampled_states": sampled_states,
        "max_proxy_penetration_mm": max_proxy_penetration * 1000.0,
        "max_visual_mesh_penetration_mm": max_visual_penetration * 1000.0,
        "minimum_joint_limit_margin_rad": minimum_joint_margin,
        "common_checks": common_checks,
        "semantic_checks": semantic_checks,
        "accepted": all(common_checks.values()) and all(semantic_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate measured-pen P1-P5 canonical grasps.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--visual-model", type=Path, default=DEFAULT_VISUAL_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settle-ms", type=float, default=100.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.008)
    parser.add_argument("--friction", type=float, default=0.6)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model,
        episode_seconds=max(0.25, args.settle_ms / 1000.0 + 0.05),
        init_noise_scale=0.0,
        disable_thumb_mesh_collision=False,
        disable_distal_mesh_collision=False,
    )
    configure_proxy_model(env, args.tip_proxy_radius, args.friction)
    rows = [evaluate_phase(env, args.visual_model, phase, args.settle_ms) for phase in manifest["phases"]]
    phases = {int(phase["phase"]): phase for phase in manifest["phases"]}
    p1 = phases[1]
    p2 = phases[2]
    p5 = phases[5]
    p2_frames = [p1, *p2["transition_keyframes"]]
    p3 = phases[3]
    p3_frames = [p2, *p3["transition_keyframes"]]
    p4 = phases[4]
    p4_frames = [p3, *p4["transition_keyframes"]]
    p5_frames = p5["transition_keyframes"]
    p1_hand = np.asarray(p1["hand_qpos"], dtype=np.float64)
    p2_semantics = {
        "thumb_index_middle_hold_exact": all(
            np.allclose(np.asarray(frame["hand_qpos"])[np.r_[0:8, 16:20]], p1_hand[np.r_[0:8, 16:20]])
            for frame in p2_frames
        ),
        "pen_pose_unchanged": all(
            np.allclose(frame["pen_pos"], p1["pen_pos"])
            and np.allclose(frame["pen_quat"], p1["pen_quat"])
            for frame in p2_frames
        ),
        "ring_only_moves": all(
            np.allclose(np.asarray(frame["hand_qpos"])[np.r_[0:8, 12:20]], p1_hand[np.r_[0:8, 12:20]])
            for frame in p2_frames
        ),
    }
    p5_start_hand = np.asarray(p5_frames[0]["hand_qpos"], dtype=np.float64)
    p5_end_hand = np.asarray(p5_frames[-1]["hand_qpos"], dtype=np.float64)
    p5_semantics = {
        "thumb_index_pinch_held_exact": all(
            np.allclose(
                np.asarray(frame["hand_qpos"])[np.r_[0:4, 16:20]],
                p5_start_hand[np.r_[0:4, 16:20]],
            )
            for frame in p5_frames
        ),
        "pen_pose_unchanged": all(
            np.allclose(frame["pen_pos"], p5_frames[0]["pen_pos"])
            and np.allclose(frame["pen_quat"], p5_frames[0]["pen_quat"])
            for frame in p5_frames
        ),
        "middle_ring_finish_at_p1": np.allclose(p5_end_hand[4:12], p1_hand[4:12]),
        "pinky_finish_at_p1": np.allclose(p5_end_hand[12:16], p1_hand[12:16]),
    }
    transitions = [
        evaluate_transition(env, args.visual_model, "P1_to_P2_ring_clearance_then_under", p2_frames, p2_semantics),
        evaluate_transition(
            env,
            args.visual_model,
            "P2_to_P3_index_outside_with_support_hold",
            p3_frames,
            {
                "outside_clearance_preserves_p2_support": np.allclose(
                    np.asarray(p3_frames[1]["hand_qpos"])[4:20],
                    np.asarray(p2["hand_qpos"])[4:20],
                ),
                "p3_ends_with_thumb_index_ring": set(p3["contact_set"]) == {"thumb", "index", "ring"},
                "staged_index_clearance_and_wrap": len(p3["transition_keyframes"]) >= 2,
            },
        ),
        evaluate_transition(
            env,
            args.visual_model,
            "P3_to_P4_middle_handoff_and_thumb_orbit",
            p4_frames,
            {
                "p3_starts_with_ring_support": "ring" in p3["contact_set"],
                "p4_ends_with_thumb_index_middle": set(p4["contact_set"]) == {"thumb", "index", "middle"},
                "p4_endpoint_is_explicit": len(p4["transition_keyframes"]) >= 1,
                "relative_rotation_between_45_and_65_deg": 45.0 <= abs(float(p4["phase_deg"])) <= 65.0,
            },
        ),
        evaluate_transition(env, args.visual_model, "P5_middle_ring_return_to_P1_side", p5_frames, p5_semantics),
    ]
    env.close()
    payload = {
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "representative_pen": {
            "mass_g": 9.4,
            "length_mm": 141.5,
            "diameters_mm": [11.2, 9.8, 9.5],
            "com_from_rear_mm": 61.4,
            "com_offset_toward_rear_mm": 9.35,
        },
        "gates": {
            "max_penetration_mm": 1.0,
            "settle_ms": args.settle_ms,
            "max_actuator_effort": 0.93,
            "max_settle_drift_mm": 15.0,
        },
        "joint_order": DEX5_JOINT_ORDER,
        "phases": rows,
        "transitions": transitions,
        "accepted_phases": [row["phase"] for row in rows if row["accepted"]],
        "accepted_transitions": [row["name"] for row in transitions if row["accepted"]],
        "all_accepted": all(row["accepted"] for row in rows) and all(row["accepted"] for row in transitions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
