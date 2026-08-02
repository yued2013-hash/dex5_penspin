from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

import build_m3_2_31_p2_bottom_cradle_b23_thumb_pass as m331
import build_m3_2_32_b12_ring_under_recapture as m332
import rebuild_m3_2_16_p4_p5_transitions as m316
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass(frozen=True)
class Perturbation:
    name: str
    hand_qpos: tuple[tuple[int, float], ...] = ()
    hand_qvel: tuple[tuple[int, float], ...] = ()
    pen_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pen_qvel: tuple[float, float, float, float, float, float] = (0.0,) * 6


@dataclass(frozen=True)
class B12Candidate:
    name: str
    brake_steps: int
    middle_scale: float
    thumb_scale: float
    ring_variant: int


@dataclass(frozen=True)
class B23Candidate:
    name: str
    index_scale: float
    middle_scale: float
    thumb_scale: float
    middle_distal: float
    ring_release: float
    stage_scale: float
    recapture_index: float = 0.0
    recapture_middle: float = 0.0
    recapture_thumb: float = 0.0
    recapture_steps: int = 10
    recapture_action_limit: float = 0.003
    recapture_gain: float = 0.32
    ring_prep_roll: float = 0.0
    ring_prep_proximal: float = 0.0
    ring_prep_distal: float = 0.0
    ring_action_limit: float = 0.003
    ring_gain: float = 0.60


RING_VARIANTS = (
    np.array([0.0, 0.0, 0.0], dtype=np.float64),
    np.array([-0.006, 0.012, 0.020], dtype=np.float64),
    np.array([0.006, -0.012, -0.020], dtype=np.float64),
)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def narrow_perturbations() -> list[Perturbation]:
    return [
        Perturbation("nominal"),
        Perturbation("pen_x_minus_0p25mm", pen_pos=(-0.00025, 0.0, 0.0)),
        Perturbation("pen_x_plus_0p25mm", pen_pos=(0.00025, 0.0, 0.0)),
        Perturbation("pen_y_minus_0p25mm", pen_pos=(0.0, -0.00025, 0.0)),
        Perturbation("pen_y_plus_0p25mm", pen_pos=(0.0, 0.00025, 0.0)),
        Perturbation("pen_z_minus_0p20mm", pen_pos=(0.0, 0.0, -0.00020)),
        Perturbation("pen_z_plus_0p20mm", pen_pos=(0.0, 0.0, 0.00020)),
        Perturbation("pen_vy_minus_3mm_s", pen_qvel=(0.0, -0.003, 0.0, 0.0, 0.0, 0.0)),
        Perturbation("pen_vy_plus_3mm_s", pen_qvel=(0.0, 0.003, 0.0, 0.0, 0.0, 0.0)),
        Perturbation("pen_vz_minus_2mm_s", pen_qvel=(0.0, 0.0, -0.002, 0.0, 0.0, 0.0)),
        Perturbation("pen_vz_plus_2mm_s", pen_qvel=(0.0, 0.0, 0.002, 0.0, 0.0, 0.0)),
        Perturbation("thumb_tracking_minus", hand_qpos=((17, -0.003), (18, -0.003))),
        Perturbation("thumb_tracking_plus", hand_qpos=((17, 0.003), (18, 0.003))),
        Perturbation("ring_tracking_minus", hand_qpos=((9, -0.003), (10, -0.003), (11, -0.003))),
        Perturbation("ring_tracking_plus", hand_qpos=((9, 0.003), (10, 0.003), (11, 0.003))),
        Perturbation("middle_velocity_minus", hand_qvel=((5, -0.04), (6, -0.03), (7, -0.03))),
        Perturbation("middle_velocity_plus", hand_qvel=((5, 0.04), (6, 0.03), (7, 0.03))),
    ]


def b12_candidates() -> list[B12Candidate]:
    result: list[B12Candidate] = []
    for brake_steps in (5, 6, 7):
        for middle_scale in (0.92, 1.0, 1.08):
            for thumb_scale in (0.94, 1.0, 1.06):
                for ring_variant in range(len(RING_VARIANTS)):
                    result.append(
                        B12Candidate(
                            name=(
                                f"bs{brake_steps}_m{middle_scale:.2f}_"
                                f"t{thumb_scale:.2f}_r{ring_variant}"
                            ),
                            brake_steps=brake_steps,
                            middle_scale=middle_scale,
                            thumb_scale=thumb_scale,
                            ring_variant=ring_variant,
                        )
                    )
    return result


def b23_candidates() -> list[B23Candidate]:
    result: list[B23Candidate] = []
    for index_scale in (1.15, 1.25, 1.35):
        for middle_scale in (1.15, 1.25, 1.35):
            for thumb_scale in (0.0, 0.20, 0.40):
                for middle_distal in (-0.035, -0.040, -0.045, -0.050):
                    for stage_scale in (0.92,):
                        result.append(
                            B23Candidate(
                                name=(
                                    f"i{index_scale:.2f}_m{middle_scale:.2f}_"
                                    f"t{thumb_scale:.2f}_d{middle_distal:.3f}_s{stage_scale:.2f}"
                                ),
                                index_scale=index_scale,
                                middle_scale=middle_scale,
                                thumb_scale=thumb_scale,
                                middle_distal=middle_distal,
                                ring_release=-0.020,
                                stage_scale=stage_scale,
                            )
                        )
    return result


def make_env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        action_delta_limit=0.040,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=0.0,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )


def reset_perturbed(
    env: Any,
    cache: Any,
    cache_index: int,
    perturbation: Perturbation,
    seed: int,
) -> None:
    m332.reset_from_cache(env, cache, cache_index, seed)
    for joint_index, delta in perturbation.hand_qpos:
        env.data.qpos[env.qpos_adr[joint_index]] += delta
    for joint_index, delta in perturbation.hand_qvel:
        env.data.qvel[env.qvel_adr[joint_index]] += delta
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] += np.asarray(
        perturbation.pen_pos, dtype=np.float64
    )
    env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6] += np.asarray(
        perturbation.pen_qvel, dtype=np.float64
    )
    mujoco.mj_forward(env.model, env.data)
    env.prev_phase = env._quat_twist_phase_about_x(env.data.xquat[env.pen_body_id])


def b12_targets(source_target: np.ndarray, candidate: B12Candidate) -> tuple[np.ndarray, np.ndarray]:
    brake = np.asarray(source_target, dtype=np.float64).copy()
    for joint_index, delta in m332.BRAKE_DELTA.items():
        scale = candidate.middle_scale if joint_index in (4, 5, 6) else candidate.thumb_scale
        brake[joint_index] += scale * delta
    brake[7] = brake[6]
    brake[8:12] = [
        m332.BRAKE_RING_Q[0],
        m332.BRAKE_RING_Q[1],
        m332.BRAKE_RING_Q[2],
        m332.BRAKE_RING_Q[2],
    ]
    brake = m316.enforce_shared_motor_consistency(brake)

    ring_q = m332.CAPTURE_RING_Q + RING_VARIANTS[candidate.ring_variant]
    capture = brake.copy()
    capture[8:12] = [ring_q[0], ring_q[1], ring_q[2], ring_q[2]]
    capture[16:20] += candidate.thumb_scale * m332.CAPTURE_THUMB_DELTA
    capture = m316.enforce_shared_motor_consistency(capture)
    return brake, capture


def ring_geometry(env: Any, geom_id: int) -> tuple[float, float]:
    state = env.get_state()
    relative = env.data.geom_xpos[geom_id] - state.pen_center
    radial = relative - float(np.dot(relative, state.pen_axis)) * state.pen_axis
    return float(radial[2] * 1000.0), float(np.linalg.norm(radial) * 1000.0)


def snapshot(env: Any, contacts: str, penetration_mm: float, t: float) -> dict[str, Any]:
    state = env.get_state()
    pen_qpos = env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7]
    return {
        "t": float(t),
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "target_qpos": env.target_qpos.copy(),
        "hand_qpos": state.qpos.copy(),
        "hand_qvel": state.qvel.copy(),
        "pen_pos": pen_qpos[:3].copy(),
        "pen_quat": pen_qpos[3:].copy(),
        "pen_qvel": env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6].copy(),
        "pen_center": state.pen_center.copy(),
        "pen_axis": state.pen_axis.copy(),
        "rotation_phase": float(state.rotation_phase),
        "contacts": contacts,
        "max_penetration_mm": float(penetration_mm),
    }


def b12_rollout(
    env: Any,
    cache: Any,
    cache_index: int,
    perturbation: Perturbation,
    candidate: B12Candidate,
    args: argparse.Namespace,
    *,
    duration_s: float,
    render: bool,
    reset_state: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if reset_state:
        reset_perturbed(env, cache, cache_index, perturbation, args.seed)
        source_target = cache["target_qpos"][cache_index].astype(np.float64)
    else:
        source_target = env.target_qpos.astype(np.float64).copy()
    brake_target, capture_target = b12_targets(source_target, candidate)
    ring_geom_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, m316.TIP_GEOMS["ring"]
    )
    initial_center = env.get_state().pen_center.copy()
    steps = int(round(duration_s * env.control_hz))
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    first_under_step: int | None = None
    max_penetration_mm = 0.0
    no_contact_gap = 0
    current_gap = 0
    hard_upper_contacts = 0
    terminated = False
    truncated = False

    if render:
        key_frames["01_B12_running_021"] = m316.overlay_frame(
            env.render(), f"B12 running | t=0.21s | {perturbation.name}"
        )

    for step in range(steps):
        brake_phase = step < candidate.brake_steps
        target = brake_target if brake_phase else capture_target
        action_limit = m332.BRAKE_ACTION_LIMIT if brake_phase else m332.CAPTURE_ACTION_LIMIT
        gain = m332.BRAKE_TRACKING_GAIN if brake_phase else m332.CAPTURE_TRACKING_GAIN
        action = np.clip(
            gain * (target - env.target_qpos.astype(np.float64)),
            -action_limit,
            action_limit,
        ).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        state = env.get_state()
        metrics = contact_metrics(env)
        fingers = set(metrics["fingers"])
        contacts = ",".join(sorted(fingers)) or "none"
        ring_rz_mm, ring_rn_mm = ring_geometry(env, ring_geom_id)
        penetration_mm = float(metrics["max_penetration"] * 1000.0)
        max_penetration_mm = max(max_penetration_mm, penetration_mm)
        current_gap = current_gap + 1 if not fingers else 0
        no_contact_gap = max(no_contact_gap, current_gap)
        ring_force = float(metrics["normal_force"].get("ring", 0.0))
        if "ring" in fingers and ring_rz_mm < 0.0 and first_under_step is None:
            first_under_step = step
        if "ring" in fingers and ring_rz_mm > 1.0 and ring_force > 0.10:
            hard_upper_contacts += 1

        displacement = (state.pen_center - initial_center) * 1000.0
        linear_speed = float(
            np.linalg.norm(env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3])
        )
        angular_speed = float(
            np.linalg.norm(env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6])
        )
        row: dict[str, Any] = {
            "step": step,
            "t": float((step + 1) / env.control_hz),
            "stage": "thumb_brake" if brake_phase else "ring_under_recapture",
            "contacts": contacts,
            "dx_mm": float(displacement[0]),
            "dy_mm": float(displacement[1]),
            "dz_mm": float(displacement[2]),
            "pen_linear_speed": linear_speed,
            "pen_angular_speed": angular_speed,
            "ring_radial_z_mm": ring_rz_mm,
            "ring_radial_norm_mm": ring_rn_mm,
            "max_penetration_mm": penetration_mm,
        }
        for finger in FINGERS:
            row[f"{finger}_share"] = float(metrics["force_share"].get(finger, 0.0))
        rows.append(row)
        snapshots.append(snapshot(env, contacts, penetration_mm, row["t"]))

        if render:
            label = (
                f"{row['stage']} | +{row['t']:.2f}s | {contacts} | "
                f"thumb/ring={row['thumb_share']:.2f}/{row['ring_share']:.2f}"
            )
            if step % args.render_every == 0 or step == steps - 1:
                frames.append(m316.overlay_frame(env.render(), label))
            if step == candidate.brake_steps - 1:
                key_frames["02_thumb_brake"] = m316.overlay_frame(env.render(), label)
            if first_under_step == step:
                key_frames["03_ring_under_touch"] = m316.overlay_frame(env.render(), label)
            if step == steps - 1:
                key_frames["04_projected_P2_arrival"] = m316.overlay_frame(env.render(), label)

        if terminated or truncated:
            break

    sample_count = len(rows)
    final_window = rows[-min(12, sample_count) :]
    brake_row = rows[min(candidate.brake_steps - 1, sample_count - 1)]

    def ratio(finger: str) -> float:
        return float(
            sum(finger in row["contacts"].split(",") for row in final_window)
            / max(1, len(final_window))
        )

    def share(finger: str) -> float:
        return float(
            sum(float(row[f"{finger}_share"]) for row in final_window)
            / max(1, len(final_window))
        )

    final_contacts = [] if not rows or rows[-1]["contacts"] == "none" else rows[-1]["contacts"].split(",")
    brake_ok = bool(
        sample_count >= candidate.brake_steps
        and brake_row["pen_linear_speed"] <= 0.018
        and brake_row["thumb_share"] >= 0.65
        and brake_row["middle_share"] <= 0.20
    )
    recapture_ok = bool(
        not terminated
        and first_under_step is not None
        and first_under_step <= 28
        and no_contact_gap <= 2
        and hard_upper_contacts == 0
        and max_penetration_mm <= 4.0
    )
    arrival_ok = bool(
        sample_count == steps
        and ratio("thumb") >= 0.85
        and ratio("ring") >= 0.70
        and share("thumb") >= 0.48
        and share("ring") <= 0.38
        and "thumb" in final_contacts
        and "ring" in final_contacts
        and rows[-1]["ring_radial_z_mm"] < 0.0
        and rows[-1]["pen_linear_speed"] <= 0.035
    )
    summary = {
        "perturbation": perturbation.name,
        "candidate": candidate.name,
        "survival_s": float(sample_count / env.control_hz),
        "first_ring_under_s": (
            None if first_under_step is None else float((first_under_step + 1) / env.control_hz)
        ),
        "no_contact_gap_steps": int(no_contact_gap),
        "hard_upper_contacts": int(hard_upper_contacts),
        "max_penetration_mm": float(max_penetration_mm),
        "final_contacts": final_contacts,
        "final_linear_speed_m_s": float(rows[-1]["pen_linear_speed"]),
        "final_angular_speed_rad_s": float(rows[-1]["pen_angular_speed"]),
        "final_window_contact_ratio": {finger: ratio(finger) for finger in FINGERS},
        "final_window_force_share": {finger: share(finger) for finger in FINGERS},
        "brake": {
            "linear_speed_m_s": float(brake_row["pen_linear_speed"]),
            "thumb_share": float(brake_row["thumb_share"]),
            "middle_share": float(brake_row["middle_share"]),
        },
        "acceptance": {
            "thumb_brake": brake_ok,
            "ring_under_recapture": recapture_ok,
            "projected_p2_arrival": arrival_ok,
        },
        "passed": bool(brake_ok and recapture_ok and arrival_ok),
    }
    trace = {
        "rows": rows,
        "snapshots": snapshots,
        "frames": frames,
        "key_frames": key_frames,
        "brake_target": brake_target,
        "capture_target": capture_target,
    }
    return summary, trace


def b12_score(results: list[dict[str, Any]]) -> float:
    score = 0.0
    for result in results:
        acceptance = result["acceptance"]
        score += 1.5 * float(acceptance["thumb_brake"])
        score += 2.5 * float(acceptance["ring_under_recapture"])
        score += 4.0 * float(acceptance["projected_p2_arrival"])
        score -= 0.10 * max(0.0, result["max_penetration_mm"] - 2.0)
        score -= 3.0 * max(0.0, result["final_linear_speed_m_s"] - 0.015)
    passed = sum(result["passed"] for result in results)
    return float(score + 5.0 * passed)


def search_b12(
    env: Any,
    cache: Any,
    cache_index: int,
    perturbations: list[Perturbation],
    args: argparse.Namespace,
) -> tuple[B12Candidate, list[dict[str, Any]], list[dict[str, Any]]]:
    leaderboard: list[dict[str, Any]] = []
    best_candidate: B12Candidate | None = None
    best_results: list[dict[str, Any]] = []
    best_key: tuple[int, float] = (-1, -math.inf)
    candidates = b12_candidates()
    for candidate_index, candidate in enumerate(candidates):
        results = [
            b12_rollout(
                env,
                cache,
                cache_index,
                perturbation,
                candidate,
                args,
                duration_s=args.b12_search_duration,
                render=False,
            )[0]
            for perturbation in perturbations
        ]
        passed = int(sum(result["passed"] for result in results))
        score = b12_score(results)
        leaderboard.append(
            {
                "candidate": candidate.name,
                "passed": passed,
                "total": len(results),
                "score": score,
                "worst_penetration_mm": max(row["max_penetration_mm"] for row in results),
                "worst_final_speed_m_s": max(row["final_linear_speed_m_s"] for row in results),
            }
        )
        key = (passed, score)
        if key > best_key:
            best_key = key
            best_candidate = candidate
            best_results = results
        if args.verbose and (candidate_index + 1) % 12 == 0:
            print(f"B12 search {candidate_index + 1}/{len(candidates)} best={best_key}", flush=True)
    if best_candidate is None:
        raise RuntimeError("B12 search produced no candidate")
    leaderboard.sort(key=lambda row: (row["passed"], row["score"]), reverse=True)
    return best_candidate, best_results, leaderboard


def reset_snapshot(env: Any, state: dict[str, Any], seed: int) -> None:
    env.reset(seed=seed)
    env.data.qpos[:] = state["qpos"]
    env.data.qvel[:] = state["qvel"]
    env.target_qpos = np.asarray(state["target_qpos"], dtype=np.float32).copy()
    env.data.ctrl[:] = env.target_qpos
    env.last_action[:] = 0.0
    env.step_count = 0
    env.rotation_phase = float(state["rotation_phase"])
    mujoco.mj_forward(env.model, env.data)
    env.prev_phase = env._quat_twist_phase_about_x(env.data.xquat[env.pen_body_id])


def scaled_delta(
    raw: np.ndarray,
    candidate: B23Candidate,
    *,
    ring_release: bool,
) -> np.ndarray:
    delta = np.asarray(raw, dtype=np.float64).copy()
    delta[0:4] *= candidate.index_scale
    delta[4:8] *= candidate.middle_scale
    delta[12:16] *= 0.75 * candidate.middle_scale
    delta[16:20] *= candidate.thumb_scale
    delta[8:12] = 0.0
    if ring_release:
        delta[9] -= 0.45 * candidate.ring_release
        delta[10] -= candidate.ring_release
        delta[11] = delta[10]
    return m316.enforce_shared_motor_consistency(delta)


def b23_targets(
    base_target: np.ndarray,
    reference_p2: np.ndarray,
    references: dict[str, np.ndarray],
    candidate: B23Candidate,
) -> list[tuple[str, int, np.ndarray]]:
    approach_delta = references["b23a"] - reference_p2
    approach_profile = B23Candidate(
        "contact_profile", 0.60, 0.68, 0.58, 0.0, 0.0, candidate.stage_scale
    )
    approach_full = base_target + scaled_delta(
        approach_delta, approach_profile, ring_release=False
    )
    contact = base_target + 0.72 * (approach_full - base_target)
    # The old static-P2 drive delta opens the distal links at the wrong pen-axis
    # orientation. Build a local tangential push from the established contact
    # state instead: proximal index/middle push while distal closure is retained.
    drive_extra = np.zeros_like(base_target)
    drive_extra[0] = -0.012 * candidate.index_scale
    drive_extra[1] = 0.075 * candidate.index_scale
    drive_extra[2] = 0.012 * candidate.index_scale
    drive_extra[3] = drive_extra[2]
    drive_extra[4] = -0.012 * candidate.middle_scale
    drive_extra[5] = 0.095 * candidate.middle_scale
    drive_extra[6] = candidate.middle_distal * candidate.middle_scale
    drive_extra[7] = drive_extra[6]
    drive_extra[9] = -0.35 * candidate.ring_release
    drive_extra[10] = -candidate.ring_release
    drive_extra[11] = drive_extra[10]
    drive_extra[16] = 0.004 * candidate.thumb_scale
    drive_extra[17] = -0.018 * candidate.thumb_scale
    drive_extra[18] = -0.010 * candidate.thumb_scale
    drive_extra[19] = -0.006 * candidate.thumb_scale
    drive = m316.enforce_shared_motor_consistency(contact + drive_extra)
    ring_prep_delta = np.asarray(
        [
            candidate.ring_prep_roll,
            candidate.ring_prep_proximal,
            candidate.ring_prep_distal,
            candidate.ring_prep_distal,
        ],
        dtype=np.float64,
    )
    drive_targets = []
    for fraction in (0.10, 0.20, 0.30, 0.40, 0.50):
        stage_target = contact + fraction * (drive - contact)
        # Ring preparation is normalized over the five drive stages so the
        # ring can move early without changing the index/middle push profile.
        stage_target[8:12] = contact[8:12] + (fraction / 0.50) * ring_prep_delta
        drive_targets.append((fraction, stage_target))
    stages = [
        ("P2_dynamic_settle", 5, base_target.copy()),
        ("B23_index_front_contact", 13, contact),
        ("B23_front_contact_dwell", 7, contact),
    ]
    stages.extend(
        (f"B23_drive_{int(round(fraction * 1000)):03d}permille", 8, target)
        for fraction, target in drive_targets
    )
    recapture = drive_targets[-1][1].copy()
    recapture[1] += candidate.recapture_index
    recapture[2] += 0.35 * candidate.recapture_index
    recapture[3] = recapture[2]
    recapture[5] += candidate.recapture_middle
    recapture[6] += 0.35 * candidate.recapture_middle
    recapture[7] = recapture[6]
    recapture[17] += 0.50 * candidate.recapture_thumb
    recapture[18] -= candidate.recapture_thumb
    recapture[19] -= 0.40 * candidate.recapture_thumb
    stages.append(
        (
            "P3_dynamic_recapture",
            candidate.recapture_steps,
            m316.enforce_shared_motor_consistency(recapture),
        )
    )
    scaled: list[tuple[str, int, np.ndarray]] = []
    for label, steps, target in stages:
        scaled_steps = max(2, int(round(steps * candidate.stage_scale)))
        scaled.append((label, scaled_steps, m316.enforce_shared_motor_consistency(target)))
    return scaled


def signed_planar_rotation_deg(initial_axis: np.ndarray, final_axis: np.ndarray) -> float:
    initial = math.atan2(float(initial_axis[1]), float(initial_axis[0]))
    final = math.atan2(float(final_axis[1]), float(final_axis[0]))
    return float(math.degrees((final - initial + math.pi) % (2.0 * math.pi) - math.pi))


def b23_rollout(
    env: Any,
    arrival: dict[str, Any],
    references: dict[str, np.ndarray],
    candidate: B23Candidate,
    args: argparse.Namespace,
    *,
    label: str,
    render: bool,
    reset_state: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if reset_state:
        reset_snapshot(env, arrival, args.seed)
        base_target = np.asarray(arrival["target_qpos"], dtype=np.float64)
    else:
        base_target = env.target_qpos.astype(np.float64).copy()
    stages = b23_targets(base_target, references["p2"], references, candidate)
    initial_axis = env.get_state().pen_axis.copy()
    initial_center = env.get_state().pen_center.copy()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    stage_windows: dict[str, tuple[int, int]] = {}
    max_penetration_mm = 0.0
    terminated = False
    truncated = False
    global_step = 0
    safe_drive_target = base_target.copy()

    if render:
        key_frames["05_projected_P2_dynamic"] = m316.overlay_frame(
            env.render(), f"projected P2 dynamic | {label}"
        )

    for stage_name, stage_steps, target in stages:
        stage_start = global_step
        for local_step in range(stage_steps):
            if stage_name == "P2_dynamic_settle":
                gain = 0.62
                action_limit = 0.026
            elif stage_name == "P3_dynamic_recapture":
                gain = candidate.recapture_gain
                action_limit = candidate.recapture_action_limit
            elif stage_name.startswith("B23_drive"):
                gain = 0.32
                action_limit = 0.003
            else:
                gain = 0.72
                action_limit = 0.022
            pre_metrics = contact_metrics(env)
            pre_fingers = set(pre_metrics["fingers"])
            drive_phase = stage_name.startswith("B23_drive") or stage_name == "P3_dynamic_recapture"
            effective_target = target
            if drive_phase and not {"thumb", "index", "middle"}.issubset(pre_fingers):
                effective_target = safe_drive_target
            action = np.clip(
                gain * (effective_target - env.target_qpos.astype(np.float64)),
                -action_limit,
                action_limit,
            ).astype(np.float32)
            if drive_phase and candidate.ring_action_limit > action_limit:
                ring_action = np.clip(
                    candidate.ring_gain
                    * (target[8:12] - env.target_qpos[8:12].astype(np.float64)),
                    -candidate.ring_action_limit,
                    candidate.ring_action_limit,
                )
                action[8:12] = ring_action.astype(np.float32)
            pre_penetration_mm = float(pre_metrics["max_penetration"] * 1000.0)
            pre_linear_speed = float(
                np.linalg.norm(env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3])
            )
            pre_angular_speed = float(
                np.linalg.norm(env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6])
            )
            if drive_phase:
                if pre_penetration_mm > 1.4:
                    action[[1, 2, 3, 5, 6, 7]] *= 0.20
                if pre_linear_speed > 0.10 or pre_angular_speed > 12.0:
                    action *= 0.15
            _, _, terminated, truncated, _ = env.step(action)
            state = env.get_state()
            metrics = contact_metrics(env)
            fingers = set(metrics["fingers"])
            contacts = ",".join(sorted(fingers)) or "none"
            penetration_mm = float(metrics["max_penetration"] * 1000.0)
            max_penetration_mm = max(max_penetration_mm, penetration_mm)
            displacement = (state.pen_center - initial_center) * 1000.0
            row: dict[str, Any] = {
                "step": global_step,
                "t": float((global_step + 1) / env.control_hz),
                "stage": stage_name,
                "contacts": contacts,
                "dx_mm": float(displacement[0]),
                "dy_mm": float(displacement[1]),
                "dz_mm": float(displacement[2]),
                "max_penetration_mm": penetration_mm,
                "pen_linear_speed": float(
                    np.linalg.norm(env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3])
                ),
                "pen_angular_speed": float(
                    np.linalg.norm(env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6])
                ),
                "planar_rotation_deg": signed_planar_rotation_deg(initial_axis, state.pen_axis),
            }
            for finger in FINGERS:
                row[f"{finger}_share"] = float(metrics["force_share"].get(finger, 0.0))
            rows.append(row)
            snapshots.append(snapshot(env, contacts, penetration_mm, row["t"]))
            if (
                {"thumb", "index", "middle"}.issubset(fingers)
                and row["pen_linear_speed"] < 0.05
                and row["pen_angular_speed"] < 6.0
            ):
                safe_drive_target = env.target_qpos.astype(np.float64).copy()

            if render:
                frame_label = (
                    f"{stage_name} | +{row['t']:.2f}s | {contacts} | "
                    f"thumb/index/middle={row['thumb_share']:.2f}/"
                    f"{row['index_share']:.2f}/{row['middle_share']:.2f}"
                )
                if global_step % args.render_every == 0:
                    frames.append(m316.overlay_frame(env.render(), frame_label))
                if local_step == stage_steps - 1:
                    key_frames[f"b23_{len(key_frames) + 1:02d}_{stage_name}"] = m316.overlay_frame(
                        env.render(), frame_label
                    )
            global_step += 1
            if terminated or truncated:
                break
        stage_windows[stage_name] = (stage_start, global_step)
        if terminated or truncated:
            break

    final_axis = env.get_state().pen_axis.copy()
    rotation_deg = signed_planar_rotation_deg(initial_axis, final_axis)

    def stage_ratio(stage_name: str, finger: str) -> float:
        start, end = stage_windows.get(stage_name, (0, 0))
        window = rows[start:end]
        return float(
            sum(finger in row["contacts"].split(",") for row in window) / max(1, len(window))
        )

    def prefix_ratio(prefix: str, finger: str) -> float:
        window = [row for row in rows if str(row["stage"]).startswith(prefix)]
        return float(
            sum(finger in row["contacts"].split(",") for row in window) / max(1, len(window))
        )

    def all_ratio(finger: str) -> float:
        return float(
            sum(finger in row["contacts"].split(",") for row in rows) / max(1, len(rows))
        )

    deflect_index = stage_ratio("B23_front_contact_dwell", "index")
    drive_index = prefix_ratio("B23_drive_", "index")
    drive_middle = prefix_ratio("B23_drive_", "middle")
    drive_thumb = prefix_ratio("B23_drive_", "thumb")
    p3_thumb = stage_ratio("P3_dynamic_recapture", "thumb")
    p3_index = stage_ratio("P3_dynamic_recapture", "index")
    p3_middle = stage_ratio("P3_dynamic_recapture", "middle")
    p3_support = min(p3_thumb, p3_index, p3_middle)
    expected_steps = sum(steps for _, steps, _ in stages)
    short_entry_passed = bool(
        len(rows) == expected_steps
        and not terminated
        and deflect_index >= 0.45
        and drive_index >= 0.65
        and drive_middle >= 0.65
        and drive_thumb >= 0.10
        and p3_support >= 0.65
        and abs(rotation_deg) >= 4.0
        and max_penetration_mm <= 4.0
        and min(row["dz_mm"] for row in rows) >= -8.0
    )
    full_angle_passed = bool(short_entry_passed and abs(rotation_deg) >= 7.0)
    summary = {
        "label": label,
        "candidate": candidate.name,
        "survival_s": float(len(rows) / env.control_hz),
        "terminated": bool(terminated),
        "planar_rotation_deg": rotation_deg,
        "max_penetration_mm": float(max_penetration_mm),
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "final_contacts": (
            [] if not rows or rows[-1]["contacts"] == "none" else rows[-1]["contacts"].split(",")
        ),
        "contact_ratio": {finger: all_ratio(finger) for finger in FINGERS},
        "phase_contact_ratio": {
            "index_deflect": deflect_index,
            "drive_index": drive_index,
            "drive_middle": drive_middle,
            "drive_thumb": drive_thumb,
            "p3_support": p3_support,
            "p3_thumb": p3_thumb,
            "p3_index": p3_index,
            "p3_middle": p3_middle,
        },
        "passed": short_entry_passed,
        "full_angle_passed": full_angle_passed,
    }
    return summary, {
        "rows": rows,
        "snapshots": snapshots,
        "frames": frames,
        "key_frames": key_frames,
        "stages": stages,
    }


def b23_score(results: list[dict[str, Any]]) -> float:
    score = 0.0
    for result in results:
        ratios = result["phase_contact_ratio"]
        score += 2.0 * min(1.0, ratios["index_deflect"])
        score += 2.5 * min(1.0, ratios["drive_index"])
        score += 2.5 * min(1.0, ratios["drive_middle"])
        score += 1.2 * min(1.0, ratios["drive_thumb"])
        score += 1.5 * min(1.0, ratios["p3_support"])
        score += min(2.0, abs(result["planar_rotation_deg"]) / 8.0)
        score -= 0.25 * max(0.0, result["max_penetration_mm"] - 2.0)
        score -= 0.20 * max(0.0, -result["min_dz_mm"] - 3.0)
        score += 5.0 * float(result["passed"])
    return float(score)


def search_b23(
    env: Any,
    arrivals: list[tuple[str, dict[str, Any]]],
    references: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[B23Candidate, list[dict[str, Any]], list[dict[str, Any]]]:
    leaderboard: list[dict[str, Any]] = []
    best_candidate: B23Candidate | None = None
    best_results: list[dict[str, Any]] = []
    best_key: tuple[int, float] = (-1, -math.inf)
    candidates = b23_candidates()
    for candidate_index, candidate in enumerate(candidates):
        results = [
            b23_rollout(
                env,
                arrival,
                references,
                candidate,
                args,
                label=label,
                render=False,
            )[0]
            for label, arrival in arrivals
        ]
        passed = int(sum(row["passed"] for row in results))
        score = b23_score(results)
        leaderboard.append(
            {
                "candidate": candidate.name,
                "passed": passed,
                "full_angle_passed": int(
                    sum(row["full_angle_passed"] for row in results)
                ),
                "total": len(results),
                "score": score,
                "min_abs_rotation_deg": min(abs(row["planar_rotation_deg"]) for row in results),
                "worst_penetration_mm": max(row["max_penetration_mm"] for row in results),
            }
        )
        key = (passed, score)
        if key > best_key:
            best_key = key
            best_candidate = candidate
            best_results = results
        if args.verbose and (candidate_index + 1) % 12 == 0:
            print(f"B23 search {candidate_index + 1}/{len(candidates)} best={best_key}", flush=True)
    if best_candidate is None:
        raise RuntimeError("B23 search produced no candidate")
    leaderboard.sort(key=lambda row: (row["passed"], row["score"]), reverse=True)
    return best_candidate, best_results, leaderboard


def save_snapshot_cache(path: Path, named_states: list[tuple[str, dict[str, Any]]]) -> None:
    np.savez_compressed(
        path,
        label=np.asarray([label for label, _ in named_states]),
        t=np.asarray([state["t"] for _, state in named_states], dtype=np.float64),
        qpos=np.asarray([state["qpos"] for _, state in named_states], dtype=np.float64),
        qvel=np.asarray([state["qvel"] for _, state in named_states], dtype=np.float64),
        target_qpos=np.asarray(
            [state["target_qpos"] for _, state in named_states], dtype=np.float64
        ),
        hand_qpos=np.asarray([state["hand_qpos"] for _, state in named_states], dtype=np.float64),
        hand_qvel=np.asarray([state["hand_qvel"] for _, state in named_states], dtype=np.float64),
        pen_pos=np.asarray([state["pen_pos"] for _, state in named_states], dtype=np.float64),
        pen_quat=np.asarray([state["pen_quat"] for _, state in named_states], dtype=np.float64),
        pen_qvel=np.asarray([state["pen_qvel"] for _, state in named_states], dtype=np.float64),
        pen_center=np.asarray([state["pen_center"] for _, state in named_states], dtype=np.float64),
        pen_axis=np.asarray([state["pen_axis"] for _, state in named_states], dtype=np.float64),
        rotation_phase=np.asarray(
            [state["rotation_phase"] for _, state in named_states], dtype=np.float64
        ),
        contacts=np.asarray([state["contacts"] for _, state in named_states]),
        max_penetration_mm=np.asarray(
            [state["max_penetration_mm"] for _, state in named_states], dtype=np.float64
        ),
    )


def save_trace_plot(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    if plt is None or not rows:
        return
    t = np.asarray([row["t"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, [row["dx_mm"] for row in rows], label="dx")
    axes[0].plot(t, [row["dy_mm"] for row in rows], label="dy")
    axes[0].plot(t, [row["dz_mm"] for row in rows], label="dz")
    axes[0].set_ylabel("pen delta (mm)")
    axes[0].legend(ncol=3)
    for finger in FINGERS:
        axes[1].plot(t, [row[f"{finger}_share"] for row in rows], label=finger)
    axes[1].set_ylabel("force share")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(ncol=5)
    axes[2].plot(t, [row["pen_linear_speed"] for row in rows], label="linear speed")
    axes[2].plot(t, [row["pen_angular_speed"] for row in rows], label="angular speed")
    axes[2].plot(t, [row["max_penetration_mm"] for row in rows], label="penetration mm")
    axes[2].set_ylabel("speed / penetration")
    axes[2].set_xlabel("primitive time (s)")
    axes[2].legend(ncol=3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def copy_existing_assets(output_dir: Path) -> None:
    sources = {
        "P3_to_P4_existing.gif": (
            "dex5_penspin/reports/manus_mujoco_presets/rebuilt_p4_p5_transitions_m3_2_16/"
            "transition_P3_B34_P4.gif"
        ),
        "P4_to_P5_existing.gif": (
            "dex5_penspin/reports/manus_mujoco_presets/rebuilt_p4_p5_transitions_m3_2_16/"
            "transition_P4_B45_P5.gif"
        ),
        "P5_to_P1_existing.gif": (
            "dex5_penspin/reports/manus_mujoco_presets/cycle_closure_eval_m3_2_17/"
            "transition_P5_to_P1_cycle_entry.gif"
        ),
    }
    for filename, source in sources.items():
        shutil.copy2(Path(source), output_dir / filename)


def write_visual_summary(output_dir: Path, report: dict[str, Any]) -> None:
    b12 = report["b12_narrow"]
    b23 = report["b23_dynamic"]
    rows = report["five_step_status"]
    table_rows = "\n".join(
        f"<tr><td>{row['leg']}</td><td>{row['status']}</td><td>{row['evidence']}</td></tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3.2.33 narrow 0.21s five-step chain</title>
<style>
body {{ margin:0; background:#f4f5f7; color:#202124; font-family:sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:24px; }} p {{ line-height:1.55; }}
.status {{ padding:12px 0; border-top:1px solid #c7cbd1; border-bottom:1px solid #c7cbd1; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:18px; }}
figure {{ margin:0; overflow:hidden; background:white; border:1px solid #d6dae0; border-radius:6px; }}
img {{ display:block; width:100%; height:auto; }} figcaption {{ padding:10px 12px; line-height:1.45; font-size:14px; }}
table {{ width:100%; margin-top:18px; border-collapse:collapse; background:white; }}
th,td {{ padding:8px 10px; border:1px solid #d6dae0; text-align:left; font-size:14px; }}
@media(max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:14px; }} }}
</style></head><body><main>
<h1>0.21s 窄入口五步真实动作链</h1>
<p class="status">B12 窄扰动通过 <strong>{b12['pass_count']}/{b12['total']}</strong>；动态 B23/P3-entry 通过 <strong>{b23['pass_count']}/{b23['total']}</strong>，其中完整 7° 偏转 <strong>{b23['full_angle_pass_count']}/{b23['total']}</strong>。本页按短 primitive 分段确认，不把它描述为完整连续转笔。</p>
<table><tr><th>动作段</th><th>状态</th><th>证据</th></tr>{table_rows}</table>
<div class="grid">
<figure><img src="B12_021_narrow_recapture.gif"><figcaption>0.21s running-state：thumb 先刹停/上托，ring 外撤后从下侧再捕获，形成动态 P2 入口。</figcaption></figure>
<figure><img src="B12_021_narrow_trace.png"><figcaption>B12 代表轨迹：笔位移、承力分配、速度和接触穿透。</figcaption></figure>
<figure><img src="P2_to_P3_dynamic_B23.gif"><figcaption>从保留 qvel 的 projected P2 直接进入 B23：index 前绕偏转，index/middle 推滚，thumb 避让但维持支撑。</figcaption></figure>
<figure><img src="P2_to_P3_dynamic_B23_trace.png"><figcaption>动态 B23 代表轨迹与分阶段接触/承力变化。</figcaption></figure>
<figure><img src="P3_to_P4_existing.gif"><figcaption>现有 P3→B34→P4 短 primitive 复核资产。</figcaption></figure>
<figure><img src="P4_to_P5_existing.gif"><figcaption>现有 P4→B45→P5 短 primitive；当前穿透指标仍需继续压低。</figcaption></figure>
<figure><img src="P5_to_P1_existing.gif"><figcaption>现有 P5→P1-cycle-entry：thumb/index 支撑可连，strict ring 回接仍未完成。</figcaption></figure>
</div></main></body></html>"""
    (output_dir / "visual_summary.html").write_text(html, encoding="utf-8")


def load_existing_rows() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_316 = json.loads(
        Path(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "rebuilt_p4_p5_transitions_m3_2_16/manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest_317 = json.loads(
        Path(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "cycle_closure_eval_m3_2_17/manifest.json"
        ).read_text(encoding="utf-8")
    )
    eval_by_label = {row["label"]: row for row in manifest_316["evaluation"]}
    transition_by_label = {row["label"]: row for row in manifest_317["transition_rows"]}
    return (
        eval_by_label["transition_P3_B34_P4"],
        eval_by_label["transition_P4_B45_P5"],
        transition_by_label["transition_P5_to_P1_cycle_entry"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/"
            "dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
        ),
    )
    parser.add_argument(
        "--model-seed",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/p1_p2_micro_rebuild_m3_2_18/"
            "P2_ring_wrap_to_front_micro_ring_guarded_seed.json"
        ),
    )
    parser.add_argument(
        "--b23-reference-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p2_bottom_cradle_b23_thumb_pass_m3_2_31"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "narrow_021_five_step_chain_m3_2_33"
        ),
    )
    parser.add_argument("--source-t", type=float, default=0.21)
    parser.add_argument("--b12-search-duration", type=float, default=0.58)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--reuse-selected-b12", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(Path(args.cache))
    cache_index = m332.cache_index(cache, args.source_t)
    model_seed = json.loads(Path(args.model_seed).read_text(encoding="utf-8"))
    perturbations = narrow_perturbations()
    env_args = make_env_args(args)
    search_env = m316.make_env(model_seed, env_args, duration=1.0, render=False)

    reference_dir = Path(args.b23_reference_dir)
    reference_files = {
        "p2": "P2_ring_bottom_thumb_cradle_support_seed.json",
        "b23a": "B23a_index_front_deflect_seed.json",
        "b23a_contact": "B23a_index_front_contact_deflect_seed.json",
        "b23b": "B23b_index_middle_drive_thumb_yield_seed.json",
        "p3": "P3_index_front_pass_exit_seed.json",
    }
    references = {
        key: np.asarray(
            json.loads((reference_dir / filename).read_text(encoding="utf-8"))["hand_qpos"],
            dtype=np.float64,
        )
        for key, filename in reference_files.items()
    }

    try:
        if args.reuse_selected_b12:
            best_b12 = B12Candidate("bs6_m1.00_t1.06_r0", 6, 1.0, 1.06, 0)
            b12_search_results = [
                b12_rollout(
                    search_env,
                    cache,
                    cache_index,
                    perturbation,
                    best_b12,
                    args,
                    duration_s=args.b12_search_duration,
                    render=False,
                )[0]
                for perturbation in perturbations
            ]
            b12_leaderboard = [
                {
                    "candidate": best_b12.name,
                    "passed": int(sum(row["passed"] for row in b12_search_results)),
                    "total": len(b12_search_results),
                    "score": b12_score(b12_search_results),
                }
            ]
        else:
            best_b12, b12_search_results, b12_leaderboard = search_b12(
                search_env, cache, cache_index, perturbations, args
            )
        selected_b12_results: list[dict[str, Any]] = []
        arrival_states: list[tuple[str, dict[str, Any]]] = []
        for perturbation in perturbations:
            result, trace = b12_rollout(
                search_env,
                cache,
                cache_index,
                perturbation,
                best_b12,
                args,
                duration_s=args.b12_search_duration,
                render=False,
            )
            selected_b12_results.append(result)
            if result["passed"]:
                arrival_states.append((perturbation.name, trace["snapshots"][-1]))

        if not arrival_states:
            nominal_result, nominal_trace = b12_rollout(
                search_env,
                cache,
                cache_index,
                perturbations[0],
                best_b12,
                args,
                duration_s=args.b12_search_duration,
                render=False,
            )
            arrival_states = [("nominal_fallback", nominal_trace["snapshots"][-1])]

        representative_arrivals: list[tuple[str, dict[str, Any]]] = []
        for preferred in ("nominal", "pen_y_minus_0p25mm", "thumb_tracking_minus"):
            match = next((row for row in arrival_states if row[0] == preferred), None)
            if match is not None:
                representative_arrivals.append(match)
        for row in arrival_states:
            if len(representative_arrivals) >= 3:
                break
            if row[0] not in {label for label, _ in representative_arrivals}:
                representative_arrivals.append(row)

        best_b23, b23_search_results, b23_leaderboard = search_b23(
            search_env, representative_arrivals, references, args
        )
        b23_validation_results: list[dict[str, Any]] = []
        p3_arrival_states: list[tuple[str, dict[str, Any]]] = []
        for label, arrival in arrival_states:
            result, trace = b23_rollout(
                search_env,
                arrival,
                references,
                best_b23,
                args,
                label=label,
                render=False,
            )
            b23_validation_results.append(result)
            if result["passed"] and trace["snapshots"]:
                p3_arrival_states.append((label, trace["snapshots"][-1]))
    finally:
        search_env.close()

    render_env = m316.make_env(model_seed, env_args, duration=1.2, render=True)
    try:
        nominal_perturbation = perturbations[0]
        nominal_b12, nominal_b12_trace = b12_rollout(
            render_env,
            cache,
            cache_index,
            nominal_perturbation,
            best_b12,
            args,
            duration_s=args.b12_search_duration,
            render=True,
        )
        nominal_arrival = nominal_b12_trace["snapshots"][-1]
        nominal_b23, nominal_b23_trace = b23_rollout(
            render_env,
            nominal_arrival,
            references,
            best_b23,
            args,
            label="nominal_021_arrival",
            render=True,
        )
    finally:
        render_env.close()

    if nominal_b12_trace["frames"]:
        imageio.mimsave(
            output_dir / "B12_021_narrow_recapture.gif",
            nominal_b12_trace["frames"],
            duration=args.render_every / 100.0,
        )
    if nominal_b23_trace["frames"]:
        imageio.mimsave(
            output_dir / "P2_to_P3_dynamic_B23.gif",
            nominal_b23_trace["frames"],
            duration=args.render_every / 100.0,
        )
    all_key_frames = {
        **nominal_b12_trace["key_frames"],
        **nominal_b23_trace["key_frames"],
    }
    for filename, frame in all_key_frames.items():
        imageio.imwrite(output_dir / f"{filename}.png", frame)
    if all_key_frames:
        m316.save_montage(
            list(all_key_frames.values()), output_dir / "m3_2_33_early_chain_montage.png", cols=3
        )

    write_csv(output_dir / "B12_021_narrow_trace.csv", nominal_b12_trace["rows"])
    write_csv(output_dir / "P2_to_P3_dynamic_B23_trace.csv", nominal_b23_trace["rows"])
    save_trace_plot(
        output_dir / "B12_021_narrow_trace.png",
        "M3.2.33 B12 narrow 0.21s ring-under recapture",
        nominal_b12_trace["rows"],
    )
    save_trace_plot(
        output_dir / "P2_to_P3_dynamic_B23_trace.png",
        "M3.2.33 dynamic projected P2 to P3",
        nominal_b23_trace["rows"],
    )
    save_snapshot_cache(output_dir / "narrow_021_projected_P2_arrivals.npz", arrival_states)
    if p3_arrival_states:
        save_snapshot_cache(
            output_dir / "narrow_021_dynamic_P3_arrivals.npz",
            p3_arrival_states,
        )
    copy_existing_assets(output_dir)

    p3_p4, p4_p5, p5_p1 = load_existing_rows()
    b12_pass_count = int(sum(row["passed"] for row in selected_b12_results))
    b23_pass_count = int(sum(row["passed"] for row in b23_validation_results))
    b23_full_angle_count = int(
        sum(row["full_angle_passed"] for row in b23_validation_results)
    )
    five_step_status = [
        {
            "leg": "P1/B12 running -> P2",
            "status": "PASS" if nominal_b12["passed"] else "CHECK",
            "evidence": (
                f"0.21s narrow {b12_pass_count}/{len(selected_b12_results)}; "
                f"ring-under {nominal_b12['first_ring_under_s']:.2f}s; "
                f"penetration {nominal_b12['max_penetration_mm']:.2f}mm"
            ),
        },
        {
            "leg": "P2 dynamic -> P3",
            "status": "PASS" if nominal_b23["passed"] else "CHECK",
            "evidence": (
                f"rotation {nominal_b23['planar_rotation_deg']:.1f}deg; "
                f"index/middle drive {nominal_b23['phase_contact_ratio']['drive_index']:.2f}/"
                f"{nominal_b23['phase_contact_ratio']['drive_middle']:.2f}; "
                f"penetration {nominal_b23['max_penetration_mm']:.2f}mm; "
                f"7deg full-angle {'PASS' if nominal_b23['full_angle_passed'] else 'CHECK'}"
            ),
        },
        {
            "leg": "P3 -> B34 -> P4",
            "status": "PASS" if p3_p4["max_penetration_mm"] <= 4.0 else "CHECK",
            "evidence": (
                f"survival {p3_p4['survival_s']:.2f}s; final {','.join(p3_p4['final_contacts'])}; "
                f"penetration {p3_p4['max_penetration_mm']:.2f}mm"
            ),
        },
        {
            "leg": "P4 -> B45 -> P5",
            "status": "PASS" if p4_p5["max_penetration_mm"] <= 4.0 else "CHECK",
            "evidence": (
                f"survival {p4_p5['survival_s']:.2f}s; final {','.join(p4_p5['final_contacts'])}; "
                f"penetration {p4_p5['max_penetration_mm']:.2f}mm"
            ),
        },
        {
            "leg": "P5 -> P1 cycle entry",
            "status": "CHECK",
            "evidence": (
                f"support-connectable {p5_p1['survival_s']:.2f}s; thumb/index 1.00/1.00; "
                "strict ring recapture remains absent"
            ),
        },
    ]
    report = {
        "stage": "M3.2.33 narrow 0.21s five-step real-action chain",
        "source_t": float(args.source_t),
        "perturbation_scope": {
            "pen_position_mm": {"xy": 0.25, "z": 0.20},
            "pen_linear_velocity_mm_s": {"y": 3.0, "z": 2.0},
            "hand_qpos_rad": 0.003,
            "hand_qvel_rad_s": 0.04,
            "count": len(perturbations),
        },
        "b12_narrow": {
            "selected_candidate": best_b12.__dict__,
            "pass_count": b12_pass_count,
            "total": len(selected_b12_results),
            "results": selected_b12_results,
            "leaderboard_top10": b12_leaderboard[:10],
            "nominal": nominal_b12,
            "arrival_cache_states": len(arrival_states),
        },
        "b23_dynamic": {
            "selected_candidate": best_b23.__dict__,
            "pass_count": b23_pass_count,
            "full_angle_pass_count": b23_full_angle_count,
            "total": len(b23_validation_results),
            "representative_search_results": b23_search_results,
            "validation_results": b23_validation_results,
            "leaderboard_top10": b23_leaderboard[:10],
            "nominal": nominal_b23,
            "arrival_cache_states": len(p3_arrival_states),
        },
        "five_step_status": five_step_status,
        "limits": (
            "The new B12 and B23 segments are evaluated from a narrow dynamic neighborhood centered "
            "on the 0.21s cache state. P3->P4, P4->P5, and P5->P1 are segmented legacy rollouts, not "
            "a single continuous state relay. P4->P5 penetration and strict P5->P1 ring recapture remain open."
        ),
    }
    save_json(output_dir / "summary.json", report)
    write_csv(output_dir / "B12_narrow_validation.csv", selected_b12_results)
    write_csv(output_dir / "B12_search_leaderboard.csv", b12_leaderboard)
    write_csv(output_dir / "B23_search_leaderboard.csv", b23_leaderboard)
    write_visual_summary(output_dir, report)

    assessment = f"""# M3.2.33: 0.21s narrow five-step chain

## New work

- B12 runs from the exact cached 0.21s state plus {len(perturbations)} deterministic narrow perturbations.
- Selected controller `{best_b12.name}` passes {b12_pass_count}/{len(selected_b12_results)} perturbations.
- Successful projected-P2 arrivals are exported with qpos/qvel/target_qpos; no zero-velocity reset is inserted.
- Dynamic B23 starts from those arrivals and uses index-front deflection, index/middle drive, and thumb-yield stages.
- Selected B23 controller `{best_b23.name}` passes {b23_pass_count}/{len(b23_validation_results)} narrow P3-entry arrivals; {b23_full_angle_count}/{len(b23_validation_results)} also exceed 7 degrees.

## Nominal 0.21s evidence

- B12 pass: `{nominal_b12['passed']}`; first ring-under contact `{nominal_b12['first_ring_under_s']}` s; penetration `{nominal_b12['max_penetration_mm']:.3f}` mm.
- Dynamic P2->P3 pass: `{nominal_b23['passed']}`; planar rotation `{nominal_b23['planar_rotation_deg']:.2f}` deg; penetration `{nominal_b23['max_penetration_mm']:.3f}` mm.
- B23 drive index/middle contact ratios: `{nominal_b23['phase_contact_ratio']['drive_index']:.3f}/{nominal_b23['phase_contact_ratio']['drive_middle']:.3f}`.

## Remaining boundaries

This is still a segmented primitive assessment. The dynamic P3 arrival has not yet been relayed into the old P3->P4 rollout. Existing P4->P5 reaches the target contact state but has penetration above 4 mm, and P5->P1 remains thumb/index support-connectable without strict dynamic ring recapture.

## Artifacts

- `visual_summary.html`
- `B12_021_narrow_recapture.gif`
- `P2_to_P3_dynamic_B23.gif`
- `m3_2_33_early_chain_montage.png`
- `narrow_021_projected_P2_arrivals.npz`
- `narrow_021_dynamic_P3_arrivals.npz`
"""
    (output_dir / "ASSESSMENT.zh.md").write_text(assessment, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "b12_candidate": best_b12.name,
                "b12_pass": f"{b12_pass_count}/{len(selected_b12_results)}",
                "b23_candidate": best_b23.name,
                "b23_narrow_validation_pass": f"{b23_pass_count}/{len(b23_validation_results)}",
                "b23_full_angle_pass": f"{b23_full_angle_count}/{len(b23_validation_results)}",
                "nominal_b12_pass": nominal_b12["passed"],
                "nominal_b23_pass": nominal_b23["passed"],
                "nominal_b23_rotation_deg": nominal_b23["planar_rotation_deg"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
