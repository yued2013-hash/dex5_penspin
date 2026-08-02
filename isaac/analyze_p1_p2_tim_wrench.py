from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Audit TIM grasp stability around ring contact loss.")
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--tim-contact-force-n", type=float, default=0.03)
parser.add_argument("--tim-torque-growth-ratio", type=float, default=1.5)
parser.add_argument("--tim-moment-arm-growth-mm", type=float, default=3.0)
parser.add_argument("--pen-speed-growth-m-s", type=float, default=0.01)
parser.add_argument("--pen-angular-speed-growth-rad-s", type=float, default=2.0)
args = parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def longest_run(mask: np.ndarray) -> int:
    running = 0
    best = 0
    for value in mask:
        running = running + 1 if value else 0
        best = max(best, running)
    return best


def first_consecutive_pair(mask: np.ndarray) -> tuple[int, int] | None:
    for step in range(mask.size - 1):
        if mask[step] and mask[step + 1]:
            return step, step + 1
    return None


def gather(values: np.ndarray, indices: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([values[row, step] for row, step in indices])


def main() -> None:
    input_path = args.input.resolve()
    with np.load(input_path, allow_pickle=False) as data:
        eligible = np.asarray(data["accepted_step"], dtype=np.bool_)
        contact_force = np.asarray(
            data["contact_force_thumb_index_middle_ring_pinky"], dtype=np.float64
        )
        metrics = {
            "tim_force_n": np.linalg.norm(data["tim_force_on_pen_w"], axis=-1),
            "tim_torque_nm": np.linalg.norm(
                data["tim_torque_on_pen_about_com_w"], axis=-1
            ),
            "tim_equivalent_moment_arm_mm": (
                np.asarray(data["tim_equivalent_moment_arm_m"], dtype=np.float64)
                * 1000.0
            ),
            "tim_gravity_force_residual_n": np.linalg.norm(
                data["tim_gravity_force_residual_w"], axis=-1
            ),
            "total_gravity_force_residual_n": np.linalg.norm(
                data["total_gravity_force_residual_w"], axis=-1
            ),
            "pen_com_speed_m_s": np.linalg.norm(data["pen_com_velocity_w"], axis=-1),
            "pen_angular_speed_rad_s": np.linalg.norm(
                data["pen_angular_velocity"], axis=-1
            ),
            "ring_force_n": contact_force[:, :, 3],
            "ring_tangential_slip_m_s": np.asarray(
                data["ring_contact_tangential_slip_speed"], dtype=np.float64
            ),
        }
        pen_com_position = np.asarray(data["pen_com_position_w"], dtype=np.float64)
        pen_quaternion = np.asarray(data["state_qpos"][:, :, 23:27], dtype=np.float64)

    com_displacement = np.linalg.norm(
        pen_com_position - pen_com_position[:, :1], axis=-1
    )
    quat_norm = np.linalg.norm(pen_quaternion, axis=-1, keepdims=True).clip(1.0e-8)
    normalized_quaternion = pen_quaternion / quat_norm
    quaternion_dot = np.abs(
        np.sum(normalized_quaternion * normalized_quaternion[:, :1], axis=-1)
    ).clip(0.0, 1.0)
    orientation_drift_deg = np.rad2deg(2.0 * np.arccos(quaternion_dot))
    metrics["pen_com_displacement_mm"] = com_displacement * 1000.0
    metrics["pen_orientation_drift_deg"] = orientation_drift_deg

    max_consecutive = np.asarray([longest_run(mask) for mask in eligible])
    frontier_before: list[tuple[int, int]] = []
    frontier_exit: list[tuple[int, int]] = []
    success_first: list[tuple[int, int]] = []
    success_second: list[tuple[int, int]] = []

    for row, mask in enumerate(eligible):
        if max_consecutive[row] == 1:
            true_steps = np.flatnonzero(mask)
            if true_steps.size and true_steps[0] + 1 < mask.size:
                frontier_before.append((row, int(true_steps[0])))
                frontier_exit.append((row, int(true_steps[0] + 1)))
        elif max_consecutive[row] >= 2:
            pair = first_consecutive_pair(mask)
            if pair is not None:
                success_first.append((row, pair[0]))
                success_second.append((row, pair[1]))

    classifications: Counter[str] = Counter()
    tim_contact_loss = [0, 0, 0]
    tim_instability_signals: Counter[str] = Counter()
    for before, after in zip(frontier_before, frontier_exit, strict=True):
        row, before_step = before
        _, after_step = after
        individual_lost = contact_force[row, after_step, :3] < args.tim_contact_force_n
        for finger in range(3):
            tim_contact_loss[finger] += int(individual_lost[finger])

        torque_before = metrics["tim_torque_nm"][row, before_step]
        torque_after = metrics["tim_torque_nm"][row, after_step]
        torque_growth = torque_after > max(
            torque_before * args.tim_torque_growth_ratio, torque_before + 1.0e-4
        )
        arm_growth = (
            metrics["tim_equivalent_moment_arm_mm"][row, after_step]
            - metrics["tim_equivalent_moment_arm_mm"][row, before_step]
            > args.tim_moment_arm_growth_mm
        )
        speed_growth = (
            metrics["pen_com_speed_m_s"][row, after_step]
            - metrics["pen_com_speed_m_s"][row, before_step]
            > args.pen_speed_growth_m_s
        )
        angular_growth = (
            metrics["pen_angular_speed_rad_s"][row, after_step]
            - metrics["pen_angular_speed_rad_s"][row, before_step]
            > args.pen_angular_speed_growth_rad_s
        )
        signals = {
            "tim_contact_loss": bool(np.any(individual_lost)),
            "tim_torque_growth": bool(torque_growth),
            "tim_moment_arm_growth": bool(arm_growth),
            "pen_com_speed_growth": bool(speed_growth),
            "pen_angular_speed_growth": bool(angular_growth),
        }
        for name, active in signals.items():
            tim_instability_signals[name] += int(active)

        tim_contact_lost = bool(np.any(individual_lost))
        ring_unloaded = metrics["ring_force_n"][row, after_step] < 0.03
        if tim_contact_lost and ring_unloaded:
            classifications["TIM接触与ring同时丢失"] += 1
        elif tim_contact_lost:
            classifications["TIM接触丢失_ring仍在"] += 1
        elif ring_unloaded:
            classifications["TIM接触完整_ring卸载"] += 1
        else:
            classifications["TIM与ring接触均在_其他物理门退出"] += 1

    def group_summary(indices: list[tuple[int, int]]) -> dict[str, dict[str, float]]:
        return {name: stats(gather(values, indices)) for name, values in metrics.items()}

    def transition_delta_summary(
        before: list[tuple[int, int]], after: list[tuple[int, int]]
    ) -> dict[str, dict[str, float]]:
        return {
            name: stats(gather(values, after) - gather(values, before))
            for name, values in metrics.items()
        }

    temporal_summary = []
    for step in range(eligible.shape[1]):
        temporal_summary.append(
            {
                "step": step + 1,
                "time_ms": float((step + 1) * 1000.0 / 120.0),
                "TIM_all_contact_ratio": float(
                    np.mean(np.all(contact_force[:, step, :3] >= args.tim_contact_force_n, axis=-1))
                ),
                "thumb_contact_ratio": float(
                    np.mean(contact_force[:, step, 0] >= args.tim_contact_force_n)
                ),
                "index_contact_ratio": float(
                    np.mean(contact_force[:, step, 1] >= args.tim_contact_force_n)
                ),
                "middle_contact_ratio": float(
                    np.mean(contact_force[:, step, 2] >= args.tim_contact_force_n)
                ),
                "ring_contact_ratio": float(
                    np.mean(contact_force[:, step, 3] >= args.tim_contact_force_n)
                ),
                "TIM_force_n": stats(metrics["tim_force_n"][:, step]),
                "TIM_torque_nm": stats(metrics["tim_torque_nm"][:, step]),
                "TIM_equivalent_moment_arm_mm": stats(
                    metrics["tim_equivalent_moment_arm_mm"][:, step]
                ),
                "pen_COM_displacement_mm": stats(
                    metrics["pen_com_displacement_mm"][:, step]
                ),
                "pen_orientation_drift_deg": stats(
                    metrics["pen_orientation_drift_deg"][:, step]
                ),
                "pen_COM_speed_m_s": stats(metrics["pen_com_speed_m_s"][:, step]),
                "pen_angular_speed_rad_s": stats(
                    metrics["pen_angular_speed_rad_s"][:, step]
                ),
            }
        )

    summary = {
        "input": str(input_path),
        "trajectories": int(eligible.shape[0]),
        "control_steps_per_trajectory": int(eligible.shape[1]),
        "max_consecutive_physical_steps": {
            str(value): int((max_consecutive == value).sum())
            for value in np.unique(max_consecutive)
        },
        "h2_success_trajectories": len(success_first),
        "one_step_frontier_trajectories": len(frontier_before),
        "frontier_before_exit": group_summary(frontier_before),
        "frontier_first_exit": group_summary(frontier_exit),
        "h2_success_first_step": group_summary(success_first),
        "h2_success_second_step": group_summary(success_second),
        "frontier_exit_minus_before_delta": transition_delta_summary(
            frontier_before, frontier_exit
        ),
        "h2_success_second_minus_first_delta": transition_delta_summary(
            success_first, success_second
        ),
        "tim_contact_loss_at_frontier_exit": {
            "thumb": tim_contact_loss[0],
            "index": tim_contact_loss[1],
            "middle": tim_contact_loss[2],
            "any_TIM": int(
                sum(
                    np.any(contact_force[row, step, :3] < args.tim_contact_force_n)
                    for row, step in frontier_exit
                )
            ),
        },
        "tim_instability_signal_counts": dict(tim_instability_signals),
        "frontier_hard_evidence_classification": dict(classifications),
        "temporal_summary_all_trajectories": temporal_summary,
        "classification_thresholds": {
            "TIM_contact_force_n": args.tim_contact_force_n,
            "TIM_torque_growth_ratio": args.tim_torque_growth_ratio,
            "TIM_moment_arm_growth_mm": args.tim_moment_arm_growth_mm,
            "pen_COM_speed_growth_m_s": args.pen_speed_growth_m_s,
            "pen_angular_speed_growth_rad_s": args.pen_angular_speed_growth_rad_s,
        },
        "interpretation": {
            "contact_force": (
                "Fingertip ContactSensor net force is used as a contact proxy; it is not a "
                "pen-only contact-pair impulse decomposition."
            ),
            "wrench": (
                "Pen contact points are estimated on the cylindrical surface, so wrench and "
                "moment arm are comparative audit signals rather than exact PhysX contact wrench."
            ),
            "gravity_residual": (
                "The residual omits inertial terms; compare groups and temporal changes instead "
                "of interpreting its absolute value as static force-balance error."
            ),
            "soft_instability_signals": (
                "Torque, moment-arm, and velocity-growth counts are diagnostic flags only. "
                "They are not used as causal failure labels because successful H2 transitions "
                "can exhibit the same contact-wrench redistribution."
            ),
        },
        "ppo_started": False,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_TIM_WRENCH_AUDIT=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
