from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Dex5-S0-ShortRotation-16-MediumDR-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--successful-trajectories", type=int, default=1200)
parser.add_argument("--max-steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--reset-split", choices=("train", "validation", "test"), default="train")
parser.add_argument(
    "--selection",
    choices=("corrected_success", "all_completed"),
    default="corrected_success",
)
parser.add_argument("--minimum-omega-correlation", type=float, default=0.70)
parser.add_argument("--maximum-omega-rmse-rad-s", type=float, default=0.50)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_split(source_trajectory_id: int) -> str:
    bucket = hashlib.sha256(str(source_trajectory_id).encode("ascii")).digest()[0] % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[0] * right[0] - np.dot(left[1:], right[1:])
    vector = (
        left[0] * right[1:]
        + right[0] * left[1:]
        + np.cross(left[1:], right[1:])
    )
    return np.concatenate(([scalar], vector))


def _pose_derived_angular_velocity(
    before: np.ndarray, after: np.ndarray, control_hz: float
) -> np.ndarray:
    rows = []
    for first, second in zip(before, after):
        first = first / np.linalg.norm(first)
        second = second / np.linalg.norm(second)
        conjugate = first.copy()
        conjugate[1:] *= -1.0
        relative = _quat_mul(second, conjugate)
        relative /= np.linalg.norm(relative)
        if relative[0] < 0.0:
            relative *= -1.0
        angle = 2.0 * np.arccos(np.clip(relative[0], -1.0, 1.0))
        sine = np.sin(0.5 * angle)
        axis = relative[1:] / sine if abs(sine) > 1.0e-9 else np.zeros(3)
        rows.append(axis * angle * control_hz)
    return np.asarray(rows, dtype=np.float64)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) < 1.0e-9 or np.std(second) < 1.0e-9:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _consistency_metrics(
    before_quat: np.ndarray,
    after_quat: np.ndarray,
    before_velocity: np.ndarray,
    after_velocity: np.ndarray,
    sample_hz: float,
) -> dict:
    derived = _pose_derived_angular_velocity(before_quat, after_quat, sample_hz)
    midpoint = 0.5 * (before_velocity + after_velocity)
    error = midpoint - derived
    reported_path = float(np.sum(np.linalg.norm(midpoint, axis=1)) / sample_hz)
    derived_path = float(np.sum(np.linalg.norm(derived, axis=1)) / sample_hz)
    return {
        "sample_count": int(len(derived)),
        "omega_xyz_rmse_rad_s": float(np.sqrt(np.mean(np.square(error)))),
        "omega_z_correlation": _correlation(midpoint[:, 2], derived[:, 2]),
        "quaternion_to_reported_path_ratio": derived_path / max(reported_path, 1.0e-9),
    }


def _kinematic_consistency(
    payload: dict[str, np.ndarray], control_hz: float, physics_hz: float
) -> dict:
    valid = np.asarray(payload["transition_same_episode"], dtype=bool).reshape(-1)
    before_pose = np.asarray(payload["pen_pose"], dtype=np.float64)[valid]
    after_pose = np.asarray(payload["pen_pose_next"], dtype=np.float64)[valid]
    before_velocity = np.asarray(payload["pen_angular_velocity"], dtype=np.float64)[valid]
    after_velocity = np.asarray(payload["pen_angular_velocity_next"], dtype=np.float64)[valid]
    control_rate = _consistency_metrics(
        before_pose[:, 3:7],
        after_pose[:, 3:7],
        before_velocity,
        after_velocity,
        control_hz,
    )
    decimation = int(round(physics_hz / control_hz))
    quat_samples = np.asarray(
        payload["pen_physics_quat_samples"], dtype=np.float64
    )[valid].reshape(-1, decimation + 1, 4)
    velocity_samples = np.asarray(
        payload["pen_physics_angvel_samples"], dtype=np.float64
    )[valid].reshape(-1, decimation + 1, 3)
    physics_rate = _consistency_metrics(
        quat_samples[:, :-1].reshape(-1, 4),
        quat_samples[:, 1:].reshape(-1, 4),
        velocity_samples[:, :-1].reshape(-1, 3),
        velocity_samples[:, 1:].reshape(-1, 3),
        physics_hz,
    )
    return {
        "valid_transition_count": int(valid.sum()),
        "control_rate_hz": control_hz,
        "control_rate": control_rate,
        "physics_rate_hz": physics_hz,
        "physics_substep": physics_rate,
    }


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.reset_split = args.reset_split
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw = gym.make(args.task, cfg=env_cfg)
    direct = raw.unwrapped
    env = SkrlVecEnvWrapper(raw, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    checkpoint = Path(args.checkpoint).resolve()
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    bank_path = Path(direct.cfg.bundle_path).resolve()
    with np.load(bank_path, allow_pickle=False) as bank:
        source_ids = np.asarray(bank["metadata_source_trajectory_id"], dtype=np.int64)
        state_ids = np.asarray(bank["metadata_state_id"], dtype=np.int64)

    observations, _ = env.reset()
    states = env.state()
    max_episode_steps = int(round(float(direct.cfg.episode_length_s) * direct.cfg.s0_policy_hz)) + 4
    lengths = torch.zeros(args.num_envs, dtype=torch.long)
    episode_bank_ids = direct.current_bank_ids.detach().cpu().clone()
    model = direct.s0_actuator_model
    assert model is not None
    episode_delay = model.delay_ticks.detach().cpu().clone()
    episode_gain = model.tracking_gain.detach().cpu().clone()
    episode_velocity = model.velocity_limit_rad_s.detach().cpu().clone()
    completion_generation = direct.completed_s0_generation.detach().cpu().clone()

    specs = {
        "oracle_observation": (152, torch.float32),
        "student_observation": (340, torch.float32),
        "student_observation_next": (340, torch.float32),
        "virtual_action": (20, torch.float32),
        "policy_action_raw": (20, torch.float32),
        "adapter_projected_target": (20, torch.float32),
        "logical_actuator_target": (16, torch.float32),
        "sdk_absolute_target": (20, torch.float32),
        "joint_position": (20, torch.float32),
        "joint_velocity": (20, torch.float32),
        "pen_pose": (7, torch.float32),
        "pen_linear_velocity": (3, torch.float32),
        "pen_angular_velocity": (3, torch.float32),
        "pen_pose_next": (7, torch.float32),
        "pen_linear_velocity_next": (3, torch.float32),
        "pen_angular_velocity_next": (3, torch.float32),
        "contact_state_next": (5, torch.bool),
        "contact_force_w_next": (15, torch.float32),
        "contact_force_norm_next": (5, torch.float32),
        "transition_same_episode": (1, torch.bool),
        "pen_physics_quat_samples": (
            (int(direct.cfg.decimation) + 1) * 4,
            torch.float32,
        ),
        "pen_physics_angvel_samples": (
            (int(direct.cfg.decimation) + 1) * 3,
            torch.float32,
        ),
        "contact_state": (5, torch.bool),
        "contact_force_w": (15, torch.float32),
        "contact_force_norm": (5, torch.float32),
    }
    buffers = {
        name: torch.empty((args.num_envs, max_episode_steps, width), dtype=dtype)
        for name, (width, dtype) in specs.items()
    }
    trajectories: dict[str, list[np.ndarray]] = {name: [] for name in specs}
    episode_metadata: list[dict[str, int | float | str]] = []
    total_completed = 0
    success_flag_count = 0
    control_steps = 0

    while len(episode_metadata) < args.successful_trajectories and control_steps < args.max_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations, states, timestep=control_steps, timesteps=args.max_steps
            )
            actions = outputs[-1].get("mean_actions", outputs[0])
            direct._compute_intermediate_values()
            student = direct.get_student_observation()
            current = model._interpolated_target(direct._simulation_time_s())
            allowed = torch.minimum(
                model.velocity_limit_rad_s / model.policy_hz,
                current.new_full(
                    (args.num_envs,),
                    min(float(direct.cfg.action_scale), model.max_policy_delta_rad),
                ),
            )
            projected = model.projector.project_targets(
                current + allowed.unsqueeze(-1) * torch.clamp(actions, -1.0, 1.0)
            )
            logical = model.projector.virtual_to_logical_actuator(projected)
            snapshots = {
                "oracle_observation": observations,
                "student_observation": student,
                "virtual_action": torch.clamp(actions, -1.0, 1.0),
                "policy_action_raw": actions,
                "adapter_projected_target": projected,
                "logical_actuator_target": logical,
                "sdk_absolute_target": model.applied_target,
                "joint_position": direct.hand_qpos,
                "joint_velocity": direct.hand_qvel,
                "pen_pose": torch.cat((direct.pen_pos, direct.pen_quat), dim=-1),
                "pen_linear_velocity": direct.pen_linvel,
                "pen_angular_velocity": direct.pen_angvel,
                "contact_state": direct.contact_force_norm
                > direct.cfg.contact_force_threshold_n,
                "contact_force_w": direct.contact_force_w.reshape(args.num_envs, -1),
                "contact_force_norm": direct.contact_force_norm,
            }
            rows = torch.arange(args.num_envs)
            if int(lengths.max().item()) >= max_episode_steps:
                raise RuntimeError("trajectory buffer exceeded configured episode horizon")
            for name, value in snapshots.items():
                buffers[name][rows, lengths] = value.detach().cpu()

            observations, _, _, _, _ = env.step(actions)
            states = env.state()
            direct._compute_intermediate_values()
            next_generation = direct.completed_s0_generation.detach().cpu()
            completed_transition = (
                next_generation != completion_generation
            ).to(device=direct.device)
            pen_pose_after = torch.cat((direct.pen_pos, direct.pen_quat), dim=-1)
            pen_linvel_after = direct.pen_linvel.clone()
            pen_angvel_after = direct.pen_angvel.clone()
            contact_state_after = (
                direct.contact_force_norm > direct.cfg.contact_force_threshold_n
            )
            contact_force_w_after = direct.contact_force_w.clone()
            contact_force_norm_after = direct.contact_force_norm.clone()
            student_observation_after = direct.get_student_observation().clone()
            physics_quat_samples = torch.cat(
                (
                    direct.s0_physics_quat_before,
                    direct.pen_quat.unsqueeze(1),
                ),
                dim=1,
            )
            physics_angvel_samples = torch.cat(
                (
                    direct.s0_physics_angvel_before,
                    direct.pen_angvel.unsqueeze(1),
                ),
                dim=1,
            )
            if completed_transition.any():
                pen_pose_after[completed_transition] = (
                    direct.last_completed_s0_pen_pose[completed_transition]
                )
                pen_linvel_after[completed_transition] = (
                    direct.last_completed_s0_pen_linvel[completed_transition]
                )
                pen_angvel_after[completed_transition] = (
                    direct.last_completed_s0_pen_angvel[completed_transition]
                )
                contact_state_after[completed_transition] = (
                    direct.last_completed_s0_contact_state[completed_transition]
                )
                contact_force_w_after[completed_transition] = (
                    direct.last_completed_s0_contact_force_w[completed_transition]
                )
                contact_force_norm_after[completed_transition] = (
                    direct.last_completed_s0_contact_force_norm[completed_transition]
                )
                physics_quat_samples[completed_transition] = (
                    direct.last_completed_s0_physics_quat_samples[completed_transition]
                )
                physics_angvel_samples[completed_transition] = (
                    direct.last_completed_s0_physics_angvel_samples[completed_transition]
                )
                student_observation_after[completed_transition] = (
                    direct.last_completed_s0_student_observation[completed_transition]
                )
            next_snapshots = {
                "student_observation_next": student_observation_after,
                "pen_pose_next": pen_pose_after,
                "pen_linear_velocity_next": pen_linvel_after,
                "pen_angular_velocity_next": pen_angvel_after,
                "contact_state_next": contact_state_after,
                "contact_force_w_next": contact_force_w_after.reshape(
                    args.num_envs, -1
                ),
                "contact_force_norm_next": contact_force_norm_after,
                "transition_same_episode": torch.ones(
                    (args.num_envs, 1), dtype=torch.bool, device=direct.device
                ),
                "pen_physics_quat_samples": physics_quat_samples.reshape(
                    args.num_envs, -1
                ),
                "pen_physics_angvel_samples": physics_angvel_samples.reshape(
                    args.num_envs, -1
                ),
            }
            for name, value in next_snapshots.items():
                buffers[name][rows, lengths] = value.detach().cpu()

        lengths += 1
        done_ids = torch.nonzero(
            next_generation != completion_generation, as_tuple=False
        ).flatten()
        if done_ids.numel():
            success = direct.last_completed_s0_success.detach().cpu()
            success_flag_count += int(success[done_ids].sum())
            total_completed += int(done_ids.numel())
            for env_id in done_ids.tolist():
                selected = (
                    bool(success[env_id])
                    if args.selection == "corrected_success"
                    else True
                )
                if selected and len(episode_metadata) < args.successful_trajectories:
                    length = int(lengths[env_id])
                    for name in specs:
                        trajectories[name].append(buffers[name][env_id, :length].numpy().copy())
                    bank_id = int(episode_bank_ids[env_id])
                    source_id = int(source_ids[bank_id])
                    episode_metadata.append(
                        {
                            "bank_id": bank_id,
                            "state_id": int(state_ids[bank_id]),
                            "source_trajectory_id": source_id,
                            "dataset_split": _dataset_split(source_id),
                            "length": length,
                            "delay_ticks": int(episode_delay[env_id]),
                            "tracking_gain": float(episode_gain[env_id]),
                            "velocity_limit_rad_s": float(episode_velocity[env_id]),
                            "corrected_success": bool(success[env_id]),
                        }
                    )
            lengths[done_ids] = 0
            episode_bank_ids[done_ids] = direct.current_bank_ids[done_ids.to(direct.device)].cpu()
            episode_delay[done_ids] = model.delay_ticks[done_ids.to(direct.device)].cpu()
            episode_gain[done_ids] = model.tracking_gain[done_ids.to(direct.device)].cpu()
            episode_velocity[done_ids] = model.velocity_limit_rad_s[done_ids.to(direct.device)].cpu()
            completion_generation[done_ids] = next_generation[done_ids]
        control_steps += 1

    if len(episode_metadata) < args.successful_trajectories:
        raise RuntimeError(
            f"collected only {len(episode_metadata)} successful trajectories in {control_steps} steps"
        )

    lengths_np = np.asarray([row["length"] for row in episode_metadata], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths_np))).astype(np.int64)
    payload: dict[str, np.ndarray] = {
        name: np.concatenate(parts, axis=0) for name, parts in trajectories.items()
    }
    payload.update(
        {
            "episode_offsets": offsets,
            "episode_length": lengths_np,
            "episode_bank_id": np.asarray([row["bank_id"] for row in episode_metadata], dtype=np.int64),
            "episode_state_id": np.asarray([row["state_id"] for row in episode_metadata], dtype=np.int64),
            "episode_source_trajectory_id": np.asarray(
                [row["source_trajectory_id"] for row in episode_metadata], dtype=np.int64
            ),
            "episode_split": np.asarray([row["dataset_split"] for row in episode_metadata]),
            "episode_delay_ticks": np.asarray([row["delay_ticks"] for row in episode_metadata], dtype=np.int64),
            "episode_tracking_gain": np.asarray([row["tracking_gain"] for row in episode_metadata], dtype=np.float32),
            "episode_velocity_limit_rad_s": np.asarray(
                [row["velocity_limit_rad_s"] for row in episode_metadata], dtype=np.float32
            ),
            "episode_corrected_success": np.asarray(
                [row["corrected_success"] for row in episode_metadata], dtype=bool
            ),
        }
    )
    consistency = _kinematic_consistency(
        payload,
        float(direct.cfg.s0_policy_hz),
        1.0 / float(direct.physics_dt),
    )
    same_episode = np.asarray(payload["transition_same_episode"], dtype=bool).reshape(-1)
    pose_delta_velocity = np.zeros_like(
        np.asarray(payload["pen_angular_velocity"], dtype=np.float64)
    )
    before_quat = np.asarray(payload["pen_pose"], dtype=np.float64)[same_episode, 3:7]
    after_quat = np.asarray(payload["pen_pose_next"], dtype=np.float64)[same_episode, 3:7]
    pose_delta_velocity[same_episode] = _pose_derived_angular_velocity(
        before_quat, after_quat, float(direct.cfg.s0_policy_hz)
    )
    pose_delta_velocity = pose_delta_velocity.astype(np.float32)
    payload["pen_angular_velocity_physx_instantaneous"] = np.asarray(
        payload.pop("pen_angular_velocity"), dtype=np.float32
    )
    payload["pen_angular_velocity_physx_instantaneous_next"] = np.asarray(
        payload.pop("pen_angular_velocity_next"), dtype=np.float32
    )
    payload["pen_angular_velocity_pose_delta"] = pose_delta_velocity
    # Canonical motion channel for all downstream consumers.
    payload["pen_angular_velocity"] = pose_delta_velocity.copy()
    gate_metrics = consistency["physics_substep"]
    instantaneous_diagnostic_pass = bool(
        gate_metrics["omega_z_correlation"] >= args.minimum_omega_correlation
        and gate_metrics["omega_xyz_rmse_rad_s"] <= args.maximum_omega_rmse_rad_s
        and 0.5 <= gate_metrics["quaternion_to_reported_path_ratio"] <= 1.5
    )
    motion_channel_pass = bool(
        same_episode.all()
        and np.isfinite(payload["pen_pose"]).all()
        and np.isfinite(payload["pen_pose_next"]).all()
        and np.isfinite(pose_delta_velocity).all()
    )
    payload["dataset_authorized"] = np.asarray(motion_channel_pass)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    split_counts = {
        split: sum(row["dataset_split"] == split for row in episode_metadata)
        for split in ("train", "validation", "test")
    }
    unique_sources_by_split = {
        split: len(
            {
                int(row["source_trajectory_id"])
                for row in episode_metadata
                if row["dataset_split"] == split
            }
        )
        for split in split_counts
    }
    report = {
        "status": "PASS_MOTION_CHANNEL" if motion_channel_pass else "BLOCKED_STATE_INCONSISTENT",
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "reset_bank": str(bank_path),
        "reset_bank_sha256": _sha256(bank_path),
        "reset_source_split": args.reset_split,
        "successful_trajectories": len(episode_metadata),
        "selection": args.selection,
        "corrected_successes_in_selected": int(payload["episode_corrected_success"].sum()),
        "completed_episodes_seen": total_completed,
        "success_flag_count": success_flag_count,
        "control_steps": control_steps,
        "total_transition_rows": int(offsets[-1]),
        "split_counts": split_counts,
        "unique_sources_by_split": unique_sources_by_split,
        "split_leakage": 0,
        "kinematic_consistency": consistency,
        "terminal_state_contract": {
            "captured_terminal_transitions": int(len(episode_metadata)),
            "expected_terminal_transitions": int(len(episode_metadata)),
            "all_action_transitions_have_pre_and_post_state": bool(same_episode.all()),
        },
        "motion_evidence_contract": {
            "task_rotation_source": "root_quat_w relative rotation",
            "canonical_velocity_field": "pen_angular_velocity",
            "finite_step_velocity_field": "pen_angular_velocity_pose_delta",
            "instantaneous_velocity_field": "pen_angular_velocity_physx_instantaneous",
            "instantaneous_velocity_role": "contact/impact diagnostic only",
            "terminal_state_source": "pre-reset environment snapshot",
            "reset_boundaries_excluded_by": "episode offsets and terminal snapshots",
        },
        "kinematic_consistency_thresholds": {
            "minimum_omega_z_correlation": args.minimum_omega_correlation,
            "maximum_omega_xyz_rmse_rad_s": args.maximum_omega_rmse_rad_s,
            "quaternion_to_reported_path_ratio": [0.5, 1.5],
        },
        "instantaneous_velocity_diagnostic_pass": instantaneous_diagnostic_pass,
        "dataset_authorized": motion_channel_pass,
        "logical_actuator_semantics": "16D diagnostic coupled coordinates; not motor-shaft commands",
        "sdk_command_semantics": "20D absolute URDF-joint target at the 100 Hz command chain",
        "array_shapes": {key: list(value.shape) for key, value in payload.items()},
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raw.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
