from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / (
    "Dex5_Sim2Real_S0/reset_distribution/candidate/"
    "Dex5_S0_HW_P3_projected_candidates.npz"
)
DEFAULT_OUTPUT = ROOT / (
    "Dex5_Sim2Real_S0/reset_distribution/settled_candidate/"
    "Dex5_S0_HW_P3_settled_candidates.npz"
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument(
    "--task", default="Dex5-S0-HardwareCompatibleReset-Direct-v0"
)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--settle-steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--min-contact-force", type=float, default=0.03)
parser.add_argument("--max-penetration-mm", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from dex5_penspin.isaac.canonical_reset_schema import (  # noqa: E402
    save_npz_atomic,
    sha256_file,
    stable_text_id,
    write_json_atomic,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[:, 1:4]
    v = np.broadcast_to(vector, (quat.shape[0], 3))
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (quat[:, :1] * uv + uuv)


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    with np.load(input_path, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    count = int(source["robot_qpos"].shape[0])
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.bundle_path = str(input_path)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped

    settled_qpos = np.empty((count, 20), dtype=np.float32)
    settled_qvel = np.empty((count, 20), dtype=np.float32)
    settled_target = np.empty((count, 20), dtype=np.float32)
    settled_pose = np.empty((count, 7), dtype=np.float32)
    settled_linvel = np.empty((count, 3), dtype=np.float32)
    settled_angvel = np.empty((count, 3), dtype=np.float32)
    contact_state = np.empty((count, 5), dtype=np.bool_)
    contact_force = np.empty((count, 5), dtype=np.float32)
    contact_points_world = np.empty((count, 5, 3), dtype=np.float32)
    contact_points_local = np.empty((count, 5, 3), dtype=np.float32)
    contact_ratio = np.empty((count, 5), dtype=np.float32)
    max_penetration = np.empty(count, dtype=np.float32)
    max_drift = np.empty(count, dtype=np.float32)
    max_drop = np.empty(count, dtype=np.float32)
    finite_all = np.empty(count, dtype=np.bool_)
    wrench_force = np.empty((count, 3), dtype=np.float32)
    wrench_torque = np.empty((count, 3), dtype=np.float32)

    for start in range(0, count, direct.num_envs):
        end = min(start + direct.num_envs, count)
        batch_count = end - start
        rows = torch.arange(start, end, device=direct.device, dtype=torch.long)
        if batch_count < direct.num_envs:
            rows = torch.cat((rows, rows[:1].repeat(direct.num_envs - batch_count)))
        direct.forced_bank_ids = rows
        env.reset()
        initial_pen = direct.init_pen_pos[:batch_count].clone()
        contact_steps = torch.zeros((batch_count, 5), device=direct.device)
        batch_finite = torch.ones(
            batch_count, dtype=torch.bool, device=direct.device
        )
        batch_max_drift = torch.zeros(batch_count, device=direct.device)
        batch_max_drop = torch.zeros(batch_count, device=direct.device)

        for _ in range(args.settle_steps):
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(direct.cfg.sim.dt)
            direct._compute_intermediate_values()
            contacts = direct.contact_force_norm[:batch_count] > args.min_contact_force
            contact_steps += contacts.float()
            drift = torch.linalg.norm(
                direct.pen_pos[:batch_count] - initial_pen, dim=-1
            )
            drop = torch.clamp(
                initial_pen[:, 2] - direct.pen_pos[:batch_count, 2], min=0.0
            )
            batch_max_drift = torch.maximum(batch_max_drift, drift)
            batch_max_drop = torch.maximum(batch_max_drop, drop)
            batch_finite &= (
                torch.isfinite(direct.hand_qpos[:batch_count]).all(dim=-1)
                & torch.isfinite(direct.hand_qvel[:batch_count]).all(dim=-1)
                & torch.isfinite(direct.pen_pos[:batch_count]).all(dim=-1)
                & torch.isfinite(direct.pen_quat[:batch_count]).all(dim=-1)
                & torch.isfinite(direct.pen_linvel[:batch_count]).all(dim=-1)
                & torch.isfinite(direct.pen_angvel[:batch_count]).all(dim=-1)
            )

        sl = slice(start, end)
        settled_qpos[sl] = direct.hand_qpos[:batch_count].detach().cpu().numpy()
        settled_qvel[sl] = direct.hand_qvel[:batch_count].detach().cpu().numpy()
        settled_target[sl] = direct.processed_targets[:batch_count].detach().cpu().numpy()
        settled_pose[sl, :3] = direct.pen_pos[:batch_count].detach().cpu().numpy()
        settled_pose[sl, 3:7] = direct.pen_quat[:batch_count].detach().cpu().numpy()
        settled_linvel[sl] = direct.pen_linvel[:batch_count].detach().cpu().numpy()
        settled_angvel[sl] = direct.pen_angvel[:batch_count].detach().cpu().numpy()
        contact_state[sl] = (
            direct.contact_force_norm[:batch_count] > args.min_contact_force
        ).detach().cpu().numpy()
        contact_force[sl] = direct.contact_force_norm[:batch_count].detach().cpu().numpy()
        contact_points_world[sl] = (
            direct.contact_point_on_pen_w[:batch_count].detach().cpu().numpy()
        )
        origins = direct.scene.env_origins[:batch_count].unsqueeze(1)
        contact_points_local[sl] = (
            direct.contact_point_on_pen_w[:batch_count] - origins
        ).detach().cpu().numpy()
        contact_ratio[sl] = (contact_steps / float(args.settle_steps)).cpu().numpy()
        max_penetration[sl] = (
            direct.episode_max_penetration[:batch_count].detach().cpu().numpy()
        )
        max_drift[sl] = batch_max_drift.detach().cpu().numpy()
        max_drop[sl] = batch_max_drop.detach().cpu().numpy()
        finite_all[sl] = batch_finite.detach().cpu().numpy()
        forces = direct.contact_force_w[:batch_count]
        pen_world = direct.pen.data.root_pos_w[:batch_count].unsqueeze(1)
        arms = direct.contact_point_on_pen_w[:batch_count] - pen_world
        wrench_force[sl] = forces.sum(dim=1).detach().cpu().numpy()
        wrench_torque[sl] = torch.cross(arms, forces, dim=-1).sum(dim=1).detach().cpu().numpy()
        print(
            f"S0_SETTLE_PROGRESS={end}/{count}",
            flush=True,
        )

    payload = {key: np.asarray(value).copy() for key, value in source.items()}
    projected_ids = np.asarray(payload["metadata_s0_projected_state_id"], dtype=np.int64)
    settled_ids = np.asarray(
        [stable_text_id(f"s0-settled:{int(value)}:{args.settle_steps}") for value in projected_ids],
        dtype=np.int64,
    )
    payload["metadata_parent_state_id"] = projected_ids
    payload["metadata_state_id"] = settled_ids
    payload["metadata_s0_settled_state_id"] = settled_ids
    payload["robot_qpos"] = settled_qpos
    payload["robot_qvel"] = settled_qvel
    payload["robot_q_target"] = settled_target
    payload["robot_previous_action"] = np.zeros((count, 20), dtype=np.float32)
    payload["robot_action_history_buffer"] = np.zeros((count, 1, 20), dtype=np.float32)
    payload["robot_action_history_valid_length"] = np.ones(count, dtype=np.int64)
    payload["robot_control_delay_buffer"] = np.zeros((count, 1, 20), dtype=np.float32)
    payload["robot_control_delay_valid_length"] = np.zeros(count, dtype=np.int64)
    payload["robot_control_delay_enabled"] = np.ones(count, dtype=np.bool_)
    payload["robot_continuation_q_target"] = np.repeat(
        settled_target[:, None, :], 8, axis=1
    )
    payload["robot_continuation_action"] = np.zeros((count, 8, 20), dtype=np.float32)
    payload["robot_continuation_valid_length"] = np.zeros(count, dtype=np.int64)
    payload["object_pose"] = settled_pose
    payload["object_linear_velocity"] = settled_linvel
    payload["object_angular_velocity"] = settled_angvel
    payload["object_com"] = (
        settled_pose[:, :3]
        + _quat_rotate_wxyz(
            settled_pose[:, 3:7], np.asarray((0.00935, 0.0, 0.0), dtype=np.float32)
        )
    ).astype(np.float32)
    payload["contact_state"] = contact_state
    payload["contact_state_available"] = np.ones(count, dtype=np.bool_)
    payload["contact_force"] = contact_force
    payload["contact_force_available"] = np.ones(count, dtype=np.bool_)
    payload["contact_fingertip_positions"] = contact_points_world
    payload["contact_positions_available"] = np.ones(count, dtype=np.bool_)
    payload["contact_positions_frame"] = np.full(count, "world", dtype="U5")
    payload["labels_source_max_penetration_m"] = max_penetration
    payload["labels_source_penetration_available"] = np.ones(count, dtype=np.bool_)
    payload["s0_settle_steps"] = np.full(count, args.settle_steps, dtype=np.int64)
    payload["s0_settle_duration_ms"] = np.full(
        count, 1000.0 * args.settle_steps * float(cfg.sim.dt), dtype=np.float32
    )
    payload["s0_settle_contact_ratio"] = contact_ratio
    payload["s0_contact_change"] = contact_state != np.asarray(source["contact_state"])
    payload["s0_contact_points_env_local"] = contact_points_local
    payload["s0_contact_wrench_force_w"] = wrench_force
    payload["s0_contact_wrench_torque_w"] = wrench_torque
    payload["s0_wrench_margin_n"] = np.min(contact_force[:, :3], axis=1).astype(np.float32)
    payload["s0_settle_max_drift_m"] = max_drift
    payload["s0_settle_max_drop_m"] = max_drop
    precheck = (
        finite_all
        & (max_penetration < args.max_penetration_mm / 1000.0)
        & contact_state[:, :3].all(axis=1)
    )
    payload["s0_settle_precheck_pass"] = precheck
    payload["labels_source"] = np.full(
        count, "S0 P3 projected and PhysX settled", dtype="U36"
    )
    payload["labels_validation_status"] = np.full(count, "candidate", dtype="U10")
    for key in (
        "labels_r0_pass",
        "labels_r1_pass",
        "labels_r2_pass",
        "labels_r3a_pass",
        "labels_r3b_pass",
    ):
        payload[key] = np.zeros(count, dtype=np.bool_)
    payload["labels_rejection_code"] = np.where(
        precheck, "not_validated", "settle_precheck"
    ).astype("U32")

    save_npz_atomic(output_path, payload)
    report = {
        "status": "settled_candidate_not_r0_r3_validated",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "rows": count,
        "settle_steps": args.settle_steps,
        "settle_duration_ms": 1000.0 * args.settle_steps * float(cfg.sim.dt),
        "finite_rate": float(finite_all.mean()),
        "tim_final_contact_rate": float(contact_state[:, :3].all(axis=1).mean()),
        "tim_each_contact_rate": contact_state[:, :3].mean(axis=0).tolist(),
        "at_least_two_tim_rate": float((contact_state[:, :3].sum(axis=1) >= 2).mean()),
        "max_penetration_mm": float(max_penetration.max() * 1000.0),
        "precheck_pass": int(precheck.sum()),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(output_path.with_suffix(".json"), report)
    print("S0_SETTLE_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
