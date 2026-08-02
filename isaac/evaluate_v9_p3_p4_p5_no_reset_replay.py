from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Replay validated P3-to-P4 targets and switch to learned P4-to-P5 "
        "policies in one PhysX episode without rewriting hand or pen state."
    )
)
parser.add_argument("--source", required=True)
parser.add_argument(
    "--task",
    default="Dex5-V9P4-P5-CurrentP4MiddleClear-Direct-v0",
)
parser.add_argument("--middle-checkpoint", required=True)
parser.add_argument("--capture-checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--trajectory-output", default=None)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument(
    "--source-length",
    type=int,
    default=None,
    help="Restrict replay to one equal-length source cohort.",
)
parser.add_argument("--max-downstream-steps", type=int, default=360)
parser.add_argument("--blend-steps", type=int, default=6)
parser.add_argument("--capture-hold-steps", type=int, default=24)
parser.add_argument("--seed", type=int, default=2402)
parser.add_argument("--upstream-success-rate", type=float, default=0.83447265625)
parser.add_argument("--local-p4-p5-rate", type=float, default=0.9356689453125)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


ROOT = Path(__file__).resolve().parents[1]
TASK = args.task
P5_TEMPLATE = (
    ROOT
    / "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "capture_collision_free_cache_seed1503.npz"
)


class Policy(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        super().__init__()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state = checkpoint["policy"]
        input_dim = int(state["net_container.0.weight"].shape[1])
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 20),
        )
        layer_pairs = (
            (0, "net_container.0"),
            (2, "net_container.2"),
            (4, "net_container.4"),
            (6, "policy_layer"),
        )
        for layer, key in layer_pairs:
            self.net[layer].weight.data.copy_(state[f"{key}.weight"])
            self.net[layer].bias.data.copy_(state[f"{key}.bias"])
        preprocessor = checkpoint["observation_preprocessor"]
        self.register_buffer("mean", preprocessor["running_mean"].float())
        self.register_buffer(
            "variance", preprocessor["running_variance"].float()
        )
        self.to(device)
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (observation - self.mean)
            / (torch.sqrt(self.variance) + 1.0e-8),
            -5.0,
            5.0,
        )
        return self.net(normalized)


def nearest_rows(
    current_hand: torch.Tensor, source_hand: torch.Tensor
) -> torch.Tensor:
    rows = []
    for chunk in current_hand.split(64):
        distance = torch.mean(
            (chunk[:, None, :] - source_hand[None, :, :]).square(), dim=-1
        )
        rows.append(torch.argmin(distance, dim=1))
    return torch.cat(rows)


def resolve(path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def main() -> None:
    source_path = resolve(args.source)
    with np.load(source_path) as source:
        lengths_all = np.asarray(source["lengths"], dtype=np.int64)
        initial_qpos_all = np.asarray(source["initial_qpos"], dtype=np.float32)
        initial_qvel_all = np.asarray(source["initial_qvel"], dtype=np.float32)
        processed_target_all = np.asarray(
            source["processed_target"], dtype=np.float32
        )
        upstream_penetration_all = np.asarray(
            source["max_penetration_m"], dtype=np.float32
        )
        control_dt_s = float(source["control_dt_s"])

    rng = np.random.default_rng(args.seed)
    eligible_rows = np.arange(len(lengths_all), dtype=np.int64)
    if args.source_length is not None:
        eligible_rows = eligible_rows[lengths_all == args.source_length]
        if len(eligible_rows) == 0:
            raise ValueError(
                f"No source trajectories have length {args.source_length}"
            )
    requested = min(args.episodes, len(eligible_rows))
    selected_rows = rng.choice(
        eligible_rows, size=requested, replace=False
    )

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = max(
        4.0, args.max_downstream_steps * cfg.sim.dt + 0.5
    )
    cfg.reset_joint_pos_noise = 0.0
    cfg.reset_joint_vel_noise = 0.0
    cfg.reset_pen_pos_noise = 0.0
    cfg.reset_pen_vel_noise = 0.0
    env = gym.make(TASK, cfg=cfg)
    direct = env.unwrapped
    penetration_limit_m = float(direct.cfg.p5_max_penetration_m)
    middle_policy = Policy(args.middle_checkpoint, direct.device)
    capture_policy = Policy(args.capture_checkpoint, direct.device)

    with np.load(P5_TEMPLATE) as template:
        p5_hand_np = np.asarray(
            template["target_qpos"], dtype=np.float32
        )[:, :20].mean(axis=0)
    p5_hand = torch.as_tensor(p5_hand_np, device=direct.device)

    completed = 0
    downstream_successes = 0
    middle_successes = 0
    dropped = 0
    penetrated = 0
    max_penetration_mm = 0.0
    handoff_action_jump = []
    handoff_target_jump = []
    successful_rows: list[int] = []
    successful_actions: list[np.ndarray] = []
    successful_pen_qpos: list[np.ndarray] = []
    successful_lengths: list[int] = []

    for batch_start in range(0, requested, args.num_envs):
        batch_rows = selected_rows[
            batch_start : min(batch_start + args.num_envs, requested)
        ]
        count = len(batch_rows)
        env_ids = torch.arange(count, device=direct.device)
        print(
            f"batch={batch_start // args.num_envs + 1} reset_start count={count}",
            flush=True,
        )
        env.reset(seed=args.seed + batch_start)
        print(
            f"batch={batch_start // args.num_envs + 1} reset_done",
            flush=True,
        )

        initial_qpos = torch.as_tensor(
            initial_qpos_all[batch_rows], device=direct.device
        )
        initial_qvel = torch.as_tensor(
            initial_qvel_all[batch_rows], device=direct.device
        )
        direct.hand.write_joint_state_to_sim(
            initial_qpos[:, :20],
            initial_qvel[:, :20],
            joint_ids=direct.dex5_joint_ids,
            env_ids=env_ids,
        )
        pen_pose = initial_qpos[:, 20:27].clone()
        pen_pose[:, :3] += direct.scene.env_origins[env_ids]
        direct.pen.write_root_pose_to_sim(pen_pose, env_ids)
        direct.pen.write_root_velocity_to_sim(
            initial_qvel[:, 20:26], env_ids
        )
        first_target = torch.as_tensor(
            processed_target_all[batch_rows, 0], device=direct.device
        )
        direct.hand.set_joint_position_target(
            first_target,
            joint_ids=direct.dex5_joint_ids,
            env_ids=env_ids,
        )
        direct.episode_max_penetration[env_ids] = 0.0
        direct.episode_length_buf[env_ids] = 0
        direct.sim.forward()
        direct.scene.update(dt=0.0)

        upstream_penetration = np.asarray(
            [
                upstream_penetration_all[row, : lengths_all[row]].max()
                for row in batch_rows
            ],
            dtype=np.float32,
        )
        global_initial_pen_pos = initial_qpos[:, 20:23].clone()
        active = torch.ones(count, dtype=torch.bool, device=direct.device)
        max_length = int(lengths_all[batch_rows].max())
        final_replay_target = first_target.clone()

        for replay_step in range(max_length):
            valid_np = replay_step < lengths_all[batch_rows]
            if not np.any(valid_np):
                break
            valid = torch.as_tensor(valid_np, device=direct.device)
            target = torch.as_tensor(
                processed_target_all[batch_rows, replay_step],
                device=direct.device,
            )
            final_replay_target[valid] = target[valid]
            direct.hand.set_joint_position_target(
                target[valid],
                joint_ids=direct.dex5_joint_ids,
                env_ids=env_ids[valid],
            )
            direct.sim.step(render=False)
            direct.scene.update(dt=cfg.sim.dt)
            if (replay_step + 1) % 50 == 0:
                print(
                    f"batch={batch_start // args.num_envs + 1} "
                    f"replay_step={replay_step + 1}/{max_length}",
                    flush=True,
                )

        direct._compute_intermediate_values()
        print(
            "handoff_probe "
            f"finite={bool(torch.isfinite(direct.hand_qpos[env_ids]).all() and torch.isfinite(direct.pen_pos[env_ids]).all())} "
            f"max_penetration_mm={float(direct.episode_max_penetration[env_ids].max().item() * 1000.0):.6f} "
            f"max_force_n={float(direct.contact_force_norm[env_ids].max().item()):.6f} "
            f"max_speed_rad_s={float(torch.linalg.norm(direct.pen_angvel[env_ids], dim=-1).max().item()):.6f}",
            flush=True,
        )
        source_rows = nearest_rows(
            direct.hand_qpos[env_ids], direct.source_qpos_bank[:, :20]
        )
        direct.current_bank_ids[env_ids] = source_rows
        direct.source_targets[env_ids] = final_replay_target
        direct.target_hand_qpos[env_ids] = direct.target_hand_bank[source_rows]
        direct.policy_targets[env_ids] = final_replay_target
        direct.processed_targets[env_ids] = final_replay_target
        direct.actions[env_ids] = 0.0
        direct.previous_actions[env_ids] = 0.0
        direct.action_delay_buffer[env_ids] = 0.0
        direct.p5_stage_ids[env_ids] = 1
        direct.stage_hold_count[env_ids] = 0
        direct.p5_hold_count[env_ids] = 0
        direct.episode_p5_success[env_ids] = False
        direct.episode_length_buf[env_ids] = 0
        direct.init_pen_pos[env_ids] = direct.pen_pos[env_ids]
        direct.init_pen_axis[env_ids] = direct.pen_axis[env_ids]
        direct.episode_max_penetration[env_ids] = torch.as_tensor(
            upstream_penetration, device=direct.device
        )
        direct.hand.set_joint_position_target(
            final_replay_target,
            joint_ids=direct.dex5_joint_ids,
            env_ids=env_ids,
        )
        observations = direct._get_observations()["policy"]

        capture_phase = torch.zeros(
            count, dtype=torch.bool, device=direct.device
        )
        success = torch.zeros_like(capture_phase)
        failed = torch.zeros_like(capture_phase)
        blend_age = torch.full(
            (count,), args.blend_steps, dtype=torch.long, device=direct.device
        )
        previous_action = torch.zeros(
            (count, 20), device=direct.device
        )
        previous_target = final_replay_target.clone()
        action_history: list[np.ndarray] = []
        pen_history: list[np.ndarray] = []

        for downstream_step in range(args.max_downstream_steps):
            if downstream_step == 0:
                print("downstream_first_step_start", flush=True)
            with torch.inference_mode():
                middle_actions = middle_policy(observations[:count])
                capture_actions = capture_policy(observations[:count])
                actions = torch.where(
                    capture_phase.unsqueeze(-1),
                    capture_actions,
                    middle_actions,
                )
                blending = capture_phase & (blend_age < args.blend_steps)
                if torch.any(blending):
                    alpha = (
                        (blend_age[blending] + 1).float()
                        / max(args.blend_steps, 1)
                    ).unsqueeze(-1)
                    actions[blending] = torch.lerp(
                        previous_action[blending],
                        capture_actions[blending],
                        alpha,
                    )
                    blend_age[blending] += 1
                actions[~active] = 0.0
                observation_dict, _, terminated, truncated, _ = env.step(
                    actions
                )
                observations = observation_dict["policy"]
            if downstream_step == 0:
                print("downstream_first_step_done", flush=True)

            action_history.append(actions[:count].cpu().numpy())
            pen_history.append(
                torch.cat(
                    (direct.pen_pos[:count], direct.pen_quat[:count]), dim=-1
                ).cpu().numpy()
            )
            step_penetration_m = (
                direct.episode_max_penetration[:count].detach().cpu().numpy()
            )
            max_penetration_mm = max(
                max_penetration_mm,
                float(step_penetration_m.max() * 1000.0),
            )
            entered = (
                active
                & ~capture_phase
                & (
                    direct.stage_hold_count[:count]
                    >= direct.cfg.p5_hold_steps
                )
            )
            if torch.any(entered):
                middle_successes += int(entered.sum().item())
                handoff_action_jump.extend(
                    torch.linalg.norm(
                        capture_actions[entered] - actions[entered], dim=-1
                    ).cpu().tolist()
                )
                handoff_target_jump.extend(
                    torch.linalg.norm(
                        direct.processed_targets[:count][entered]
                        - previous_target[entered],
                        dim=-1,
                    ).cpu().tolist()
                )
                previous_action[entered] = actions[entered]
                blend_age[entered] = 0
                direct.p5_stage_ids[entered] = 4
                direct.stage_hold_count[entered] = 0
                direct.p5_hold_count[entered] = 0
                direct.episode_p5_success[entered] = False
                direct.target_hand_qpos[entered] = direct.policy_targets[
                    entered
                ]
                direct.target_hand_qpos[entered, :4] = p5_hand[:4]
                direct.target_hand_qpos[entered, 16:20] = p5_hand[16:20]
                capture_phase |= entered
                observations = direct._get_observations()["policy"]

            success_now = (
                active
                & capture_phase
                & (
                    direct.stage_hold_count[:count]
                    >= args.capture_hold_steps
                )
            )
            success |= success_now
            current_global_drift = torch.linalg.norm(
                direct.pen_pos[:count] - global_initial_pen_pos, dim=-1
            )
            done = (terminated | truncated).reshape(-1)[:count]
            drift_failed = current_global_drift > 0.060
            penetration_failed = torch.as_tensor(
                step_penetration_m > penetration_limit_m,
                device=direct.device,
            )
            any_failure = done | drift_failed | penetration_failed
            failed_now = active & ~success & any_failure
            failed |= failed_now
            active &= ~(success_now | failed_now)
            previous_target = direct.processed_targets[:count].clone()
            if not torch.any(active):
                break
            if (downstream_step + 1) % 50 == 0:
                print(
                    f"batch={batch_start // args.num_envs + 1} "
                    f"downstream_step={downstream_step + 1} "
                    f"active={int(active.sum().item())}",
                    flush=True,
                )

        downstream_successes += int(success.sum().item())
        dropped += int(
            (
                failed
                & (
                    torch.linalg.norm(
                        direct.pen_pos[:count] - global_initial_pen_pos, dim=-1
                    )
                    > 0.060
                )
            ).sum().item()
        )
        penetrated += int(
            (
                direct.episode_max_penetration[:count]
                > penetration_limit_m
            ).sum().item()
        )
        actions_np = np.stack(action_history, axis=1)
        pen_np = np.stack(pen_history, axis=1)
        for local_index in torch.nonzero(
            success, as_tuple=False
        ).squeeze(-1).cpu().tolist():
            successful_rows.append(int(batch_rows[local_index]))
            successful_actions.append(actions_np[local_index])
            successful_pen_qpos.append(pen_np[local_index])
            successful_lengths.append(actions_np.shape[1])
        completed += count
        print(
            f"batch={batch_start // args.num_envs + 1} "
            f"completed={completed} downstream_success={downstream_successes}",
            flush=True,
        )

    conditional_rate = downstream_successes / max(completed, 1)
    full_chain_rate = args.upstream_success_rate * conditional_rate
    local_product = args.upstream_success_rate * args.local_p4_p5_rate
    linkage_efficiency = full_chain_rate / max(local_product, 1.0e-9)
    result = {
        "task": TASK,
        "source": str(source_path),
        "source_semantics": (
            "Validated P3-to-braked-P4 teacher targets replayed physically; "
            "no hand or pen state rewrite at the P4-to-P5 handoff."
        ),
        "online_policy_scope": "P4 middle-clear -> P5 capture",
        "seed": args.seed,
        "source_length_filter": args.source_length,
        "episodes": completed,
        "middle_successes": middle_successes,
        "downstream_successes": downstream_successes,
        "conditional_p4_p5_success_rate": conditional_rate,
        "upstream_p3_p4_success_rate": args.upstream_success_rate,
        "estimated_p3_p4_p5_no_reset_rate": full_chain_rate,
        "local_oracle_product": local_product,
        "linkage_efficiency": linkage_efficiency,
        "blend_steps": args.blend_steps,
        "control_dt_s": control_dt_s,
        "mean_handoff_action_jump_l2": float(
            np.mean(handoff_action_jump) if handoff_action_jump else 0.0
        ),
        "max_handoff_action_jump_l2": float(
            np.max(handoff_action_jump) if handoff_action_jump else 0.0
        ),
        "mean_handoff_target_jump_rad_l2": float(
            np.mean(handoff_target_jump) if handoff_target_jump else 0.0
        ),
        "drop_count": dropped,
        "drop_rate": dropped / max(completed, 1),
        "penetration_failure_count": penetrated,
        "max_penetration_mm_observed": max_penetration_mm,
        "successful_no_reset_trajectories": len(successful_rows),
        "initial_gate_pass": {
            "p3_p4_p5_rate_at_least_40pct": full_chain_rate >= 0.40,
            "linkage_efficiency_at_least_70pct": linkage_efficiency >= 0.70,
            "drop_rate_below_2pct": dropped / max(completed, 1) < 0.02,
            "max_penetration_below_0_75mm": max_penetration_mm < 0.75,
        },
        "caveat": (
            "This is the first same-episode teacher-replay handoff baseline. "
            "It is not yet a fully online P3-to-P5 policy rollout."
        ),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    if args.trajectory_output and successful_rows:
        trajectory_path = resolve(args.trajectory_output)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.asarray(successful_lengths, dtype=np.int64)
        max_length = int(lengths.max())
        actions = np.zeros(
            (len(lengths), max_length, 20), dtype=np.float32
        )
        pen_qpos = np.zeros(
            (len(lengths), max_length, 7), dtype=np.float32
        )
        for index, length in enumerate(lengths):
            actions[index, :length] = successful_actions[index][:length]
            pen_qpos[index, :length] = successful_pen_qpos[index][:length]
        np.savez_compressed(
            trajectory_path,
            source_rows=np.asarray(successful_rows, dtype=np.int64),
            lengths=lengths,
            downstream_action=actions,
            downstream_pen_qpos=pen_qpos,
            upstream_source=np.asarray(str(source_path)),
            continuous_without_handoff_reset=np.asarray(True),
        )
        result["trajectory_output"] = str(trajectory_path)
        output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
