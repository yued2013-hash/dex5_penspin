from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/m3_2_30_index_near_surface/physx_alignment_rollout_100ms.npz"

parser = argparse.ArgumentParser(description="Export the PhysX half of a paired 100 ms alignment rollout.")
parser.add_argument("--task", default="Dex5-P3Entry-Direct-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--bundle_path", type=Path, default=None)
parser.add_argument(
    "--collision_proxy_scales",
    type=float,
    nargs=5,
    default=(0.99, 1.07, 1.07, 0.84, 0.80),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def capture(direct) -> dict[str, np.ndarray]:
    direct._compute_intermediate_values()
    return {
        "hand_qpos": direct.hand_qpos.detach().cpu().numpy().copy(),
        "hand_qvel": direct.hand_qvel.detach().cpu().numpy().copy(),
        "pen_pos": direct.pen_pos.detach().cpu().numpy().copy(),
        "pen_quat": direct.pen_quat.detach().cpu().numpy().copy(),
        "pen_linvel": direct.pen_linvel.detach().cpu().numpy().copy(),
        "pen_angvel": direct.pen_angvel.detach().cpu().numpy().copy(),
        "tip_surface_dist": direct.tip_surface_dist.detach().cpu().numpy().copy(),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    if args_cli.bundle_path is not None:
        bundle_path = args_cli.bundle_path
        if not bundle_path.is_absolute():
            bundle_path = ROOT / bundle_path
        env_cfg.bundle_path = str(bundle_path.resolve())
    env_cfg.collision_proxy_radius_scales = tuple(args_cli.collision_proxy_scales)
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.enable_contact_sensor = False
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 0.25)
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()

    bank_ids = direct.current_bank_ids.detach().cpu().numpy().copy()
    source_qpos = direct.source_qpos_bank[direct.current_bank_ids].detach().cpu().numpy().copy()
    source_qvel = direct.source_qvel_bank[direct.current_bank_ids].detach().cpu().numpy().copy()
    source_target = direct.source_targets.detach().cpu().numpy().copy()
    frames = [capture(direct)]
    done_frames: list[np.ndarray] = []
    actions = torch.zeros((args_cli.num_envs, 20), device=direct.device)
    for _ in range(args_cli.steps):
        direct._pre_physics_step(actions)
        for _ in range(direct.cfg.decimation):
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(dt=direct.physics_dt)
        direct.episode_length_buf += 1
        direct._compute_intermediate_values()
        terminated, truncated = direct._compute_dones()
        done_frames.append((terminated | truncated).detach().cpu().numpy().copy())
        frames.append(capture(direct))

    output = args_cli.output if args_cli.output.is_absolute() else ROOT / args_cli.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": np.asarray(args_cli.task),
        "physx_dt_s": np.asarray(float(env_cfg.sim.dt) * int(env_cfg.decimation)),
        "steps": np.asarray(args_cli.steps),
        "bank_ids": bank_ids,
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target": source_target,
        "collision_proxy_radius_scales": np.asarray(
            args_cli.collision_proxy_scales, dtype=np.float64
        ),
        "done": np.stack(done_frames),
    }
    for key in frames[0]:
        payload[f"physx_{key}"] = np.stack([frame[key] for frame in frames])
    np.savez_compressed(output, **payload)
    print(
        f"PHYSX_ALIGNMENT_ROLLOUT={output} envs={args_cli.num_envs} "
        f"steps={args_cli.steps} done={int(np.sum(payload['done']))}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
