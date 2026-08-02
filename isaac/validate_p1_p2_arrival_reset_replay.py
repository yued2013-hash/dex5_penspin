from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Replay-reset validation for dynamic P1->P2 arrival states."
)
parser.add_argument("--task", default="Dex5-P1P2-ArrivalHoldH1Slow-Direct-v0")
parser.add_argument("--bundle-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--action-mode",
    choices=("zero", "source_policy_action"),
    default="zero",
)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--max-rows", type=int, default=0)
parser.add_argument("--progress-every", type=int, default=1)
parser.add_argument("--strict-distance-mm", type=float, default=2.45)
parser.add_argument("--min-contact-force-n", type=float, default=0.03)
parser.add_argument("--max-penetration-mm", type=float, default=0.6)
parser.add_argument("--require-eligible", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subset_payload(bundle_path: Path, selected: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(bundle_path, allow_pickle=False) as data:
        count = data["source_qpos"].shape[0]
        payload = {}
        for key in data.files:
            value = data[key]
            if value.shape[:1] == (count,):
                payload[key] = value[selected]
            else:
                payload[key] = value
    return payload


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    bundle_path = Path(args_cli.bundle_path).resolve()
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    with np.load(bundle_path, allow_pickle=False) as data:
        row_count = int(data["source_qpos"].shape[0])
    if args_cli.max_rows > 0:
        row_count = min(row_count, int(args_cli.max_rows))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.bundle_path = str(bundle_path)
    env_cfg.use_all_bundle_rows = True
    env_cfg.hold_current_joint_targets_on_reset = True
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.reset_pen_rot_noise_rad = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    print(
        "P1_P2_ARRIVAL_RESET_REPLAY_START="
        + json.dumps(
            {
                "task": args_cli.task,
                "rows": row_count,
                "num_envs": args_cli.num_envs,
                "steps": args_cli.steps,
                "action_mode": args_cli.action_mode,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    strict_m = args_cli.strict_distance_mm / 1000.0
    max_pen_m = args_cli.max_penetration_mm / 1000.0
    selected_rows: list[np.ndarray] = []
    row_summaries: list[dict[str, float | int | bool]] = []
    totals = {
        "geometry": 0,
        "tim": 0,
        "geometry_tim": 0,
        "eligible": 0,
        "pass": 0,
    }

    for start in range(0, row_count, args_cli.num_envs):
        end = min(start + args_cli.num_envs, row_count)
        count = end - start
        chunk_index = start // args_cli.num_envs
        if args_cli.progress_every > 0 and chunk_index % args_cli.progress_every == 0:
            print(
                "P1_P2_ARRIVAL_RESET_REPLAY_CHUNK="
                + json.dumps({"start": start, "end": end}, ensure_ascii=False),
                flush=True,
            )
        env_ids = torch.arange(count, device=direct.device, dtype=torch.long)
        forced = torch.arange(start, end, device=direct.device, dtype=torch.long)
        if count < direct.num_envs:
            forced = torch.cat((forced, forced[:1].repeat(direct.num_envs - count)))
        direct.forced_bank_ids = forced
        env.reset()

        best_geometry = torch.zeros(count, dtype=torch.bool, device=direct.device)
        best_tim = torch.zeros(count, dtype=torch.bool, device=direct.device)
        best_geometry_tim = torch.zeros(count, dtype=torch.bool, device=direct.device)
        best_eligible = torch.zeros(count, dtype=torch.bool, device=direct.device)
        best_target_distance = torch.full((count,), float("inf"), device=direct.device)
        best_penetration = torch.zeros(count, device=direct.device)
        best_contact_force = torch.zeros((count, 5), device=direct.device)

        for _step in range(args_cli.steps):
            if args_cli.action_mode == "source_policy_action":
                local_ids = direct.current_bank_ids[env_ids]
                actions = torch.zeros((direct.num_envs, direct.cfg.action_space), device=direct.device)
                actions[env_ids] = direct.source_policy_action_bank[local_ids]
            else:
                actions = torch.zeros((direct.num_envs, direct.cfg.action_space), device=direct.device)
            observations, rewards, terminated, truncated, info = env.step(actions)
            del observations, rewards, terminated, truncated, info
            direct._compute_intermediate_values()
            eligible, _metrics = direct._instantaneous_gate()
            _, _progress, target_distance, _ = direct._ring_geometry()
            contacts = direct.contact_force_norm[:, :3] > args_cli.min_contact_force_n
            tim = contacts.all(dim=-1)
            geometry = target_distance <= strict_m
            geometry_tim = geometry & tim
            replay_pass = geometry_tim & (
                direct.episode_max_penetration <= max_pen_m
            )
            if args_cli.require_eligible:
                replay_pass &= eligible

            best_geometry |= geometry[env_ids]
            best_tim |= tim[env_ids]
            best_geometry_tim |= geometry_tim[env_ids]
            best_eligible |= eligible[env_ids]
            better = target_distance[env_ids] < best_target_distance
            best_target_distance = torch.where(
                better, target_distance[env_ids], best_target_distance
            )
            best_penetration = torch.maximum(
                best_penetration, direct.episode_max_penetration[env_ids]
            )
            best_contact_force = torch.where(
                better.unsqueeze(-1),
                direct.contact_force_norm[env_ids],
                best_contact_force,
            )

        chunk_pass = best_geometry_tim & (best_penetration <= max_pen_m)
        if args_cli.require_eligible:
            chunk_pass &= best_eligible
        selected = torch.nonzero(chunk_pass, as_tuple=False).squeeze(-1).cpu().numpy()
        if selected.size:
            selected_rows.append(np.arange(start, end, dtype=np.int64)[selected])

        totals["geometry"] += int(best_geometry.sum().item())
        totals["tim"] += int(best_tim.sum().item())
        totals["geometry_tim"] += int(best_geometry_tim.sum().item())
        totals["eligible"] += int(best_eligible.sum().item())
        totals["pass"] += int(chunk_pass.sum().item())

        for local in range(count):
            row_summaries.append(
                {
                    "row": int(start + local),
                    "pass": bool(chunk_pass[local].item()),
                    "geometry": bool(best_geometry[local].item()),
                    "tim": bool(best_tim[local].item()),
                    "geometry_tim": bool(best_geometry_tim[local].item()),
                    "eligible": bool(best_eligible[local].item()),
                    "best_target_distance_mm": float(best_target_distance[local].item() * 1000.0),
                    "max_penetration_mm": float(best_penetration[local].item() * 1000.0),
                    "contact_force_thumb_n": float(best_contact_force[local, 0].item()),
                    "contact_force_index_n": float(best_contact_force[local, 1].item()),
                    "contact_force_middle_n": float(best_contact_force[local, 2].item()),
                }
            )

    selected = (
        np.concatenate(selected_rows, axis=0)
        if selected_rows
        else np.empty((0,), dtype=np.int64)
    )
    payload = _subset_payload(bundle_path, selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    summary = {
        "task": args_cli.task,
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256(bundle_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "action_mode": args_cli.action_mode,
        "steps": args_cli.steps,
        "row_count": row_count,
        "selected_rows": int(selected.size),
        "selected_ratio": float(selected.size / max(row_count, 1)),
        "totals": totals,
        "strict_distance_mm": args_cli.strict_distance_mm,
        "min_contact_force_n": args_cli.min_contact_force_n,
        "max_penetration_mm": args_cli.max_penetration_mm,
        "require_eligible": args_cli.require_eligible,
        "row_summaries": row_summaries,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_ARRIVAL_RESET_REPLAY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
