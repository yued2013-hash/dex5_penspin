from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Action-conditioned 50-100 ms P3 dynamic-readiness audit."
)
parser.add_argument(
    "--task", default="Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0"
)
parser.add_argument("--probe-count", type=int, default=31)
parser.add_argument("--max-offset-rad", type=float, default=0.06)
parser.add_argument("--batch-states", type=int, default=24)
parser.add_argument("--horizons", type=int, nargs="+", default=(3, 6))
parser.add_argument("--anchor-ids", type=int, nargs="+", default=(996, 1245, 1171, 1203))
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from o1_14_probe_design import make_symmetric_probes  # noqa: E402


CONTACT_REBUILD_STEPS = 2
SEED = 20260731
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MIN_TOTAL_FORCE_N = 0.06
MAX_TOTAL_FORCE_N = 6.0
MAX_HEIGHT_LOSS_M = 0.005
MIN_AXIS_ALIGNMENT = 0.50
TRANSFER_FORCE_N = 0.03


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clear_events(direct) -> None:
    direct.episode_release_seen.zero_()
    direct.episode_recontact_seen.zero_()
    direct.episode_release_count.zero_()
    direct.episode_recontact_count.zero_()
    direct.episode_tim_overlap_steps.zero_()
    direct.episode_ballistic_seen.zero_()
    direct.episode_failure_type_seen.zero_()


def _actions_for_target(direct, desired_target: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        (desired_target - direct.processed_targets) / direct.cfg.action_scale,
        -1.0,
        1.0,
    )


def _run_probe_batch(
    env,
    direct,
    bank_ids: torch.Tensor,
    offsets: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    direct.forced_bank_ids = bank_ids
    env.reset()
    hold = torch.zeros(
        (direct.num_envs, direct.cfg.action_space), device=direct.device
    )
    for _ in range(CONTACT_REBUILD_STEPS):
        env.step(hold)
    _clear_events(direct)

    initial_height = direct.pen_pos[:, 2].clone()
    initial_vertical_velocity = direct.pen_linvel[:, 2].clone()
    initial_axis_rate = (
        direct.cfg.target_rotation_direction * direct.pen_angvel[:, 2]
    ).clone()
    initial_force = direct.contact_force_norm.clone()
    initial_total_force = initial_force.sum(dim=-1)
    initial_vertical_support = direct.total_contact_force_on_pen_w[:, 2].clone()
    initial_contacts = initial_force > direct.cfg.contact_force_threshold_n
    dominant = torch.argmax(initial_force[:, :3], dim=-1)
    dominant_initial_force = initial_force[:, :3].gather(
        1, dominant.unsqueeze(-1)
    ).squeeze(-1)
    non_dominant = torch.ones_like(initial_force, dtype=torch.bool)
    non_dominant.scatter_(1, dominant.unsqueeze(-1), False)

    min_height_delta = torch.zeros_like(initial_height)
    min_total_force = torch.full_like(initial_total_force, float("inf"))
    max_total_force = torch.zeros_like(initial_total_force)
    max_new_support_gain = torch.zeros_like(initial_total_force)
    max_old_support_unload = torch.zeros_like(initial_total_force)
    best_pair_margin = torch.full_like(initial_total_force, -float("inf"))
    max_positive_axis_delta = torch.full_like(initial_axis_rate, -float("inf"))
    axis_sum = torch.zeros_like(initial_axis_rate)
    axis_steps = torch.zeros_like(dominant)
    unexpected_done = torch.zeros_like(initial_contacts[:, 0])

    for step in range(horizon):
        fraction = float(step + 1) / float(horizon)
        desired = direct.source_targets + fraction * offsets
        _, _, terminated, truncated, _ = env.step(_actions_for_target(direct, desired))
        unexpected_done |= terminated | truncated

        force = direct.contact_force_norm
        total_force = force.sum(dim=-1)
        min_total_force = torch.minimum(min_total_force, total_force)
        max_total_force = torch.maximum(max_total_force, total_force)
        new_gain = torch.where(
            non_dominant,
            force - initial_force,
            torch.full_like(force, -1.0e6),
        ).max(dim=-1).values
        old_force = force[:, :3].gather(1, dominant.unsqueeze(-1)).squeeze(-1)
        old_unload = dominant_initial_force - old_force
        max_new_support_gain = torch.maximum(max_new_support_gain, new_gain)
        max_old_support_unload = torch.maximum(max_old_support_unload, old_unload)
        pair_margin = torch.minimum(
            new_gain / TRANSFER_FORCE_N - 1.0,
            old_unload / TRANSFER_FORCE_N - 1.0,
        )
        best_pair_margin = torch.maximum(best_pair_margin, pair_margin)
        min_height_delta = torch.minimum(
            min_height_delta, direct.pen_pos[:, 2] - initial_height
        )
        signed_axis_rate = (
            direct.cfg.target_rotation_direction * direct.pen_angvel[:, 2]
        )
        max_positive_axis_delta = torch.maximum(
            max_positive_axis_delta, signed_axis_rate - initial_axis_rate
        )
        angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        moving = angular_speed >= direct.cfg.axis_alignment_min_angular_speed_rad_s
        alignment = signed_axis_rate / angular_speed.clamp_min(1.0e-6)
        axis_sum += torch.where(moving, alignment, 0.0)
        axis_steps += moving.long()

    duration = horizon * float(direct.cfg.sim.dt * direct.cfg.decimation)
    final_force = direct.contact_force_norm.clone()
    final_total_force = final_force.sum(dim=-1)
    final_vertical_support = direct.total_contact_force_on_pen_w[:, 2].clone()
    final_contacts = final_force > direct.cfg.contact_force_threshold_n
    final_axis_rate = direct.cfg.target_rotation_direction * direct.pen_angvel[:, 2]
    angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
    final_alignment = final_axis_rate / angular_speed.clamp_min(1.0e-6)
    contact_changed = initial_contacts ^ final_contacts
    event_seen = direct.episode_release_seen | direct.episode_recontact_seen
    tim_response = (contact_changed[:, :3] | event_seen[:, :3]).any(dim=-1)
    unsafe = (
        unexpected_done
        | direct.episode_ballistic_seen
        | direct.episode_failure_type_seen[:, 0]
        | direct.episode_failure_type_seen[:, 2]
        | direct.episode_failure_type_seen[:, 3]
        | direct.episode_failure_type_seen[:, 4]
        | direct.episode_failure_type_seen[:, 5]
        | direct.episode_failure_type_seen[:, 6]
    )
    short_admissible = (
        tim_response
        & (best_pair_margin >= 0.0)
        & (min_total_force >= MIN_TOTAL_FORCE_N)
        & (max_total_force <= MAX_TOTAL_FORCE_N)
        & (min_height_delta >= -MAX_HEIGHT_LOSS_M)
        & (axis_steps >= 2)
        & ((axis_sum / axis_steps.clamp_min(1).float()) >= MIN_AXIS_ALIGNMENT)
        & ~unsafe
    )
    return {
        "initial_force_n": initial_force,
        "final_force_delta_n": final_force - initial_force,
        "initial_total_force_n": initial_total_force,
        "final_total_force_delta_n": final_total_force - initial_total_force,
        "total_force_slope_n_s": (final_total_force - initial_total_force) / duration,
        "vertical_support_slope_n_s": (
            final_vertical_support - initial_vertical_support
        ) / duration,
        "min_total_force_n": min_total_force,
        "max_total_force_n": max_total_force,
        "new_support_gain_n": max_new_support_gain,
        "old_support_unload_n": max_old_support_unload,
        "transfer_pair_margin": best_pair_margin,
        "axis_rate_delta_rad_s": final_axis_rate - initial_axis_rate,
        "max_positive_axis_delta_rad_s": max_positive_axis_delta,
        "mean_axis_alignment": axis_sum / axis_steps.clamp_min(1).float(),
        "final_axis_alignment": final_alignment,
        "wrong_axis_speed_rad_s": torch.linalg.norm(direct.pen_angvel[:, :2], dim=-1),
        "final_height_delta_m": direct.pen_pos[:, 2] - initial_height,
        "min_height_delta_m": min_height_delta,
        "height_acceleration_m_s2": (
            direct.pen_linvel[:, 2] - initial_vertical_velocity
        ) / duration,
        "contact_changed": contact_changed,
        "release_seen": direct.episode_release_seen.clone(),
        "recontact_seen": direct.episode_recontact_seen.clone(),
        "tim_response": tim_response,
        "short_admissible": short_admissible,
        "unsafe": unsafe,
    }


def main() -> None:
    started = time.perf_counter()
    cfg_probe = parse_env_cfg(args.task, device=args.device, num_envs=1)
    bank_path = Path(cfg_probe.bundle_path).resolve()
    bank_hash_before = _sha256(bank_path)
    with np.load(bank_path, allow_pickle=False) as bank:
        canonical = np.asarray(bank["metadata_canonical_id"])
        split = np.asarray(bank["metadata_split"])
        bank_ids = np.flatnonzero(
            (canonical == "P3") & (split == cfg_probe.reset_split)
        ).astype(np.int64)
        parents = np.asarray(bank["metadata_parent_state_id"])[bank_ids].astype(np.int64)
        sources = np.asarray(bank["metadata_source_trajectory_id"])[bank_ids].astype(np.int64)
    missing = sorted(set(args.anchor_ids) - set(bank_ids.tolist()))
    if missing:
        raise ValueError(f"Anchor IDs are not frozen P3/{cfg_probe.reset_split}: {missing}")

    probe_offsets_np = make_symmetric_probes(
        int(cfg_probe.action_space), args.probe_count, args.max_offset_rad, SEED
    )
    num_envs = args.batch_states * args.probe_count
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = SEED
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (max(args.horizons) + CONTACT_REBUILD_STEPS + 8) * control_dt
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    if bank_hash_before != direct.cfg.expected_bundle_sha256:
        raise RuntimeError("Configured frozen-bank hash differs from the on-disk bank")
    probe_offsets = torch.as_tensor(
        probe_offsets_np, dtype=torch.float32, device=direct.device
    )

    collected: dict[int, dict[str, list[np.ndarray]]] = {
        horizon: {} for horizon in args.horizons
    }
    for start in range(0, len(bank_ids), args.batch_states):
        actual = bank_ids[start : start + args.batch_states]
        padded = np.pad(
            actual,
            (0, args.batch_states - len(actual)),
            mode="edge",
        )
        forced = torch.as_tensor(padded, dtype=torch.long, device=direct.device)
        forced = forced.repeat_interleave(args.probe_count)
        offsets = probe_offsets.repeat(args.batch_states, 1)
        for horizon in args.horizons:
            metrics = _run_probe_batch(env, direct, forced, offsets, horizon)
            keep = len(actual) * args.probe_count
            for key, value in metrics.items():
                shaped = value[:keep].reshape(len(actual), args.probe_count, *value.shape[1:])
                collected[horizon].setdefault(key, []).append(
                    shaped.detach().cpu().numpy()
                )
        print(
            f"O1_14_BATCH={start // args.batch_states + 1} "
            f"rows={start}:{start + len(actual)}",
            flush=True,
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "bank_ids": bank_ids,
        "parent_state_ids": parents,
        "source_trajectory_ids": sources,
        "probe_offsets_rad": probe_offsets_np,
        "horizons_steps": np.asarray(args.horizons, dtype=np.int64),
        "horizons_ms": np.asarray(args.horizons, dtype=np.float64) * control_dt * 1000.0,
        "anchor_ids": np.asarray(args.anchor_ids, dtype=np.int64),
    }
    for horizon in args.horizons:
        for key, chunks in collected[horizon].items():
            payload[f"h{horizon}_{key}"] = np.concatenate(chunks, axis=0)
    np.savez_compressed(output, **payload)
    bank_hash_after = _sha256(bank_path)
    if bank_hash_after != bank_hash_before:
        raise RuntimeError("Frozen O0 bank changed during O1.14")
    sidecar = {
        "status": "PASS_RAW_DYNAMIC_PROBE_AUDIT",
        "phase": "O1.14 Dynamic Handoff Readiness Audit",
        "artifact_role": "analysis_only_action_response_not_teacher_or_reset_dataset",
        "output_npz": str(output),
        "task": args.task,
        "bank_path": str(bank_path),
        "bank_sha256_before": bank_hash_before,
        "bank_sha256_after": bank_hash_after,
        "P3_rows": int(len(bank_ids)),
        "probe_count": args.probe_count,
        "probe_design": "zero plus deterministic antithetic random TIM target offsets",
        "max_offset_rad": args.max_offset_rad,
        "horizons_steps": list(args.horizons),
        "horizons_ms": [step * control_dt * 1000.0 for step in args.horizons],
        "anchor_ids": list(args.anchor_ids),
        "contact_rebuild_steps": CONTACT_REBUILD_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "teacher_generated": False,
            "O0_modified": False,
            "reset_bank_modified": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("O1_14_RAW=" + json.dumps(sidecar, ensure_ascii=False), flush=True)
    del direct.forced_bank_ids
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
