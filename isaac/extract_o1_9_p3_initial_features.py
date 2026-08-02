from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Read-only PhysX extraction of initial P3 handoff features."
)
parser.add_argument(
    "--task", default="Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0"
)
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


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _append_vector(
    names: list[str],
    groups: list[str],
    values: list[torch.Tensor],
    prefix: str,
    group: str,
    tensor: torch.Tensor,
    components: tuple[str, ...],
    scale: float = 1.0,
) -> None:
    for index, component in enumerate(components):
        names.append(f"{prefix}_{component}")
        groups.append(group)
        values.append(scale * tensor[:, index])


def _build_features(direct) -> tuple[list[str], list[str], torch.Tensor]:
    names: list[str] = []
    groups: list[str] = []
    values: list[torch.Tensor] = []

    relative_tip = direct.fingertip_pos_local - direct.pen_pos.unsqueeze(1)
    axial_mm = 1000.0 * torch.sum(
        relative_tip * direct.pen_axis.unsqueeze(1), dim=-1
    )
    surface_gap_mm = 1000.0 * (
        direct.tip_surface_dist - direct.fingertip_radii
    )
    for finger_index, finger in enumerate(FINGERS):
        names.extend((f"geometry_{finger}_axial_mm", f"geometry_{finger}_gap_mm"))
        groups.extend(("geometry", "geometry"))
        values.extend((axial_mm[:, finger_index], surface_gap_mm[:, finger_index]))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        names.append(f"geometry_{FINGERS[first]}_{FINGERS[second]}_distance_mm")
        groups.append("geometry")
        values.append(
            1000.0
            * torch.linalg.norm(
                direct.fingertip_pos_local[:, first]
                - direct.fingertip_pos_local[:, second],
                dim=-1,
            )
        )

    for finger_index, finger in enumerate(FINGERS):
        _append_vector(
            names,
            groups,
            values,
            f"wrench_{finger}_force_on_pen_n",
            "wrench",
            direct.contact_force_on_pen_w[:, finger_index],
            ("x", "y", "z"),
        )
        names.extend(
            (
                f"wrench_{finger}_force_norm_n",
                f"wrench_{finger}_tangential_force_n",
            )
        )
        groups.extend(("wrench", "wrench"))
        values.extend(
            (
                direct.contact_force_norm[:, finger_index],
                direct.contact_tangential_force_norm[:, finger_index],
            )
        )
    _append_vector(
        names,
        groups,
        values,
        "wrench_total_force_on_pen_n",
        "wrench",
        direct.total_contact_force_on_pen_w,
        ("x", "y", "z"),
    )
    _append_vector(
        names,
        groups,
        values,
        "wrench_total_torque_about_com_nmm",
        "wrench",
        direct.total_contact_torque_on_pen_w,
        ("x", "y", "z"),
        scale=1000.0,
    )

    _append_vector(
        names,
        groups,
        values,
        "pen_position_mm",
        "pen_pose",
        direct.pen_pos,
        ("x", "y", "z"),
        scale=1000.0,
    )
    _append_vector(
        names,
        groups,
        values,
        "pen_axis",
        "pen_pose",
        direct.pen_axis,
        ("x", "y", "z"),
    )
    _append_vector(
        names,
        groups,
        values,
        "pen_linear_velocity_m_s",
        "velocity",
        direct.pen_linvel,
        ("x", "y", "z"),
    )
    _append_vector(
        names,
        groups,
        values,
        "pen_angular_velocity_rad_s",
        "velocity",
        direct.pen_angvel,
        ("x", "y", "z"),
    )

    force_norm = direct.contact_force_norm
    force_sum = force_norm.sum(dim=-1)
    tim_force_sum = force_norm[:, :3].sum(dim=-1)
    force_share = force_norm / force_sum.unsqueeze(-1).clamp_min(1.0e-9)
    support_values = (
        ("support_total_vertical_surplus_n", direct.total_gravity_force_residual_w[:, 2]),
        ("support_TIM_vertical_surplus_n", direct.tim_gravity_force_residual_w[:, 2]),
        (
            "support_total_gravity_residual_norm_n",
            torch.linalg.norm(direct.total_gravity_force_residual_w, dim=-1),
        ),
        (
            "support_TIM_gravity_residual_norm_n",
            torch.linalg.norm(direct.tim_gravity_force_residual_w, dim=-1),
        ),
        ("support_total_contact_force_sum_n", force_sum),
        ("support_TIM_contact_force_sum_n", tim_force_sum),
        ("support_strongest_finger_share", force_share.max(dim=-1).values),
        (
            "support_active_finger_count",
            (force_norm >= direct.cfg.contact_force_threshold_n).float().sum(dim=-1),
        ),
        (
            "support_TIM_equivalent_moment_arm_mm",
            1000.0 * direct.tim_equivalent_moment_arm,
        ),
        ("support_force_ceiling_margin_n", 6.0 - force_sum),
    )
    for name, value in support_values:
        names.append(name)
        groups.append("support_margin")
        values.append(value)

    return names, groups, torch.stack(values, dim=-1)


def main() -> None:
    cfg_probe = parse_env_cfg(args.task, device=args.device, num_envs=1)
    with np.load(cfg_probe.bundle_path, allow_pickle=False) as bank:
        canonical = np.asarray(bank["metadata_canonical_id"])
        split = np.asarray(bank["metadata_split"])
        source_rows = np.flatnonzero(
            (canonical == "P3") & (split == cfg_probe.reset_split)
        ).astype(np.int64)
        metadata = {
            "parent_state_id": np.asarray(bank["metadata_parent_state_id"])[
                source_rows
            ].astype(np.int64),
            "source_trajectory_id": np.asarray(
                bank["metadata_source_trajectory_id"]
            )[source_rows].astype(np.int64),
            "state_id": np.asarray(bank["metadata_state_id"])[source_rows].astype(
                np.int64
            ),
        }

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(source_rows))
    cfg.seed = 20260731
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    direct.forced_bank_ids = torch.as_tensor(
        source_rows, dtype=torch.long, device=direct.device
    )
    env.reset()
    contact_rebuild_steps = 2
    hold_action = torch.zeros(
        (direct.num_envs, direct.cfg.action_space), device=direct.device
    )
    for _ in range(contact_rebuild_steps):
        env.step(hold_action)
    names, groups, features = _build_features(direct)
    finite = torch.isfinite(features).all(dim=-1)
    if not bool(finite.all().item()):
        bad_ids = source_rows[~finite.detach().cpu().numpy()].tolist()
        raise RuntimeError(f"Non-finite O1.9 features for bank IDs: {bad_ids}")

    report = {
        "status": "PASS",
        "artifact_role": "analysis_only_reset_replay_feature_snapshot_not_training_data",
        "task": args.task,
        "bank_sha256": direct.cfg.expected_bundle_sha256,
        "canonical_id": "P3",
        "reset_split": direct.cfg.reset_split,
        "rows": len(source_rows),
        "feature_names": names,
        "feature_groups": groups,
        "bank_ids": source_rows.tolist(),
        "parent_state_ids": metadata["parent_state_id"].tolist(),
        "source_trajectory_ids": metadata["source_trajectory_id"].tolist(),
        "state_ids": metadata["state_id"].tolist(),
        "features": features.detach().cpu().tolist(),
        "physics_reinstantiation": (
            "env.reset with forced_bank_ids plus two zero-residual hold steps "
            "to refresh PhysX contacts"
        ),
        "contact_rebuild_steps": contact_rebuild_steps,
        "ppo_trained": False,
        "reward_modified": False,
        "o0_modified": False,
        "teacher_generated": False,
        "trajectory_exported": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "O1_9_P3_FEATURE_EXTRACTION="
        + json.dumps(
            {
                "rows": len(source_rows),
                "features": len(names),
                "bank_sha256": report["bank_sha256"],
            }
        ),
        flush=True,
    )
    del direct.forced_bank_ids
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
