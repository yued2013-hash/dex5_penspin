from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/candidate"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0"
)

parser = argparse.ArgumentParser(description="Run O0 R0-R3 canonical reset validation.")
parser.add_argument(
    "--canonical-id",
    required=True,
    choices=("C1", "C2", "C3", "C4", "C5", "C6", "P1", "P2", "P3", "P4", "P5"),
)
parser.add_argument("--input", default="")
parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--task", default="Dex5-CanonicalReset-Physics-Direct-v0")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--max-candidates", type=int, default=1000)
parser.add_argument("--seed", type=int, default=20260729)
parser.add_argument("--r1-physics-steps", type=int, default=2)
parser.add_argument("--static-duration-ms", type=float, default=100.0)
parser.add_argument("--dynamic-duration-ms", type=float, default=100.0)
parser.add_argument("--r3a-trials", type=int, default=10)
parser.add_argument("--r3b-trials", type=int, default=20)
parser.add_argument("--max-penetration-mm", type=float, default=0.5)
parser.add_argument("--max-drop-mm", type=float, default=20.0)
parser.add_argument("--max-drift-mm", type=float, default=25.0)
parser.add_argument("--max-linear-speed", type=float, default=1.0)
parser.add_argument("--max-angular-speed", type=float, default=50.0)
parser.add_argument("--static-max-final-angular-speed", type=float, default=8.0)
parser.add_argument("--min-contact-force", type=float, default=0.03)
parser.add_argument("--static-min-contacts", type=int, default=2)
parser.add_argument("--static-contact-ratio", type=float, default=0.95)
parser.add_argument("--dynamic-contact-ratio", type=float, default=0.50)
parser.add_argument("--required-contact-ratio", type=float, default=0.80)
parser.add_argument("--q-noise", type=float, default=0.0010)
parser.add_argument("--dq-noise", type=float, default=0.025)
parser.add_argument("--object-pos-noise", type=float, default=0.0002)
parser.add_argument("--object-rot-noise-deg", type=float, default=0.25)
parser.add_argument("--object-vel-noise", type=float, default=0.010)
parser.add_argument("--mass-scale", type=float, default=0.03)
parser.add_argument("--friction-scale", type=float, default=0.05)
parser.add_argument("--r3a-min-ratio", type=float, default=0.90)
parser.add_argument("--r3b-r1-min-ratio", type=float, default=0.90)
parser.add_argument("--r3b-r2-min-ratio", type=float, default=0.80)
parser.add_argument("--progress-every", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from dex5_penspin.isaac.canonical_reset_schema import (  # noqa: E402
    save_npz_atomic,
    sha256_file,
    subset_payload,
    validate_r0_arrays,
    write_json_atomic,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _select_candidates(payload: dict[str, np.ndarray], maximum: int) -> np.ndarray:
    count = payload["metadata_state_id"].shape[0]
    all_rows = np.arange(count, dtype=np.int64)
    if maximum <= 0 or count <= maximum:
        return all_rows
    state_ids = np.asarray(payload["metadata_state_id"], dtype=np.int64)
    # Stable hash-like ordering avoids selecting only one contiguous trajectory region.
    order = np.argsort(state_ids ^ np.int64(0x5DEECE66D), kind="stable")
    return np.sort(order[:maximum].astype(np.int64))


def _physics_steps(duration_ms: float, physics_dt: float) -> int:
    return max(1, int(np.ceil((duration_ms / 1000.0) / physics_dt)))


def _reset_perturbation(direct, enabled: bool) -> None:
    if enabled:
        direct.cfg.reset_joint_pos_noise = args_cli.q_noise
        direct.cfg.reset_joint_vel_noise = args_cli.dq_noise
        direct.cfg.reset_pen_pos_noise = args_cli.object_pos_noise
        direct.cfg.reset_pen_vel_noise = args_cli.object_vel_noise
        direct.cfg.canonical_reset_pen_rot_noise_rad = np.deg2rad(
            args_cli.object_rot_noise_deg
        )
        direct.cfg.canonical_mass_scale_range = (
            1.0 - args_cli.mass_scale,
            1.0 + args_cli.mass_scale,
        )
        direct.cfg.canonical_static_friction_scale_range = (
            1.0 - args_cli.friction_scale,
            1.0 + args_cli.friction_scale,
        )
        direct.cfg.canonical_dynamic_friction_scale_range = (
            1.0 - args_cli.friction_scale,
            1.0 + args_cli.friction_scale,
        )
    else:
        direct.cfg.reset_joint_pos_noise = 0.0
        direct.cfg.reset_joint_vel_noise = 0.0
        direct.cfg.reset_pen_pos_noise = 0.0
        direct.cfg.reset_pen_vel_noise = 0.0
        direct.cfg.canonical_reset_pen_rot_noise_rad = 0.0
        direct.cfg.canonical_mass_scale_range = (1.0, 1.0)
        direct.cfg.canonical_static_friction_scale_range = (1.0, 1.0)
        direct.cfg.canonical_dynamic_friction_scale_range = (1.0, 1.0)


def _run_replays(
    env,
    bank_rows: np.ndarray,
    *,
    trials: int,
    perturbed: bool,
    state_type: str,
    physics_steps: int,
    tag: str,
) -> dict[str, np.ndarray]:
    direct = env.unwrapped
    _reset_perturbation(direct, perturbed)
    repeated_rows = np.repeat(np.asarray(bank_rows, dtype=np.int64), trials)
    result_count = repeated_rows.size
    r1_pass = np.zeros(result_count, dtype=np.bool_)
    r2_pass = np.zeros(result_count, dtype=np.bool_)
    max_penetration = np.zeros(result_count, dtype=np.float32)
    max_linear_speed = np.zeros(result_count, dtype=np.float32)
    max_angular_speed = np.zeros(result_count, dtype=np.float32)
    max_drop = np.zeros(result_count, dtype=np.float32)
    max_drift = np.zeros(result_count, dtype=np.float32)
    contact_ratio = np.zeros(result_count, dtype=np.float32)
    final_contact_force = np.zeros((result_count, 5), dtype=np.float32)
    final_contact_positions = np.zeros((result_count, 5, 3), dtype=np.float32)
    final_contact_state = np.zeros((result_count, 5), dtype=np.bool_)

    max_penetration_m = args_cli.max_penetration_mm / 1000.0
    max_drop_m = args_cli.max_drop_mm / 1000.0
    max_drift_m = args_cli.max_drift_mm / 1000.0
    total_steps = max(args_cli.r1_physics_steps, physics_steps)
    required_contact_ids = {
        # C6 is the cycle-reset pinch: middle/ring/pinky may move nearby, but
        # thumb and index must both carry contact through the replay.
        "C6": (0, 1),
        "P1": (0, 1, 2),
        "P2": (0, 1, 2),
        "P3": (0, 1, 2),
        "P4": (0, 1, 2),
        "P5": (0, 1),
    }.get(args_cli.canonical_id, ())
    forbidden_contact_ids = {
        "C6": (2, 3, 4),
        "P3": (3, 4),
        "P4": (3, 4),
        "P5": (2, 3, 4),
    }.get(args_cli.canonical_id, ())

    for batch_index, start in enumerate(range(0, result_count, direct.num_envs)):
        end = min(start + direct.num_envs, result_count)
        count = end - start
        rows = torch.as_tensor(
            repeated_rows[start:end], device=direct.device, dtype=torch.long
        )
        if count < direct.num_envs:
            rows = torch.cat((rows, rows[:1].repeat(direct.num_envs - count)))
        direct.forced_bank_ids = rows
        env.reset()
        valid = torch.ones(count, dtype=torch.bool, device=direct.device)
        maximum_pen = torch.zeros(count, device=direct.device)
        maximum_lin = torch.zeros(count, device=direct.device)
        maximum_ang = torch.zeros(count, device=direct.device)
        maximum_drop = torch.zeros(count, device=direct.device)
        maximum_drift = torch.zeros(count, device=direct.device)
        contact_step_count = torch.zeros(count, device=direct.device)
        required_contact_step_count = torch.zeros(
            (count, len(required_contact_ids)), device=direct.device
        )
        forbidden_contact_step_count = torch.zeros(
            (count, len(forbidden_contact_ids)), device=direct.device
        )
        r1_snapshot = torch.zeros(count, dtype=torch.bool, device=direct.device)

        for step in range(total_steps):
            if step % int(direct.cfg.decimation) == 0:
                direct.set_replay_control_step(step // int(direct.cfg.decimation))
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(direct.cfg.sim.dt)
            direct._compute_intermediate_values()

            finite = (
                torch.isfinite(direct.hand_qpos[:count]).all(dim=-1)
                & torch.isfinite(direct.hand_qvel[:count]).all(dim=-1)
                & torch.isfinite(direct.pen_pos[:count]).all(dim=-1)
                & torch.isfinite(direct.pen_quat[:count]).all(dim=-1)
                & torch.isfinite(direct.pen_linvel[:count]).all(dim=-1)
                & torch.isfinite(direct.pen_angvel[:count]).all(dim=-1)
            )
            penetration = direct.episode_max_penetration[:count]
            linear_speed = torch.linalg.norm(direct.pen_linvel[:count], dim=-1)
            angular_speed = torch.linalg.norm(direct.pen_angvel[:count], dim=-1)
            drop = torch.clamp(
                direct.init_pen_pos[:count, 2] - direct.pen_pos[:count, 2], min=0.0
            )
            drift = torch.linalg.norm(
                direct.pen_pos[:count] - direct.init_pen_pos[:count], dim=-1
            )
            contacts = direct.contact_force_norm[:count] > args_cli.min_contact_force
            proximity = direct.tip_surface_dist[:count] <= (
                direct.fingertip_radii[:count] + 0.001
            )
            reasonable_contact = (contacts | proximity).any(dim=-1)
            if state_type == "static":
                contact_ok = contacts.sum(dim=-1) >= args_cli.static_min_contacts
            else:
                contact_ok = reasonable_contact
            contact_step_count += contact_ok.float()
            if required_contact_ids:
                required_contact_step_count += contacts[
                    :, list(required_contact_ids)
                ].float()
            if forbidden_contact_ids:
                forbidden_contact_step_count += contacts[
                    :, list(forbidden_contact_ids)
                ].float()
            step_valid = (
                finite
                & (penetration <= max_penetration_m)
                & (linear_speed <= args_cli.max_linear_speed)
                & (angular_speed <= args_cli.max_angular_speed)
                & (drop <= max_drop_m)
                & (drift <= max_drift_m)
            )
            valid &= step_valid
            maximum_pen = torch.maximum(maximum_pen, penetration)
            maximum_lin = torch.maximum(maximum_lin, linear_speed)
            maximum_ang = torch.maximum(maximum_ang, angular_speed)
            maximum_drop = torch.maximum(maximum_drop, drop)
            maximum_drift = torch.maximum(maximum_drift, drift)
            if step + 1 == args_cli.r1_physics_steps:
                required_contact_now = (
                    contacts[:, list(required_contact_ids)].all(dim=-1)
                    if required_contact_ids
                    else torch.ones(count, dtype=torch.bool, device=direct.device)
                )
                forbidden_contact_clear = (
                    ~contacts[:, list(forbidden_contact_ids)].any(dim=-1)
                    if forbidden_contact_ids
                    else torch.ones(count, dtype=torch.bool, device=direct.device)
                )
                r1_snapshot = (
                    step_valid
                    & reasonable_contact
                    & required_contact_now
                    & forbidden_contact_clear
                )

        if state_type == "static":
            required_contact_ratio = args_cli.static_contact_ratio
            final_speed_ok = (
                torch.linalg.norm(direct.pen_angvel[:count], dim=-1)
                <= args_cli.static_max_final_angular_speed
            )
        else:
            required_contact_ratio = args_cli.dynamic_contact_ratio
            final_speed_ok = torch.ones(count, dtype=torch.bool, device=direct.device)
        replay_contact_ratio = contact_step_count / float(total_steps)
        required_contacts_ok = (
            (
                required_contact_step_count / float(total_steps)
                >= args_cli.required_contact_ratio
            ).all(dim=-1)
            if required_contact_ids
            else torch.ones(count, dtype=torch.bool, device=direct.device)
        )
        forbidden_contacts_ok = (
            (
                forbidden_contact_step_count / float(total_steps)
                <= (1.0 - args_cli.required_contact_ratio)
            ).all(dim=-1)
            if forbidden_contact_ids
            else torch.ones(count, dtype=torch.bool, device=direct.device)
        )
        required_contacts_final = (
            contacts[:, list(required_contact_ids)].all(dim=-1)
            if required_contact_ids
            else torch.ones(count, dtype=torch.bool, device=direct.device)
        )
        forbidden_contacts_final = (
            ~contacts[:, list(forbidden_contact_ids)].any(dim=-1)
            if forbidden_contact_ids
            else torch.ones(count, dtype=torch.bool, device=direct.device)
        )
        r2 = (
            valid
            & final_speed_ok
            & (replay_contact_ratio >= required_contact_ratio)
            & required_contacts_ok
            & forbidden_contacts_ok
            & required_contacts_final
            & forbidden_contacts_final
        )

        selection = slice(start, end)
        r1_pass[selection] = r1_snapshot.detach().cpu().numpy()
        r2_pass[selection] = r2.detach().cpu().numpy()
        max_penetration[selection] = maximum_pen.detach().cpu().numpy()
        max_linear_speed[selection] = maximum_lin.detach().cpu().numpy()
        max_angular_speed[selection] = maximum_ang.detach().cpu().numpy()
        max_drop[selection] = maximum_drop.detach().cpu().numpy()
        max_drift[selection] = maximum_drift.detach().cpu().numpy()
        contact_ratio[selection] = replay_contact_ratio.detach().cpu().numpy()
        final_contact_force[selection] = direct.contact_force_norm[:count].detach().cpu().numpy()
        final_contact_positions[selection] = direct.contact_point_on_pen_w[:count].detach().cpu().numpy()
        final_contact_state[selection] = (
            direct.contact_force_norm[:count] > args_cli.min_contact_force
        ).detach().cpu().numpy()

        if args_cli.progress_every > 0 and batch_index % args_cli.progress_every == 0:
            print(
                "CANONICAL_REPLAY_PROGRESS="
                + json.dumps(
                    {
                        "tag": tag,
                        "batch": batch_index,
                        "completed": end,
                        "total": result_count,
                        "r1_rate": float(r1_pass[:end].mean()),
                        "r2_rate": float(r2_pass[:end].mean()),
                    }
                ),
                flush=True,
            )

    return {
        "row": repeated_rows,
        "r1_pass": r1_pass,
        "r2_pass": r2_pass,
        "max_penetration_m": max_penetration,
        "max_linear_speed_m_s": max_linear_speed,
        "max_angular_speed_rad_s": max_angular_speed,
        "max_drop_m": max_drop,
        "max_drift_m": max_drift,
        "contact_ratio": contact_ratio,
        "final_contact_force": final_contact_force,
        "final_contact_positions": final_contact_positions,
        "final_contact_state": final_contact_state,
    }


def _per_row_rates(
    result: dict[str, np.ndarray], rows: np.ndarray, trials: int, key: str
) -> np.ndarray:
    values = np.asarray(result[key], dtype=np.float32).reshape(len(rows), trials)
    return values.mean(axis=1)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    input_path = (
        _resolve(args_cli.input)
        if args_cli.input
        else DEFAULT_INPUT_DIR / f"{args_cli.canonical_id}_candidate.npz"
    ).resolve()
    output_dir = _resolve(args_cli.output_dir).resolve()
    with np.load(input_path, allow_pickle=False) as loaded:
        full_payload = {key: loaded[key] for key in loaded.files}
    canonical_values = np.unique(full_payload["metadata_canonical_id"])
    if canonical_values.tolist() != [args_cli.canonical_id]:
        raise ValueError(
            f"Expected only {args_cli.canonical_id}, found {canonical_values.tolist()}"
        )
    selected_rows = _select_candidates(full_payload, args_cli.max_candidates)
    payload = subset_payload(full_payload, selected_rows)
    state_types = np.unique(payload["labels_static_or_dynamic"])
    if state_types.size != 1:
        raise ValueError(f"A canonical validator run requires one state type, got {state_types}")
    state_type = str(state_types[0])

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.seed = args_cli.seed
    env_cfg.bundle_path = str(input_path)
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    print("CANONICAL_VALIDATOR_ENV_READY", flush=True)

    joint_lower = direct.joint_lower[0].detach().cpu().numpy()
    joint_upper = direct.joint_upper[0].detach().cpu().numpy()
    r0_full, r0_checks = validate_r0_arrays(
        payload,
        max_source_penetration_m=args_cli.max_penetration_mm / 1000.0,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )
    current_asset_hash = "sha256:" + sha256_file(
        PROJECT_ROOT / "isaac/assets/dex5_hand.usd"
    )
    r0_asset = payload["metadata_asset_hash"] == current_asset_hash
    r0_physics = np.isclose(payload["physics_dt"], float(direct.cfg.sim.dt))
    r0_physics &= payload["physics_solver_position_iterations"] == 12
    r0_physics &= payload["physics_solver_velocity_iterations"] == 2
    expected_com = np.asarray(
        (float(direct.cfg.pen_com_offset_x), 0.0, 0.0), dtype=np.float32
    )
    r0_physics &= np.isclose(
        payload["physics_com_offset"], expected_com[None, :], atol=1.0e-7
    ).all(axis=1)
    r0 = r0_full & r0_asset & r0_physics
    print(
        "CANONICAL_VALIDATOR_R0="
        + json.dumps({"pass": int(r0.sum()), "total": int(r0.size)}),
        flush=True,
    )

    local_rows = np.arange(payload["metadata_state_id"].shape[0], dtype=np.int64)
    source_bank_rows = selected_rows.copy()
    duration_ms = (
        args_cli.static_duration_ms if state_type == "static" else args_cli.dynamic_duration_ms
    )
    continuation_steps = _physics_steps(duration_ms, float(direct.cfg.sim.dt))
    baseline = _run_replays(
        env,
        source_bank_rows,
        trials=1,
        perturbed=False,
        state_type=state_type,
        physics_steps=continuation_steps,
        tag="baseline",
    )
    baseline_r1 = baseline["r1_pass"] & r0
    baseline_r2 = baseline["r2_pass"] & baseline_r1

    r3a = _run_replays(
        env,
        source_bank_rows,
        trials=args_cli.r3a_trials,
        perturbed=False,
        state_type=state_type,
        physics_steps=continuation_steps,
        tag="r3a",
    )
    r3a_rate = _per_row_rates(r3a, local_rows, args_cli.r3a_trials, "r2_pass")
    r3a_pass = r3a_rate >= args_cli.r3a_min_ratio

    r3b = _run_replays(
        env,
        source_bank_rows,
        trials=args_cli.r3b_trials,
        perturbed=True,
        state_type=state_type,
        physics_steps=continuation_steps,
        tag="r3b",
    )
    r3b_r1_rate = _per_row_rates(r3b, local_rows, args_cli.r3b_trials, "r1_pass")
    r3b_r2_rate = _per_row_rates(r3b, local_rows, args_cli.r3b_trials, "r2_pass")
    r3b_pass = (r3b_r1_rate >= args_cli.r3b_r1_min_ratio) & (
        r3b_r2_rate >= args_cli.r3b_r2_min_ratio
    )
    final_pass = r0 & baseline_r1 & baseline_r2 & r3a_pass & r3b_pass

    payload["labels_r0_pass"] = r0
    payload["labels_r1_pass"] = baseline_r1
    payload["labels_r2_pass"] = baseline_r2
    payload["labels_r3a_pass"] = r3a_pass
    payload["labels_r3b_pass"] = r3b_pass
    payload["labels_r3a_success_ratio"] = r3a_rate.astype(np.float32)
    payload["labels_r3b_r1_success_ratio"] = r3b_r1_rate.astype(np.float32)
    payload["labels_r3b_r2_success_ratio"] = r3b_r2_rate.astype(np.float32)
    payload["labels_baseline_max_penetration_m"] = baseline[
        "max_penetration_m"
    ]
    payload["labels_baseline_max_linear_speed_m_s"] = baseline[
        "max_linear_speed_m_s"
    ]
    payload["labels_baseline_max_angular_speed_rad_s"] = baseline[
        "max_angular_speed_rad_s"
    ]
    payload["labels_baseline_max_drop_m"] = baseline["max_drop_m"]
    payload["labels_baseline_max_drift_m"] = baseline["max_drift_m"]
    payload["labels_baseline_contact_ratio"] = baseline["contact_ratio"]
    payload["contact_state"] = baseline["final_contact_state"]
    payload["contact_state_available"] = np.ones(len(local_rows), dtype=np.bool_)
    payload["contact_force"] = baseline["final_contact_force"]
    payload["contact_force_available"] = np.ones(len(local_rows), dtype=np.bool_)
    payload["contact_fingertip_positions"] = baseline["final_contact_positions"]
    payload["contact_positions_available"] = np.ones(len(local_rows), dtype=np.bool_)
    payload["contact_positions_frame"] = np.full(
        len(local_rows), "world", dtype="U5"
    )
    payload["labels_validation_status"] = np.where(
        final_pass, "validated", "rejected"
    ).astype("U10")

    rejection = np.full(len(local_rows), "pass", dtype="U64")
    rejection[~r0] = "r0"
    rejection[r0 & ~baseline_r1] = "r1"
    rejection[r0 & baseline_r1 & ~baseline_r2] = "r2"
    rejection[r0 & baseline_r1 & baseline_r2 & ~r3a_pass] = "r3a"
    rejection[r0 & baseline_r1 & baseline_r2 & r3a_pass & ~r3b_pass] = "r3b"
    payload["labels_rejection_code"] = rejection

    reports_dir = output_dir / "validation_reports"
    banks_dir = output_dir / "canonical_reset_bank"
    audited_path = reports_dir / f"{args_cli.canonical_id}_audited.npz"
    r1_path = banks_dir / f"{args_cli.canonical_id}_r1.npz"
    validated_path = banks_dir / f"{args_cli.canonical_id}_validated.npz"
    save_npz_atomic(audited_path, payload)
    save_npz_atomic(r1_path, subset_payload(payload, np.flatnonzero(baseline_r1)))
    save_npz_atomic(validated_path, subset_payload(payload, np.flatnonzero(final_pass)))

    report = {
        "canonical_id": args_cli.canonical_id,
        "state_type": state_type,
        "status": "validated" if np.any(final_pass) else "no_validated_states",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "candidate_total": int(full_payload["metadata_state_id"].shape[0]),
        "candidate_audited": int(len(local_rows)),
        "duration_ms": duration_ms,
        "physics_steps": continuation_steps,
        "counts": {
            "r0": int(r0.sum()),
            "r1": int(baseline_r1.sum()),
            "r2": int(baseline_r2.sum()),
            "r3a": int((baseline_r2 & r3a_pass).sum()),
            "r3b": int(final_pass.sum()),
        },
        "rates": {
            "r0": float(r0.mean()),
            "r1": float(baseline_r1.mean()),
            "r2": float(baseline_r2.mean()),
            "r3a": float(r3a_pass.mean()),
            "r3b": float(r3b_pass.mean()),
            "final": float(final_pass.mean()),
        },
        "r0_check_failures": {
            key: int((~value).sum()) for key, value in r0_checks.items()
        },
        "r0_asset_failures": int((~r0_asset).sum()),
        "r0_physics_failures": int((~r0_physics).sum()),
        "metrics": {
            "max_penetration_mm": float(
                baseline["max_penetration_m"].max(initial=0.0) * 1000.0
            ),
            "max_linear_speed_m_s": float(
                baseline["max_linear_speed_m_s"].max(initial=0.0)
            ),
            "max_angular_speed_rad_s": float(
                baseline["max_angular_speed_rad_s"].max(initial=0.0)
            ),
            "max_drop_mm": float(baseline["max_drop_m"].max(initial=0.0) * 1000.0),
            "max_drift_mm": float(
                baseline["max_drift_m"].max(initial=0.0) * 1000.0
            ),
            "mean_contact_ratio": float(baseline["contact_ratio"].mean()),
        },
        "thresholds": {
            "max_penetration_mm": args_cli.max_penetration_mm,
            "r3a_min_ratio": args_cli.r3a_min_ratio,
            "r3b_r1_min_ratio": args_cli.r3b_r1_min_ratio,
            "r3b_r2_min_ratio": args_cli.r3b_r2_min_ratio,
        },
        "outputs": {
            "audited": str(audited_path),
            "audited_sha256": sha256_file(audited_path),
            "r1": str(r1_path),
            "r1_sha256": sha256_file(r1_path),
            "validated": str(validated_path),
            "validated_sha256": sha256_file(validated_path),
        },
    }
    report_path = reports_dir / f"{args_cli.canonical_id}_validation_report.json"
    write_json_atomic(report_path, report)
    print("CANONICAL_VALIDATION_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
