from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose oracle support handoff without training.")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--controller", choices=("policy", "zero"), default="policy")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.controller == "policy" and not args.checkpoint:
    parser.error("--checkpoint is required for policy control")
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PRIMARY_TYPES = (
    "ballistic",
    "late_recontact_no_height_recovery",
    "simultaneous_TIM_release",
    "force_collapse_during_overlap",
    "wrench_destabilization",
    "height_loss_without_recovery",
    "stable_support",
    "unclassified",
)


def _debounce(raw: np.ndarray, initial: np.ndarray, steps_required: int = 3) -> np.ndarray:
    state = initial.copy()
    candidate = initial.copy()
    candidate_steps = np.zeros_like(initial, dtype=np.int64)
    output = np.empty_like(raw, dtype=np.bool_)
    for step in range(raw.shape[0]):
        changed = raw[step] != candidate
        candidate = np.where(changed, raw[step], candidate)
        candidate_steps = np.where(changed, 1, candidate_steps + 1)
        commit = (candidate_steps >= steps_required) & (candidate != state)
        state = np.where(commit, candidate, state)
        output[step] = state
    return output


def _first_true(values: np.ndarray) -> int:
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else -1


def _max_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _window_mean(values: np.ndarray, event_step: int, before: bool) -> float:
    if event_step < 0:
        return float("nan")
    start = max(0, event_step - 3) if before else event_step
    stop = event_step if before else min(values.shape[0], event_step + 3)
    if stop <= start:
        return float("nan")
    return float(np.mean(values[start:stop]))


def _finite_quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {"mean": None, "p10": None, "median": None, "p90": None}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def _analyze(
    force_norm: np.ndarray,
    force_w: np.ndarray,
    fingertip_pos: np.ndarray,
    pen_pos: np.ndarray,
    pen_linvel: np.ndarray,
    pen_angvel: np.ndarray,
    initial_contacts: np.ndarray,
    initial_pen_pos: np.ndarray,
    bank_ids: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    raw_contact = force_norm > 0.03
    contact = _debounce(raw_contact, initial_contacts, steps_required=3)
    height_delta = pen_pos[:, :, 2] - initial_pen_pos[None, :, 2]
    tim_contact = contact[:, :, :3]
    tim_count = tim_contact.sum(axis=-1)
    tim_force = (force_norm[:, :, :3] * tim_contact).sum(axis=-1)

    force_on_pen = -force_w
    lever = fingertip_pos - pen_pos[:, :, None, :]
    net_force = force_on_pen.sum(axis=2)
    net_torque = np.cross(lever, force_on_pen).sum(axis=2)
    force_magnitude = np.linalg.norm(net_force, axis=-1)
    lateral_force = np.linalg.norm(net_force[:, :, :2], axis=-1)
    torque_magnitude = np.linalg.norm(net_torque, axis=-1)
    target_speed = pen_angvel[:, :, 2]
    wrong_axis_speed = np.linalg.norm(pen_angvel[:, :, :2], axis=-1)
    linear_speed = np.linalg.norm(pen_linvel, axis=-1)
    angular_speed = np.linalg.norm(pen_angvel, axis=-1)

    episodes = []
    for env_index in range(force_norm.shape[1]):
        first_break = []
        first_recontact = []
        for finger_index in range(len(FINGERS)):
            finger_contact = contact[:, env_index, finger_index]
            transitions = np.diff(
                np.concatenate(([initial_contacts[env_index, finger_index]], finger_contact))
                .astype(np.int8)
            )
            break_step = _first_true(transitions < 0)
            recontact_step = -1
            if break_step >= 0:
                later = np.flatnonzero((transitions > 0) & (np.arange(transitions.size) > break_step))
                if later.size:
                    recontact_step = int(later[0])
            first_break.append(break_step)
            first_recontact.append(recontact_step)

        first_drop = _first_true(height_delta[:, env_index] <= -0.005)
        recovered = False
        if first_drop >= 0:
            recovered = bool(np.any(height_delta[first_drop + 1 :, env_index] >= -0.002))
        zero_tim = tim_count[:, env_index] == 0
        max_zero_run = _max_true_run(zero_tim)
        ballistic = bool(
            np.any(
                (zero_tim)
                & ((linear_speed[:, env_index] > 0.20) | (angular_speed[:, env_index] > 12.0))
            )
        )
        late_recontact = bool(
            first_drop >= 0
            and any(step > first_drop for step in first_recontact if step >= 0)
            and not recovered
        )
        simultaneous_release = max_zero_run >= 2
        pre_force = _window_mean(tim_force[:, env_index], first_drop, before=True)
        post_force = _window_mean(tim_force[:, env_index], first_drop, before=False)
        pre_torque = _window_mean(torque_magnitude[:, env_index], first_drop, before=True)
        post_torque = _window_mean(torque_magnitude[:, env_index], first_drop, before=False)
        force_collapse = bool(
            first_drop >= 0
            and np.isfinite(pre_force)
            and pre_force > 0.03
            and np.isfinite(post_force)
            and post_force < 0.5 * pre_force
            and np.any(tim_count[max(0, first_drop - 3) : first_drop + 1, env_index] >= 2)
        )
        wrench_destabilization = bool(
            first_drop >= 0
            and np.isfinite(pre_torque)
            and np.isfinite(post_torque)
            and post_torque > max(2.0 * pre_torque, pre_torque + 1.0e-5)
        )
        height_loss = first_drop >= 0 and not recovered
        stable_support = first_drop < 0 and max_zero_run < 2
        labels = {
            "ballistic": ballistic,
            "late_recontact_no_height_recovery": late_recontact,
            "simultaneous_TIM_release": simultaneous_release,
            "force_collapse_during_overlap": force_collapse,
            "wrench_destabilization": wrench_destabilization,
            "height_loss_without_recovery": height_loss,
            "stable_support": stable_support,
        }
        primary = next((name for name in PRIMARY_TYPES[:-1] if labels.get(name, False)), "unclassified")
        episodes.append(
            {
                "env_id": env_index,
                "bank_id": int(bank_ids[env_index]),
                "first_height_drop_step": first_drop,
                "height_recovered": recovered,
                "minimum_height_delta_mm": float(1000.0 * np.min(height_delta[:, env_index])),
                "max_zero_TIM_run_steps": max_zero_run,
                "TIM_overlap_ratio": float(np.mean(tim_count[:, env_index] >= 2)),
                "first_break_step": dict(zip(FINGERS, first_break, strict=True)),
                "first_recontact_step": dict(zip(FINGERS, first_recontact, strict=True)),
                "pre_drop_TIM_force_n": pre_force,
                "post_drop_TIM_force_n": post_force,
                "pre_drop_force_magnitude_n": _window_mean(
                    force_magnitude[:, env_index], first_drop, before=True
                ),
                "post_drop_force_magnitude_n": _window_mean(
                    force_magnitude[:, env_index], first_drop, before=False
                ),
                "pre_drop_vertical_force_n": _window_mean(
                    net_force[:, env_index, 2], first_drop, before=True
                ),
                "post_drop_vertical_force_n": _window_mean(
                    net_force[:, env_index, 2], first_drop, before=False
                ),
                "pre_drop_lateral_force_n": _window_mean(
                    lateral_force[:, env_index], first_drop, before=True
                ),
                "post_drop_lateral_force_n": _window_mean(
                    lateral_force[:, env_index], first_drop, before=False
                ),
                "pre_drop_torque_magnitude_nm": pre_torque,
                "post_drop_torque_magnitude_nm": post_torque,
                "pre_drop_target_speed_rad_s": _window_mean(
                    target_speed[:, env_index], first_drop, before=True
                ),
                "post_drop_target_speed_rad_s": _window_mean(
                    target_speed[:, env_index], first_drop, before=False
                ),
                "pre_drop_wrong_axis_speed_rad_s": _window_mean(
                    wrong_axis_speed[:, env_index], first_drop, before=True
                ),
                "post_drop_wrong_axis_speed_rad_s": _window_mean(
                    wrong_axis_speed[:, env_index], first_drop, before=False
                ),
                "labels": labels,
                "primary_failure": primary,
                "finger_force_around_drop_n": {
                    finger: {
                        "pre": _window_mean(
                            force_norm[:, env_index, finger_index],
                            first_drop,
                            before=True,
                        ),
                        "post": _window_mean(
                            force_norm[:, env_index, finger_index],
                            first_drop,
                            before=False,
                        ),
                    }
                    for finger_index, finger in enumerate(FINGERS)
                },
            }
        )

    primary_counts = {name: 0 for name in PRIMARY_TYPES}
    label_counts = {name: 0 for name in PRIMARY_TYPES[:-1]}
    for episode in episodes:
        primary_counts[episode["primary_failure"]] += 1
        for name, value in episode["labels"].items():
            label_counts[name] += int(value)

    def values(key: str) -> list[float]:
        return [float(episode[key]) for episode in episodes]

    summary = {
        "episodes": len(episodes),
        "unique_bank_rows": int(np.unique(bank_ids).size),
        "primary_failure_counts": primary_counts,
        "label_counts": label_counts,
        "rates": {
            name: count / max(len(episodes), 1) for name, count in label_counts.items()
        },
        "minimum_height_delta_mm": _finite_quantiles(values("minimum_height_delta_mm")),
        "max_zero_TIM_run_steps": _finite_quantiles(values("max_zero_TIM_run_steps")),
        "TIM_overlap_ratio": _finite_quantiles(values("TIM_overlap_ratio")),
        "wrench_around_first_5mm_drop": {
            key: _finite_quantiles(values(key))
            for key in (
                "pre_drop_TIM_force_n",
                "post_drop_TIM_force_n",
                "pre_drop_force_magnitude_n",
                "post_drop_force_magnitude_n",
                "pre_drop_vertical_force_n",
                "post_drop_vertical_force_n",
                "pre_drop_lateral_force_n",
                "post_drop_lateral_force_n",
                "pre_drop_torque_magnitude_nm",
                "post_drop_torque_magnitude_nm",
                "pre_drop_target_speed_rad_s",
                "post_drop_target_speed_rad_s",
                "pre_drop_wrong_axis_speed_rad_s",
                "post_drop_wrong_axis_speed_rad_s",
            )
        },
        "finger_transition_rates": {
            finger: {
                "break": sum(
                    episode["first_break_step"][finger] >= 0 for episode in episodes
                )
                / max(len(episodes), 1),
                "recontact": sum(
                    episode["first_recontact_step"][finger] >= 0 for episode in episodes
                )
                / max(len(episodes), 1),
            }
            for finger in FINGERS
        },
        "finger_force_around_first_5mm_drop_n": {
            finger: {
                side: _finite_quantiles(
                    [
                        episode["finger_force_around_drop_n"][finger][side]
                        for episode in episodes
                    ]
                )
                for side in ("pre", "post")
            }
            for finger in FINGERS
        },
        "time_series_stride_10": [
            {
                "step": int(step),
                "time_s": float(step / 60.0),
                "median_height_delta_mm": float(
                    1000.0 * np.median(height_delta[step])
                ),
                "median_TIM_contact_count": float(np.median(tim_count[step])),
                "median_vertical_force_n": float(np.median(net_force[step, :, 2])),
                "median_lateral_force_n": float(np.median(lateral_force[step])),
                "median_torque_magnitude_nm": float(
                    np.median(torque_magnitude[step])
                ),
                "median_finger_force_n": {
                    finger: float(np.median(force_norm[step, :, finger_index]))
                    for finger_index, finger in enumerate(FINGERS)
                },
            }
            for step in sorted(set(range(0, force_norm.shape[0], 10)) | {force_norm.shape[0] - 1})
        ],
    }
    traces = {
        "bank_ids": bank_ids,
        "initial_pen_pos_m": initial_pen_pos,
        "contact_force_norm_n": force_norm,
        "raw_contact": raw_contact,
        "debounced_contact": contact,
        "height_delta_m": height_delta,
        "net_force_proxy_n": net_force,
        "net_torque_proxy_nm": net_torque,
        "pen_linear_velocity_m_s": pen_linvel,
        "pen_angular_velocity_rad_s": pen_angvel,
    }
    return {"summary": summary, "episodes": episodes}, traces


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = 20260730
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = (args.steps + 10) * control_dt
    env_cfg.penetration_threshold = 1.0
    env_cfg.drop_threshold = 10.0
    env_cfg.drift_threshold = 10.0
    env_cfg.no_contact_termination_steps = args.steps + 10
    env_cfg.throw_linear_speed_m_s = 1000.0
    env_cfg.throw_angular_speed_rad_s = 1000.0
    env_cfg.terminate_on_ballistic = False
    experiment_cfg["seed"] = 20260730
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["trainer"]["close_environment_at_exit"] = False

    raw_env = gym.make(args.task, cfg=env_cfg)
    direct = raw_env.unwrapped
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = None
    if args.controller == "policy":
        runner = Runner(env, experiment_cfg)
        runner.agent.load(str(Path(args.checkpoint).resolve()))
        runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    states = env.state()
    bank_ids = direct.current_bank_ids.detach().cpu().numpy().copy()
    initial_contacts = direct.source_contact_state_bank[
        direct.current_bank_ids
    ].detach().cpu().numpy().copy()
    initial_pen_pos = direct.pen_pos.detach().cpu().numpy().copy()
    recorded: dict[str, list[np.ndarray]] = {
        "action": [],
        "hand_qpos": [],
        "hand_qvel": [],
        "processed_targets": [],
        "force_norm": [],
        "force_w": [],
        "fingertip_pos": [],
        "pen_pos": [],
        "pen_linvel": [],
        "pen_angvel": [],
    }
    for _ in range(args.steps):
        with torch.inference_mode():
            if runner is None:
                actions = torch.zeros(
                    (args.num_envs, direct.cfg.action_space), device=direct.device
                )
            else:
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
            states = env.state()
        if bool(torch.any(terminated | truncated)):
            raise RuntimeError("Handoff diagnostic unexpectedly reset an episode")
        for key, tensor in (
            ("action", actions),
            ("hand_qpos", direct.hand_qpos),
            ("hand_qvel", direct.hand_qvel),
            ("processed_targets", direct.processed_targets),
            ("force_norm", direct.contact_force_norm),
            ("force_w", direct.contact_force_w),
            ("fingertip_pos", direct.fingertip_pos_local),
            ("pen_pos", direct.pen_pos),
            ("pen_linvel", direct.pen_linvel),
            ("pen_angvel", direct.pen_angvel),
        ):
            recorded[key].append(tensor.detach().cpu().numpy().copy())

    arrays = {key: np.stack(value, axis=0) for key, value in recorded.items()}
    analysis, traces = _analyze(
        arrays["force_norm"],
        arrays["force_w"],
        arrays["fingertip_pos"],
        arrays["pen_pos"],
        arrays["pen_linvel"],
        arrays["pen_angvel"],
        initial_contacts,
        initial_pen_pos,
        bank_ids,
    )
    report = {
        "status": "PASS",
        "artifact_role": "diagnostic_only_not_for_training_or_reset",
        "task": args.task,
        "controller": args.controller,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "seed": 20260730,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "control_dt_s": control_dt,
        "contact_debounce_steps": 3,
        "height_drop_threshold_m": 0.005,
        "height_recovery_threshold_m": 0.002,
        "bank_sha256": direct.cfg.expected_bundle_sha256,
        **analysis,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        artifact_role=np.asarray("diagnostic_only_not_for_training_or_reset"),
        task=np.asarray(args.task),
        controller=np.asarray(args.controller),
        checkpoint=np.asarray(report["checkpoint"] or ""),
        bank_sha256=np.asarray(report["bank_sha256"]),
        initial_contacts=initial_contacts,
        action=arrays["action"],
        hand_qpos=arrays["hand_qpos"],
        hand_qvel=arrays["hand_qvel"],
        processed_targets=arrays["processed_targets"],
        **traces,
    )
    print("HANDOFF_DIAGNOSTIC=" + json.dumps(report["summary"]), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
