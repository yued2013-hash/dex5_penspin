from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run v9 P3-to-braked-P4 policies continuously without intermediate resets."
)
parser.add_argument("--output", required=True)
parser.add_argument("--report", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_trajectories", type=int, default=200)
parser.add_argument("--max_batches", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--single_checkpoint", default=None)
parser.add_argument(
    "--final_checkpoint",
    default=None,
    help="Override only the final expert stage checkpoint for continuous distribution tests.",
)
parser.add_argument(
    "--stage_checkpoint",
    action="append",
    default=[],
    metavar="STAGE=PATH",
    help="Override a named stage checkpoint; may be specified more than once.",
)
parser.add_argument(
    "--stage_max_steps",
    action="append",
    default=[],
    metavar="STAGE=N",
    help="Override a stage timeout for timing audits; success gates are unchanged.",
)
parser.add_argument("--dagger_output", default=None)
parser.add_argument("--dagger_stride", type=int, default=1)
parser.add_argument(
    "--reached_stage_output",
    default=None,
    help=(
        "Optionally export every trajectory that reached --reached_stage_name, "
        "including final failures, for hard-entry curriculum construction."
    ),
)
parser.add_argument(
    "--reached_stage_name",
    default="approach49_to_braked50",
)
parser.add_argument("--dual_entry_eval", action="store_true")
parser.add_argument("--latest_compact", action="store_true")
parser.add_argument(
    "--upstream_approach49",
    action="store_true",
    help="Insert the current reset-free upstream-to-49 bridge before final braking.",
)
parser.add_argument(
    "--reset_seed_stride",
    type=int,
    default=1_000_003,
    help="Prime stride used to create an independent reset-bank RNG stream per batch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401, E402
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface"
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", ROOT / ".deps/IsaacLab"))
LOG_DIR = Path(
    os.environ.get(
        "ISAACLAB_LOG_ROOT", ISAACLAB_ROOT / "logs/skrl"
    )
) / "dex5_recapture_p3_physics_direct"


@dataclass(frozen=True)
class Stage:
    name: str
    bundle: str
    checkpoint: str
    target_deg: float
    min_deg: float
    max_deg: float
    hold_steps: int = 6
    max_steps: int = 55
    action_scale: float = 0.105
    min_thumb_force_n: float = 0.03
    max_hand_rmse: float = 0.22
    keep_current_pen_target: bool = False
    use_teacher_actions: bool = False
    teacher_prefill_steps: int = 0
    policy_horizon_steps: int = 24


STAGES = (
    Stage(
        "P3_to_7",
        "v9_p3_to_entry7_physx_bundle.npz",
        "2026-07-16_19-15-06_ppo_torch/checkpoints/best_agent.pt",
        7.0,
        5.0,
        10.0,
    ),
    Stage(
        "7_to_20",
        "v9_entry7_to_approach20_physx_bundle.npz",
        "2026-07-16_19-18-50_ppo_torch/checkpoints/best_agent.pt",
        20.0,
        17.0,
        23.0,
    ),
    Stage(
        "20_to_25",
        "v9_approach20_to_entry25_physx_bundle.npz",
        "2026-07-16_19-27-56_ppo_torch/checkpoints/best_agent.pt",
        25.0,
        23.0,
        28.0,
    ),
    Stage(
        "25_to_30",
        "v9_continuous_p3_p4_cascade1024_v16_new_brake_report_25_to_30_entry_bundle.npz",
        "2026-07-17_00-17-01_ppo_torch/checkpoints/agent_7680.pt",
        30.0,
        28.0,
        33.0,
    ),
    Stage(
        "30_to_35",
        "v9_mid30_to_mid35_physx_bundle.npz",
        "2026-07-16_19-51-28_ppo_torch/checkpoints/best_agent.pt",
        35.0,
        33.0,
        38.0,
    ),
    Stage(
        "35_to_40",
        "v9_mid35_to_approach40_physx_bundle.npz",
        "2026-07-16_20-13-22_ppo_torch/checkpoints/agent_3200.pt",
        40.0,
        38.0,
        43.0,
    ),
    Stage(
        "40_to_42",
        "v9_continuous_p3_p4_cascade512_report_40_to_42_entry_bundle.npz",
        "2026-07-16_22-26-11_ppo_torch/checkpoints/agent_7680.pt",
        42.0,
        41.0,
        44.0,
        action_scale=0.06,
    ),
    Stage(
        "42_to_45_teacher",
        "v9_cascade42_to_entry45_teacher_bundle.npz",
        "2026-07-16_22-34-28_ppo_torch/checkpoints/best_agent.pt",
        45.0,
        42.5,
        45.0,
        hold_steps=3,
        action_scale=0.06,
        use_teacher_actions=True,
    ),
    Stage(
        "thumb_support_correction",
        "v9_continuous_p3_p4_cascade1024_v17_mid30_brake_report_thumb_support_correction_entry_bundle.npz",
        "2026-07-17_00-27-34_ppo_torch/checkpoints/agent_2560.pt",
        44.0,
        42.0,
        47.0,
        action_scale=0.05,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.04,
        keep_current_pen_target=True,
    ),
    Stage(
        "supported_to_43",
        "v9_continuous_p3_p4_cascade512_v7_local_frames_report_supported_to_45_entry_bundle.npz",
        "2026-07-16_23-06-54_ppo_torch/checkpoints/agent_3840.pt",
        45.0,
        43.0,
        46.0,
        hold_steps=3,
        action_scale=0.06,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.08,
    ),
    Stage(
        "supported_to_435",
        "v9_continuous_p3_p4_cascade1024_v18_thumb_lowforce_report_supported_to_435_entry_bundle.npz",
        "2026-07-17_00-31-29_ppo_torch/checkpoints/agent_3840.pt",
        45.0,
        43.5,
        46.5,
        hold_steps=3,
        action_scale=0.06,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.08,
    ),
    Stage(
        "supported_to_45",
        "v9_continuous_p3_p4_cascade1024_v19_supported435_report_supported_to_45_entry_bundle.npz",
        "2026-07-17_00-36-01_ppo_torch/checkpoints/agent_5120.pt",
        45.0,
        44.0,
        47.0,
        action_scale=0.06,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.08,
    ),
    Stage(
        "45_to_47",
        "v9_continuous_p3_p4_cascade512_v12_report_45_to_47_entry_bundle.npz",
        "2026-07-16_23-24-06_ppo_torch/checkpoints/agent_5120.pt",
        47.0,
        45.0,
        49.0,
        action_scale=0.06,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.08,
    ),
    Stage(
        "47_capture",
        "v9_continuous_p3_p4_cascade512_v13_report_47_capture_entry_bundle.npz",
        "2026-07-16_23-28-40_ppo_torch/checkpoints/agent_3840.pt",
        48.0,
        47.0,
        50.0,
        action_scale=0.06,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.08,
    ),
    Stage(
        "braked_P4",
        "v9_continuous_p3_p4_cascade1024_v20_supported45_report_braked_P4_entry_bundle.npz",
        "2026-07-17_00-40-16_ppo_torch/checkpoints/best_agent.pt",
        50.0,
        50.0,
        60.0,
        hold_steps=18,
        max_steps=60,
        action_scale=0.06,
        min_thumb_force_n=0.12,
        max_hand_rmse=0.08,
    ),
)

LATEST_COMPACT_STAGES = (
    Stage("P3_to_7", "v9_p3_strict_to_entry7_physx_bundle.npz", str(REPORT_DIR / "v9_entry7_stable_agent6336.pt"), 7.0, 5.0, 10.0, max_steps=30, action_scale=0.09, teacher_prefill_steps=2, policy_horizon_steps=15),
    Stage("7_to_20", "v9_entry7_to_approach20_physx_bundle.npz", str(REPORT_DIR / "v9_approach20_stable_agent9600.pt"), 20.0, 17.0, 23.0, max_steps=40, action_scale=0.09, teacher_prefill_steps=7),
    Stage("20_to_25", "v9_approach20_to_entry25_physx_bundle.npz", str(REPORT_DIR / "v9_entry25_continuous_agent7680.pt"), 25.0, 23.0, 28.0, max_steps=35, action_scale=0.09),
    Stage("25_to_30", "v9_entry25_to_mid30_paired109_physx_bundle.npz", str(REPORT_DIR / "v9_mid30_continuous_static16_agent7680.pt"), 30.0, 28.0, 33.0, max_steps=35, action_scale=0.09, teacher_prefill_steps=8),
    Stage("30_to_35", "v9_mid30_to_mid35_physx_bundle.npz", str(REPORT_DIR / "v9_mid35_stable_agent5760.pt"), 35.0, 33.0, 38.0, max_steps=30, action_scale=0.09),
    Stage("35_to_40", "v9_mid35_to_approach40_physx_bundle.npz", str(REPORT_DIR / "v9_approach40_continuous_agent2560.pt"), 40.0, 38.0, 43.0, max_steps=30, action_scale=0.09),
    Stage("40_to_45", "v9_approach40_to_entry45_lowpen136_physx_bundle.npz", str(REPORT_DIR / "v9_entry45_continuous_lowpen325_agent5120.pt"), 45.0, 43.0, 48.0, max_steps=30, action_scale=0.09),
    Stage("45_to_50", "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz", str(REPORT_DIR / "v9_braked50_stable_best_agent.pt"), 50.0, 50.0, 60.0, hold_steps=18, max_steps=45, action_scale=0.06, policy_horizon_steps=30),
)

UPSTREAM_APPROACH49_STAGE = Stage(
    "upstream_to_approach49",
    "v9_current_upstream45_to_approach49_bundle.npz",
    str(LOG_DIR / "2026-07-17_07-45-27_ppo_torch/checkpoints/agent_5376.pt"),
    49.0,
    48.0,
    52.0,
    hold_steps=6,
    max_steps=45,
    action_scale=0.09,
    policy_horizon_steps=30,
)

UPSTREAM_BRAKED50_STAGE = Stage(
    "approach49_to_braked50",
    "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz",
    str(REPORT_DIR / "v9_braked50_hard3x_agent5120.pt"),
    50.0,
    50.0,
    60.0,
    hold_steps=18,
    max_steps=75,
    action_scale=0.06,
    policy_horizon_steps=30,
)

if args_cli.latest_compact:
    STAGES = LATEST_COMPACT_STAGES
if args_cli.upstream_approach49:
    if not args_cli.latest_compact:
        parser.error("--upstream_approach49 requires --latest_compact")
    STAGES = (*LATEST_COMPACT_STAGES[:-1], UPSTREAM_APPROACH49_STAGE, UPSTREAM_BRAKED50_STAGE)
if args_cli.final_checkpoint:
    STAGES = (*STAGES[:-1], replace(STAGES[-1], checkpoint=args_cli.final_checkpoint))
for checkpoint_override in args_cli.stage_checkpoint:
    if "=" not in checkpoint_override:
        parser.error("--stage_checkpoint must use STAGE=PATH")
    stage_name, checkpoint_path = checkpoint_override.split("=", 1)
    matches = [index for index, stage in enumerate(STAGES) if stage.name == stage_name]
    if not matches:
        parser.error(f"Unknown stage in --stage_checkpoint: {stage_name}")
    stage_index = matches[0]
    stages = list(STAGES)
    stages[stage_index] = replace(stages[stage_index], checkpoint=checkpoint_path)
    STAGES = tuple(stages)
for max_steps_override in args_cli.stage_max_steps:
    if "=" not in max_steps_override:
        parser.error("--stage_max_steps must use STAGE=N")
    stage_name, max_steps_text = max_steps_override.split("=", 1)
    matches = [index for index, stage in enumerate(STAGES) if stage.name == stage_name]
    if not matches:
        parser.error(f"Unknown stage in --stage_max_steps: {stage_name}")
    try:
        max_steps = int(max_steps_text)
    except ValueError:
        parser.error(f"Invalid max steps for {stage_name}: {max_steps_text}")
    if max_steps < 1:
        parser.error(f"Max steps must be positive for {stage_name}")
    stage_index = matches[0]
    stages = list(STAGES)
    stages[stage_index] = replace(stages[stage_index], max_steps=max_steps)
    STAGES = tuple(stages)


def yaw_rotate(quat: torch.Tensor, degrees: float | torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(degrees, dtype=quat.dtype, device=quat.device)
    angle = angle.expand(len(quat))
    half = torch.deg2rad(angle) * 0.5
    yaw = torch.zeros_like(quat)
    yaw[:, 0] = torch.cos(half)
    yaw[:, 3] = torch.sin(half)
    result = quat_mul(yaw, quat)
    return result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp_min(1.0e-6)


def load_bundle(stage: Stage, device: str) -> dict[str, torch.Tensor]:
    with np.load(REPORT_DIR / stage.bundle) as data:
        result = {
            "source_qpos": torch.as_tensor(data["source_qpos"], device=device),
            "source_target": torch.as_tensor(data["source_target_qpos"], device=device),
            "source_action": torch.as_tensor(data["source_policy_action"], device=device),
            "target_qpos": torch.as_tensor(data["target_qpos"], device=device),
            "source_phase_deg": torch.rad2deg(
                torch.as_tensor(data["source_phase_offset_rad"], device=device)
            ),
        }
        if stage.use_teacher_actions or stage.teacher_prefill_steps > 0:
            result["teacher_actions"] = torch.as_tensor(
                data["source_teacher_actions"], device=device
            )
        return result


def nearest_rows(current_hand: torch.Tensor, source_hand: torch.Tensor) -> torch.Tensor:
    result = []
    for chunk in current_hand.split(128):
        distance = torch.mean(
            (chunk[:, None, :] - source_hand[None, :, :]) ** 2, dim=-1
        )
        result.append(torch.argmin(distance, dim=1))
    return torch.cat(result)


@hydra_task_config("Dex5-V9P3-Entry7-Direct-v0", "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.dual_entry_eval:
        env_cfg.bundle_path = str(
            REPORT_DIR / "v9_dual_entry0_45_to_braked50_physx_bundle.npz"
        )
        env_cfg.reset_joint_pos_noise = 0.0
        env_cfg.reset_joint_vel_noise = 0.0
        env_cfg.reset_pen_pos_noise = 0.0
        env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.episode_length_s = 64.0 * env_cfg.sim.dt * env_cfg.decimation
    agent_cfg["seed"] = args_cli.seed
    env = gym.make("Dex5-V9P3-Entry7-Direct-v0", cfg=env_cfg)
    direct_env = env.unwrapped
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped_env, agent_cfg)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    device = direct_env.device

    policies = []
    preprocessors = []
    bundles = []
    for stage in STAGES:
        checkpoint_path = Path(stage.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = LOG_DIR / checkpoint_path
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy = copy.deepcopy(runner.agent.policy)
        policy.load_state_dict(payload["policy"])
        policy.eval()
        preprocessor = copy.deepcopy(runner.agent._observation_preprocessor)
        preprocessor.load_state_dict(payload["observation_preprocessor"])
        preprocessor.eval()
        policies.append(policy)
        preprocessors.append(preprocessor)
        bundles.append(load_bundle(stage, device))
    student_policy = None
    student_preprocessor = None
    if args_cli.single_checkpoint:
        student_checkpoint = Path(args_cli.single_checkpoint)
        if not student_checkpoint.is_absolute():
            student_checkpoint = LOG_DIR / student_checkpoint
        student_payload = torch.load(
            student_checkpoint, map_location=device, weights_only=False
        )
        student_policy = copy.deepcopy(runner.agent.policy)
        student_policy.load_state_dict(student_payload["policy"])
        student_policy.eval()
        student_preprocessor = copy.deepcopy(runner.agent._observation_preprocessor)
        student_preprocessor.load_state_dict(
            student_payload["observation_preprocessor"]
        )
        student_preprocessor.eval()
    # Use the same frozen, stateless policy/preprocessor path for every stage.
    # Reusing runner.agent only for stage 0 leaked agent/preprocessor state across
    # collection batches and made identical fresh-process seeds diverge sharply.
    use_native_stage0_agent = False

    # Keep normalized actions in observations while applying stage-specific scales internally.
    direct_env.action_scale_env = torch.full(
        (args_cli.num_envs,), STAGES[0].action_scale, device=device
    )
    direct_env.policy_horizon_env = torch.full(
        (args_cli.num_envs,), STAGES[0].policy_horizon_steps, device=device
    )
    # A single DirectRLEnv retains the stage-0 config throughout the cascade.
    # Inject each stage's prefill explicitly below so the P3 teacher bank cannot
    # overwrite later-stage actions whenever their local clock is reset.
    direct_env.cfg.teacher_prefill_steps = 0
    direct_env.cfg.max_success_penetration_m = 0.0005
    trajectories: list[dict[str, np.ndarray]] = []
    reached_stage_trajectories: list[dict[str, np.ndarray]] = []
    stage_entries = np.zeros(len(STAGES), dtype=np.int64)
    stage_successes = np.zeros(len(STAGES), dtype=np.int64)
    stage_diagnostics = [
        {
            "samples": 0,
            "phase_sum": 0.0,
            "phase_min": float("inf"),
            "phase_max": float("-inf"),
            "angular_speed_sum": 0.0,
            "triad_contact_samples": 0,
            "hand_gate_samples": 0,
            "drift_gate_samples": 0,
            "penetration_gate_samples": 0,
            "instantaneous_gate_samples": 0,
            "max_hold_steps": 0,
        }
        for _ in STAGES
    ]
    batch_full_success: list[int] = []
    entrance_entries = {"0": 0, "45": 0}
    entrance_successes = {"0": 0, "45": 0}
    reset_batch_audits: list[dict[str, object]] = []
    transition_caches: list[list[dict[str, np.ndarray]]] = [
        [] for _ in STAGES
    ]
    stage0_step_trace: list[dict[str, float | int]] = []
    final_stage_step_trace: list[dict[str, float | int]] = []
    dagger_observations: list[np.ndarray] = []
    dagger_expert_actions: list[np.ndarray] = []
    dagger_student_actions: list[np.ndarray] = []
    dagger_stage_ids: list[np.ndarray] = []
    reached_stage_id = next(
        (
            index
            for index, stage in enumerate(STAGES)
            if stage.name == args_cli.reached_stage_name
        ),
        None,
    )
    if args_cli.reached_stage_output and reached_stage_id is None:
        parser.error(f"Unknown --reached_stage_name: {args_cli.reached_stage_name}")

    for batch in range(args_cli.max_batches):
        if batch > 0:
            # DirectRLEnv state writes do not clear PhysX contact warm-start data.
            # A simulation reset makes each collection shard equivalent to a
            # fresh process before the task reset rewrites hand and pen state.
            direct_env.sim.reset()
        # Runner/model setup consumes RNG state. Reseed immediately before reset so
        # reset-bank sampling is reproducible and independent across collection batches.
        reset_seed = args_cli.seed + batch * args_cli.reset_seed_stride
        torch.manual_seed(reset_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(reset_seed)
        np.random.seed(reset_seed % (2**32 - 1))
        # skrl's IsaacLab wrapper only forwards its first reset. Runner construction
        # consumes that reset, so call DirectRLEnv here to actually resample every batch.
        reset_observations, _ = env.reset(seed=reset_seed)
        observations = reset_observations["policy"]
        direct_env._compute_intermediate_values()
        initial_quat = direct_env.pen_quat.clone()
        initial_axis = direct_env.pen_axis.clone()
        initial_pos = direct_env.pen_pos.clone()
        initial_qpos = torch.cat(
            (direct_env.hand_qpos, direct_env.pen_pos, direct_env.pen_quat), dim=-1
        ).cpu().numpy()
        initial_qvel = torch.cat(
            (direct_env.hand_qvel, direct_env.pen_linvel, direct_env.pen_angvel), dim=-1
        ).cpu().numpy()
        initial_bank_ids = direct_env.current_bank_ids.cpu().numpy().copy()
        rounded_initial_qpos = np.round(initial_qpos, decimals=6)
        reset_batch_audits.append(
            {
                "batch": batch + 1,
                "reset_seed": reset_seed,
                "unique_bank_ids": int(np.unique(initial_bank_ids).size),
                "unique_initial_states_at_1e_6": int(
                    np.unique(rounded_initial_qpos, axis=0).shape[0]
                ),
                "bank_id_min": int(initial_bank_ids.min()),
                "bank_id_max": int(initial_bank_ids.max()),
            }
        )
        stage_ids = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        if args_cli.dual_entry_eval:
            late_entry = direct_env.curriculum_stage_deg >= 40.0
            late_stage = len(STAGES) - 1 if args_cli.latest_compact else 12
            stage_ids[late_entry] = late_stage
            if torch.any(late_entry):
                late_rows = torch.div(
                    direct_env.current_bank_ids[late_entry] - 1315,
                    7,
                    rounding_mode="floor",
                )
                late_phase = bundles[-1]["source_phase_deg"][late_rows]
                initial_quat[late_entry] = yaw_rotate(
                    direct_env.pen_quat[late_entry], -late_phase
                )
                initial_axis[late_entry] = quat_apply(
                    initial_quat[late_entry], direct_env.pen_x_axis[late_entry]
                )
        else:
            late_entry = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        initial_entry_labels = torch.where(late_entry, 45, 0).cpu().numpy()
        entrance_entries["0"] += int((~late_entry).sum().item())
        entrance_entries["45"] += int(late_entry.sum().item())
        stage_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        stage_bundle_rows = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        hold_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
        final_success = np.zeros(args_cli.num_envs, dtype=np.bool_)
        histories = [
            {
                "qpos": [],
                "qvel": [],
                "action": [],
                "processed_target": [],
                "contact_force_n": [],
                "phase_deg": [],
                "angular_speed_rad_s": [],
                "linear_speed_m_s": [],
                "drift_m": [],
                "global_drift_m": [],
                "hand_rmse_rad": [],
                "max_penetration_m": [],
                "stage_id": [],
                "observation": [],
            }
            for _ in range(args_cli.num_envs)
        ]

        def configure(
            env_ids: torch.Tensor,
            stage_id: int,
            exact_bundle_rows: torch.Tensor | None = None,
        ) -> None:
            if len(env_ids) == 0:
                return
            stage = STAGES[stage_id]
            bank = bundles[stage_id]
            direct_env._compute_intermediate_values()
            entry_source_qpos = torch.cat(
                (direct_env.hand_qpos[env_ids], direct_env.pen_pos[env_ids], direct_env.pen_quat[env_ids]),
                dim=-1,
            ).clone()
            entry_source_qvel = torch.cat(
                (direct_env.hand_qvel[env_ids], direct_env.pen_linvel[env_ids], direct_env.pen_angvel[env_ids]),
                dim=-1,
            ).clone()
            entry_actuator_target = direct_env.processed_targets[env_ids].clone()
            entry_action = direct_env.actions[env_ids].clone()
            nearest = (
                exact_bundle_rows
                if exact_bundle_rows is not None
                else nearest_rows(
                    direct_env.hand_qpos[env_ids], bank["source_qpos"][:, :20]
                )
            )
            stage_bundle_rows[env_ids] = nearest
            source_qpos = bank["source_qpos"][nearest]
            target_qpos = bank["target_qpos"][nearest]
            direct_env.source_targets[env_ids] = bank["source_target"][nearest]
            direct_env.target_hand_qpos[env_ids] = target_qpos[:, :20]
            target_delta_pos = target_qpos[:, 20:23] - source_qpos[:, 20:23]
            direct_env.target_pen_pos[env_ids] = direct_env.pen_pos[env_ids] + target_delta_pos
            # Match each local policy's reset coordinate frame without changing
            # the physical hand or pen state. Global cascade gates use the
            # separately retained initial_pos/initial_axis tensors below.
            direct_env.init_pen_pos[env_ids] = direct_env.pen_pos[env_ids]
            direct_env.init_pen_axis[env_ids] = direct_env.pen_axis[env_ids]
            direct_env.bridge_source_pen_quat[env_ids] = direct_env.pen_quat[env_ids]
            if stage.keep_current_pen_target:
                direct_env.target_pen_quat[env_ids] = direct_env.pen_quat[env_ids]
            else:
                direct_env.target_pen_quat[env_ids] = yaw_rotate(
                    initial_quat[env_ids], stage.target_deg
                )
            direct_env.policy_target_pen_quat[env_ids] = direct_env.pen_quat[env_ids]
            current_axis = direct_env.pen_axis[env_ids]
            phase = torch.atan2(
                initial_axis[env_ids, 0] * current_axis[:, 1]
                - initial_axis[env_ids, 1] * current_axis[:, 0],
                initial_axis[env_ids, 0] * current_axis[:, 0]
                + initial_axis[env_ids, 1] * current_axis[:, 1],
            ) * (180.0 / torch.pi)
            direct_env.local_phase_start_deg_env[env_ids] = phase
            direct_env.previous_total_phase_deg[env_ids] = phase
            direct_env.actions[env_ids] = bank["source_action"][nearest]
            direct_env.previous_actions[env_ids] = bank["source_action"][nearest]
            direct_env.p4_hold_count[env_ids] = 0
            direct_env.action_scale_env[env_ids] = stage.action_scale
            direct_env.policy_horizon_env[env_ids] = stage.policy_horizon_steps
            stage_steps[env_ids] = 0
            hold_counts[env_ids] = 0
            direct_env.episode_length_buf[env_ids] = 0
            target_qpos = torch.cat(
                (
                    direct_env.target_hand_qpos[env_ids],
                    direct_env.target_pen_pos[env_ids],
                    direct_env.target_pen_quat[env_ids],
                ),
                dim=-1,
            )
            transition_caches[stage_id].append(
                {
                    "source_qpos": entry_source_qpos.cpu().numpy(),
                    "source_qvel": entry_source_qvel.cpu().numpy(),
                    "source_target_qpos": entry_actuator_target.cpu().numpy(),
                    "source_policy_action": entry_action.cpu().numpy(),
                    "target_qpos": target_qpos.cpu().numpy(),
                    "source_phase_deg": phase.cpu().numpy(),
                    "source_contact_force_n": direct_env.contact_force_norm[env_ids].cpu().numpy(),
                }
            )

        all_ids = torch.arange(args_cli.num_envs, device=device)
        initial_stages = (0, len(STAGES) - 1) if args_cli.latest_compact else (0, 12)
        for initial_stage in initial_stages:
            stage_env_ids = all_ids[stage_ids == initial_stage]
            if args_cli.latest_compact and not args_cli.dual_entry_eval and initial_stage == 0:
                # This environment is already the exact local P3->7 task. Keep
                # its reset target, source action, and teacher-bank row intact;
                # reconfiguration is only needed after a physical transition.
                stage_bundle_rows[stage_env_ids] = direct_env.current_bank_ids[stage_env_ids]
                direct_env.action_scale_env[stage_env_ids] = STAGES[0].action_scale
                direct_env.policy_horizon_env[stage_env_ids] = STAGES[0].policy_horizon_steps
                stage_entries[initial_stage] += len(stage_env_ids)
                continue
            exact_rows = None
            if args_cli.latest_compact and initial_stage == 0:
                # The compact P3 bundle contains many target candidates for one
                # identical physical source state. Preserve the reset bank's
                # candidate instead of collapsing every environment to row zero.
                exact_rows = direct_env.current_bank_ids[stage_env_ids]
            elif args_cli.latest_compact and args_cli.dual_entry_eval:
                # The dual bundle stores seven consecutive repeats of each of
                # the 197 late-entry rows after the 1315 P3 rows.
                exact_rows = torch.div(
                    direct_env.current_bank_ids[stage_env_ids] - 1315,
                    7,
                    rounding_mode="floor",
                )
            configure(stage_env_ids, initial_stage, exact_rows)
            stage_entries[initial_stage] += len(stage_env_ids)
        observations = direct_env._get_observations()["policy"]

        for rollout_step in range(sum(stage.max_steps for stage in STAGES)):
            observation_np = observations.cpu().numpy()
            actions = torch.zeros((args_cli.num_envs, 20), device=device)
            expert_actions = torch.zeros_like(actions)
            with torch.inference_mode():
                for stage_id, stage in enumerate(STAGES):
                    env_ids = torch.nonzero(alive & (stage_ids == stage_id), as_tuple=False).squeeze(-1)
                    if len(env_ids) == 0:
                        continue
                    if stage.use_teacher_actions:
                        teacher = bundles[stage_id]["teacher_actions"]
                        teacher_step = stage_steps[env_ids].clamp_max(
                            teacher.shape[1] - 1
                        )
                        expert_actions[env_ids] = teacher[
                            stage_bundle_rows[env_ids], teacher_step
                        ]
                    elif stage_id == 0 and use_native_stage0_agent:
                        _, native_outputs = runner.agent.act(
                            observations[env_ids],
                            None,
                            timestep=int(stage_steps[env_ids].min().item()),
                            timesteps=max(stage.max_steps, 1),
                        )
                        expert_actions[env_ids] = native_outputs["mean_actions"]
                    else:
                        processed = preprocessors[stage_id](observations[env_ids])
                        _, outputs = policies[stage_id].act(
                            {"observations": processed}, role="policy"
                        )
                        expert_actions[env_ids] = outputs["mean_actions"]
                    if stage.teacher_prefill_steps > 0:
                        prefill_mask = stage_steps[env_ids] < min(
                            stage.teacher_prefill_steps,
                            bundles[stage_id]["teacher_actions"].shape[1],
                        )
                        if torch.any(prefill_mask):
                            prefill_ids = env_ids[prefill_mask]
                            teacher_step = stage_steps[prefill_ids]
                            expert_actions[prefill_ids] = bundles[stage_id]["teacher_actions"][
                                stage_bundle_rows[prefill_ids], teacher_step
                            ]
                    if student_policy is None:
                        actions[env_ids] = expert_actions[env_ids]
                    else:
                        student_obs = student_preprocessor(observations[env_ids])
                        _, student_outputs = student_policy.act(
                            {"observations": student_obs}, role="policy"
                        )
                        actions[env_ids] = student_outputs["mean_actions"]
            acted_mask = alive.clone()
            if (
                args_cli.dagger_output
                and rollout_step % max(args_cli.dagger_stride, 1) == 0
                and torch.any(acted_mask)
            ):
                dagger_observations.append(observations[acted_mask].cpu().numpy())
                dagger_expert_actions.append(expert_actions[acted_mask].cpu().numpy())
                dagger_student_actions.append(actions[acted_mask].cpu().numpy())
                dagger_stage_ids.append(stage_ids[acted_mask].cpu().numpy())
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)

            done = (terminated | truncated).reshape(-1)
            alive &= ~done
            direct_env._compute_intermediate_values()
            current_axis = direct_env.pen_axis
            phase_deg = torch.atan2(
                initial_axis[:, 0] * current_axis[:, 1]
                - initial_axis[:, 1] * current_axis[:, 0],
                initial_axis[:, 0] * current_axis[:, 0]
                + initial_axis[:, 1] * current_axis[:, 1],
            ) * (180.0 / torch.pi)
            angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
            linear_speed = torch.linalg.norm(direct_env.pen_linvel, dim=-1)
            drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
            global_drift = torch.linalg.norm(direct_env.pen_pos - initial_pos, dim=-1)
            hand_rmse = torch.sqrt(
                torch.mean((direct_env.hand_qpos - direct_env.target_hand_qpos) ** 2, dim=-1)
            )
            qpos = torch.cat(
                (direct_env.hand_qpos, direct_env.pen_pos, direct_env.pen_quat), dim=-1
            ).cpu().numpy()
            qvel = torch.cat(
                (direct_env.hand_qvel, direct_env.pen_linvel, direct_env.pen_angvel), dim=-1
            ).cpu().numpy()
            force = direct_env.contact_force_norm.cpu().numpy()
            phase_np = phase_deg.cpu().numpy()
            angular_np = angular_speed.cpu().numpy()
            linear_np = linear_speed.cpu().numpy()
            drift_np = drift.cpu().numpy()
            global_drift_np = global_drift.cpu().numpy()
            hand_np = hand_rmse.cpu().numpy()
            penetration_np = direct_env.episode_max_penetration.cpu().numpy()
            action_np = actions.cpu().numpy()
            processed_target_np = direct_env.processed_targets.cpu().numpy()
            stage_np = stage_ids.cpu().numpy()
            trace_mask = alive & (stage_ids == 0)
            if batch == 0 and len(stage0_step_trace) < 32 and torch.any(trace_mask):
                stage0_step_trace.append(
                    {
                        "step": len(stage0_step_trace),
                        "valid_envs": int(trace_mask.sum().item()),
                        "mean_action_norm": float(
                            torch.linalg.norm(actions[trace_mask], dim=-1).mean().item()
                        ),
                        "mean_phase_deg": float(phase_deg[trace_mask].mean().item()),
                        "max_phase_deg": float(phase_deg[trace_mask].max().item()),
                        "mean_angular_speed_rad_s": float(
                            angular_speed[trace_mask].mean().item()
                        ),
                        "mean_max_penetration_mm": float(
                            direct_env.episode_max_penetration[trace_mask].mean().item()
                            * 1000.0
                        ),
                        "max_hold_steps": int(hold_counts[trace_mask].max().item()),
                        "mean_episode_step": float(
                            direct_env.episode_length_buf[trace_mask].float().mean().item()
                        ),
                    }
                )
            final_trace_mask = alive & (stage_ids == len(STAGES) - 1) & late_entry
            if (
                batch == 0
                and len(final_stage_step_trace) < 32
                and torch.any(final_trace_mask)
            ):
                final_stage_step_trace.append(
                    {
                        "step": len(final_stage_step_trace),
                        "valid_envs": int(final_trace_mask.sum().item()),
                        "mean_action_norm": float(
                            torch.linalg.norm(actions[final_trace_mask], dim=-1).mean().item()
                        ),
                        "mean_phase_deg": float(phase_deg[final_trace_mask].mean().item()),
                        "max_phase_deg": float(phase_deg[final_trace_mask].max().item()),
                        "mean_angular_speed_rad_s": float(
                            angular_speed[final_trace_mask].mean().item()
                        ),
                        "mean_max_penetration_mm": float(
                            direct_env.episode_max_penetration[final_trace_mask].mean().item()
                            * 1000.0
                        ),
                        "max_hold_steps": int(hold_counts[final_trace_mask].max().item()),
                        "mean_episode_step": float(
                            direct_env.episode_length_buf[final_trace_mask].float().mean().item()
                        ),
                    }
                )
            for env_id in torch.nonzero(alive, as_tuple=False).squeeze(-1).cpu().tolist():
                history = histories[env_id]
                history["qpos"].append(qpos[env_id])
                history["qvel"].append(qvel[env_id])
                history["action"].append(action_np[env_id])
                history["processed_target"].append(processed_target_np[env_id])
                history["contact_force_n"].append(force[env_id])
                history["phase_deg"].append(np.float32(phase_np[env_id]))
                history["angular_speed_rad_s"].append(np.float32(angular_np[env_id]))
                history["linear_speed_m_s"].append(np.float32(linear_np[env_id]))
                history["drift_m"].append(np.float32(drift_np[env_id]))
                history["global_drift_m"].append(np.float32(global_drift_np[env_id]))
                history["hand_rmse_rad"].append(np.float32(hand_np[env_id]))
                history["max_penetration_m"].append(np.float32(penetration_np[env_id]))
                history["stage_id"].append(np.int64(stage_np[env_id]))
                history["observation"].append(observation_np[env_id])

            transitioned: list[tuple[int, torch.Tensor]] = []
            for stage_id, stage in enumerate(STAGES):
                env_mask = alive & (stage_ids == stage_id)
                if not torch.any(env_mask):
                    continue
                contact = direct_env.contact_force_norm[:, :3] > 0.03
                gate = (
                    env_mask
                    & (phase_deg >= stage.min_deg)
                    & (phase_deg <= stage.max_deg)
                    & (angular_speed <= 12.0)
                    & contact.all(dim=-1)
                    & (direct_env.contact_force_norm[:, 0] >= stage.min_thumb_force_n)
                    & (hand_rmse < stage.max_hand_rmse)
                    & (drift < 0.015)
                    & (direct_env.episode_max_penetration <= 0.0005)
                )
                diagnostic = stage_diagnostics[stage_id]
                count = int(env_mask.sum().item())
                diagnostic["samples"] += count
                diagnostic["phase_sum"] += float(phase_deg[env_mask].sum().item())
                diagnostic["phase_min"] = min(
                    diagnostic["phase_min"], float(phase_deg[env_mask].min().item())
                )
                diagnostic["phase_max"] = max(
                    diagnostic["phase_max"], float(phase_deg[env_mask].max().item())
                )
                diagnostic["angular_speed_sum"] += float(
                    angular_speed[env_mask].sum().item()
                )
                diagnostic["triad_contact_samples"] += int(
                    contact[env_mask].all(dim=-1).sum().item()
                )
                diagnostic["hand_gate_samples"] += int(
                    (hand_rmse[env_mask] < stage.max_hand_rmse).sum().item()
                )
                diagnostic["drift_gate_samples"] += int(
                    (drift[env_mask] < 0.015).sum().item()
                )
                diagnostic["penetration_gate_samples"] += int(
                    (direct_env.episode_max_penetration[env_mask] <= 0.0005).sum().item()
                )
                diagnostic["instantaneous_gate_samples"] += int(gate.sum().item())
                hold_counts[env_mask] = torch.where(
                    gate[env_mask], hold_counts[env_mask] + 1, torch.zeros_like(hold_counts[env_mask])
                )
                diagnostic["max_hold_steps"] = max(
                    diagnostic["max_hold_steps"], int(hold_counts[env_mask].max().item())
                )
                passed = env_mask & (hold_counts >= stage.hold_steps)
                timed_out = env_mask & (stage_steps >= stage.max_steps)
                alive[timed_out] = False
                if torch.any(passed):
                    passed_ids = torch.nonzero(passed, as_tuple=False).squeeze(-1)
                    stage_successes[stage_id] += len(passed_ids)
                    if stage_id == len(STAGES) - 1:
                        for env_id in passed_ids.cpu().tolist():
                            final_success[env_id] = True
                            row = {
                                name: np.stack(values)
                                for name, values in histories[env_id].items()
                            }
                            row["initial_qpos"] = initial_qpos[env_id]
                            row["initial_qvel"] = initial_qvel[env_id]
                            row["initial_bank_id"] = np.int64(initial_bank_ids[env_id])
                            row["reset_seed"] = np.int64(reset_seed)
                            row["entry_label"] = np.int64(initial_entry_labels[env_id])
                            trajectories.append(row)
                            entrance_successes[str(int(initial_entry_labels[env_id]))] += 1
                        alive[passed_ids] = False
                    else:
                        transitioned.append((stage_id + 1, passed_ids))
            stage_steps[alive] += 1
            for next_stage, env_ids in transitioned:
                stage_ids[env_ids] = next_stage
                stage_entries[next_stage] += len(env_ids)
                configure(env_ids, next_stage)
            if transitioned:
                observations = direct_env._get_observations()["policy"]
            if not torch.any(alive) or len(trajectories) >= args_cli.max_trajectories:
                break

        if reached_stage_id is not None and args_cli.reached_stage_output:
            for env_id, history in enumerate(histories):
                if not history["stage_id"] or reached_stage_id not in history["stage_id"]:
                    continue
                row = {name: np.stack(values) for name, values in history.items()}
                row["initial_qpos"] = initial_qpos[env_id]
                row["initial_qvel"] = initial_qvel[env_id]
                row["initial_bank_id"] = np.int64(initial_bank_ids[env_id])
                row["reset_seed"] = np.int64(reset_seed)
                row["entry_label"] = np.int64(initial_entry_labels[env_id])
                row["final_success"] = np.bool_(final_success[env_id])
                reached_stage_trajectories.append(row)

        batch_success = sum(
            1 for row in trajectories if int(row["stage_id"][0]) == 0
        ) - sum(batch_full_success)
        batch_full_success.append(max(batch_success, 0))
        print(
            f"CASCADE_BATCH={batch + 1} NEW_FULL_SUCCESS={batch_full_success[-1]} "
            f"TOTAL={len(trajectories)}",
            flush=True,
        )
        if len(trajectories) >= args_cli.max_trajectories:
            break

    output = Path(args_cli.output)
    if not output.is_absolute():
        output = ROOT / output
    report_path = Path(args_cli.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "continuous_without_intermediate_reset": True,
        "requested_trajectories": args_cli.max_trajectories,
        "collected_trajectories": len(trajectories),
        "batches": len(batch_full_success),
        "batch_successes": batch_full_success,
        "entrance_stats": {
            label: {
                "entries": entrance_entries[label],
                "successes": entrance_successes[label],
                "success_rate": entrance_successes[label] / max(entrance_entries[label], 1),
            }
            for label in ("0", "45")
        },
        "reset_sampling": {
            "base_seed": args_cli.seed,
            "seed_stride": args_cli.reset_seed_stride,
            "batches": reset_batch_audits,
        },
        "stage_stats": {
            stage.name: {
                "entries": int(stage_entries[index]),
                "successes": int(stage_successes[index]),
                "conditional_success_rate": float(stage_successes[index])
                / max(int(stage_entries[index]), 1),
            }
            for index, stage in enumerate(STAGES)
        },
        "stage_diagnostics": {
            stage.name: {
                "samples": diagnostic["samples"],
                "mean_phase_deg": diagnostic["phase_sum"] / max(diagnostic["samples"], 1),
                "min_phase_deg": diagnostic["phase_min"]
                if diagnostic["samples"]
                else None,
                "max_phase_deg": diagnostic["phase_max"]
                if diagnostic["samples"]
                else None,
                "mean_angular_speed_rad_s": diagnostic["angular_speed_sum"]
                / max(diagnostic["samples"], 1),
                "triad_contact_ratio": diagnostic["triad_contact_samples"]
                / max(diagnostic["samples"], 1),
                "hand_gate_ratio": diagnostic["hand_gate_samples"]
                / max(diagnostic["samples"], 1),
                "drift_gate_ratio": diagnostic["drift_gate_samples"]
                / max(diagnostic["samples"], 1),
                "penetration_gate_ratio": diagnostic["penetration_gate_samples"]
                / max(diagnostic["samples"], 1),
                "instantaneous_gate_ratio": diagnostic["instantaneous_gate_samples"]
                / max(diagnostic["samples"], 1),
                "max_hold_steps": diagnostic["max_hold_steps"],
            }
            for stage, diagnostic in zip(STAGES, stage_diagnostics, strict=True)
        },
        "stage0_step_trace": stage0_step_trace,
        "final_stage_step_trace": final_stage_step_trace,
        "environment_max_episode_length": int(direct_env.max_episode_length),
    }
    if dagger_observations:
        dagger_payload = {
            "observation": np.concatenate(dagger_observations),
            "action": np.concatenate(dagger_expert_actions),
            "student_action": np.concatenate(dagger_student_actions),
            "stage_id": np.concatenate(dagger_stage_ids),
            "source_student_checkpoint": np.asarray(str(args_cli.single_checkpoint)),
        }
        dagger_path = Path(args_cli.dagger_output)
        if not dagger_path.is_absolute():
            dagger_path = ROOT / dagger_path
        dagger_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dagger_path, **dagger_payload)
        disagreement = np.linalg.norm(
            dagger_payload["action"] - dagger_payload["student_action"], axis=-1
        )
        stages, counts = np.unique(dagger_payload["stage_id"], return_counts=True)
        report["dagger"] = {
            "path": str(dagger_path),
            "rows": len(disagreement),
            "mean_action_l2_disagreement": float(disagreement.mean()),
            "p95_action_l2_disagreement": float(np.quantile(disagreement, 0.95)),
            "stage_counts": {
                str(int(stage)): int(count)
                for stage, count in zip(stages, counts, strict=True)
            },
        }
    if trajectories:
        trajectories = trajectories[: args_cli.max_trajectories]
        lengths = np.asarray([len(row["phase_deg"]) for row in trajectories], dtype=np.int64)
        max_length = int(lengths.max())
        payload: dict[str, np.ndarray] = {
            "lengths": lengths,
            "initial_qpos": np.stack([row["initial_qpos"] for row in trajectories]),
            "initial_qvel": np.stack([row["initial_qvel"] for row in trajectories]),
            "initial_bank_ids": np.asarray(
                [row["initial_bank_id"] for row in trajectories], dtype=np.int64
            ),
            "reset_seeds": np.asarray(
                [row["reset_seed"] for row in trajectories], dtype=np.int64
            ),
            "entry_labels": np.asarray(
                [row["entry_label"] for row in trajectories], dtype=np.int64
            ),
            "control_dt_s": np.asarray(1.0 / 120.0, dtype=np.float32),
            "stage_names": np.asarray([stage.name for stage in STAGES]),
            "continuous_without_intermediate_reset": np.asarray(True),
        }
        for name in trajectories[0]:
            if name in {"initial_qpos", "initial_qvel", "initial_bank_id", "reset_seed", "entry_label"}:
                continue
            sample = trajectories[0][name]
            padded = np.zeros(
                (len(trajectories), max_length, *sample.shape[1:]), dtype=sample.dtype
            )
            for index, row in enumerate(trajectories):
                padded[index, : lengths[index]] = row[name]
            payload[name] = padded
        np.savez_compressed(output, **payload)
        report["trajectory_length_steps"] = {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        }
    if args_cli.reached_stage_output:
        reached_path = Path(args_cli.reached_stage_output)
        if not reached_path.is_absolute():
            reached_path = ROOT / reached_path
        reached_path.parent.mkdir(parents=True, exist_ok=True)
        if reached_stage_trajectories:
            reached_lengths = np.asarray(
                [len(row["phase_deg"]) for row in reached_stage_trajectories],
                dtype=np.int64,
            )
            reached_max_length = int(reached_lengths.max())
            reached_payload: dict[str, np.ndarray] = {
                "lengths": reached_lengths,
                "initial_qpos": np.stack(
                    [row["initial_qpos"] for row in reached_stage_trajectories]
                ),
                "initial_qvel": np.stack(
                    [row["initial_qvel"] for row in reached_stage_trajectories]
                ),
                "initial_bank_ids": np.asarray(
                    [row["initial_bank_id"] for row in reached_stage_trajectories],
                    dtype=np.int64,
                ),
                "reset_seeds": np.asarray(
                    [row["reset_seed"] for row in reached_stage_trajectories],
                    dtype=np.int64,
                ),
                "entry_labels": np.asarray(
                    [row["entry_label"] for row in reached_stage_trajectories],
                    dtype=np.int64,
                ),
                "final_success": np.asarray(
                    [row["final_success"] for row in reached_stage_trajectories],
                    dtype=np.bool_,
                ),
                "control_dt_s": np.asarray(1.0 / 120.0, dtype=np.float32),
                "stage_names": np.asarray([stage.name for stage in STAGES]),
                "reached_stage_name": np.asarray(args_cli.reached_stage_name),
                "continuous_without_intermediate_reset": np.asarray(True),
            }
            metadata = {
                "initial_qpos",
                "initial_qvel",
                "initial_bank_id",
                "reset_seed",
                "entry_label",
                "final_success",
            }
            for name in reached_stage_trajectories[0]:
                if name in metadata:
                    continue
                sample = reached_stage_trajectories[0][name]
                padded = np.zeros(
                    (
                        len(reached_stage_trajectories),
                        reached_max_length,
                        *sample.shape[1:],
                    ),
                    dtype=sample.dtype,
                )
                for index, row in enumerate(reached_stage_trajectories):
                    padded[index, : reached_lengths[index]] = row[name]
                reached_payload[name] = padded
            np.savez_compressed(reached_path, **reached_payload)
            report["reached_stage_export"] = {
                "path": str(reached_path),
                "stage": args_cli.reached_stage_name,
                "rows": len(reached_stage_trajectories),
                "final_successes": int(
                    sum(
                        bool(row["final_success"])
                        for row in reached_stage_trajectories
                    )
                ),
            }
        else:
            report["reached_stage_export"] = {
                "path": str(reached_path),
                "stage": args_cli.reached_stage_name,
                "rows": 0,
                "final_successes": 0,
            }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cache_outputs = {}
    for stage_id, rows in enumerate(transition_caches):
        if not rows:
            continue
        merged = {
            name: np.concatenate([row[name] for row in rows], axis=0)
            for name in rows[0]
        }
        count = len(merged["source_qpos"])
        source_phase = merged.pop("source_phase_deg").astype(np.float32)
        target_remaining = np.deg2rad(
            STAGES[stage_id].target_deg - source_phase
        ).astype(np.float32)
        cache_path = report_path.with_name(
            f"{report_path.stem}_{STAGES[stage_id].name}_entry_bundle.npz"
        )
        np.savez_compressed(
            cache_path,
            **merged,
            source_teacher_actions=np.zeros((count, 0, 20), dtype=np.float32),
            target_qvel=np.zeros((count, 26), dtype=np.float32),
            phase_ids=np.full(count, 3, dtype=np.int64),
            source_episode_step=np.zeros(count, dtype=np.int64),
            source_phase_offset_rad=np.deg2rad(source_phase).astype(np.float32),
            target_remaining_rad=target_remaining,
            policy_target_pen_quat=merged["source_qpos"][:, 23:27].copy(),
            curriculum_stage_deg=source_phase,
            target_p4_candidate_ids=np.arange(count, dtype=np.int64),
        )
        cache_outputs[STAGES[stage_id].name] = {
            "path": str(cache_path),
            "rows": count,
            "phase_deg": {
                "min": float(source_phase.min()),
                "mean": float(source_phase.mean()),
                "max": float(source_phase.max()),
            },
        }
    report["transition_entry_caches"] = cache_outputs
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
