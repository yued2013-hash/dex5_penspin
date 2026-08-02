from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate a phase-conditioned P4-to-P1 exemplar teacher."
)
ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", ROOT / ".deps/IsaacLab"))
ISAACLAB_LOG_ROOT = Path(
    os.environ.get("ISAACLAB_LOG_ROOT", ISAACLAB_ROOT / "logs/skrl")
)
parser.add_argument(
    "--task",
    default=(
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapNoResetTeacher-"
        "Direct-v0"
    ),
)
parser.add_argument(
    "--early-checkpoint",
    default=(
        ISAACLAB_LOG_ROOT
        / "dex5_recapture_p3_physics_direct/"
        "2026-07-20_04-22-19_ppo_torch/checkpoints/agent_2560.pt"
    ),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=2630)
parser.add_argument(
    "--bundle",
    type=Path,
    help="Override the reset bundle, for example a true phase-7 entry.",
)
parser.add_argument("--trajectory-output", type=Path)
parser.add_argument("--phase5-cache-output", type=Path)
parser.add_argument("--phase6-cache-output", type=Path)
parser.add_argument("--phase7-cache-output", type=Path)
parser.add_argument("--phase8-cache-output", type=Path)
parser.add_argument("--max-cache", type=int, default=512)
parser.add_argument("--trace-transition-source", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import nn

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import quat_apply  # noqa: E402


REPORT_ROOT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/no_reset_chain"
)


@dataclass
class ExemplarLibrary:
    start_qpos: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    first_duration: torch.Tensor
    second_duration: torch.Tensor


LIBRARY_FILES = {
    4: (
        None,
        "phase4夹持满幅起点动作映射_seed2634.npz",
    ),
    5: (
        "phase5真实夹持入口含控制目标_seed2652.npz",
        "phase5到phase6真实控制目标搜索_seed2653.npz",
    ),
    6: (
        "phase6重锁存真实入口_seed2652.npz",
        "phase6到phase7八次全过动作_seed2661.npz",
    ),
    7: (
        "phase7真实无重置下托接触入口_seed2652.npz",
        "phase7到phase8保持后锁定承力_seed2674.npz",
    ),
    8: (
        "拇指下托phase8合并入口_127条.npz",
        "phase8到phase9最后下托搜索_seed2605.npz",
    ),
    9: (
        "phase8到phase9最后下托搜索_seed2605.npz",
        "phase9拇指绕侧搜索_seed2606.npz",
    ),
    10: (
        "phase9拇指绕侧搜索_seed2606.npz",
        "phase10拇指绕侧搜索_seed2607.npz",
    ),
    11: (
        "phase10拇指绕侧搜索_seed2607.npz",
        "phase11拇指绕侧搜索_seed2608.npz",
    ),
    12: (
        "phase11拇指绕侧搜索_seed2608.npz",
        "phase12新版绕侧完成验证_seed2619.npz",
    ),
    13: (
        "phase12拇指绕侧完成扩增_seed2610.npz",
        "phase13拇指食指接管扩增_seed2612.npz",
    ),
    14: (
        "phase13拇指食指接管扩增_seed2612.npz",
        "phase14新版拇食指接管验证_seed2620.npz",
    ),
    15: (
        "phase14拇指食指接管完成搜索_seed2613.npz",
        "phase15新版无名指卸载验证_seed2621.npz",
    ),
    16: (
        "phase15无名指卸载搜索_seed2614.npz",
        "phase16回到P1入口搜索_seed2615.npz",
    ),
}


class Policy(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        super().__init__()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state = checkpoint["policy"]
        self.net = nn.Sequential(
            nn.Linear(156, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 20),
        )
        for layer, key in (
            (0, "net_container.0"),
            (2, "net_container.2"),
            (4, "net_container.4"),
            (6, "policy_layer"),
        ):
            self.net[layer].weight.data.copy_(state[f"{key}.weight"])
            self.net[layer].bias.data.copy_(state[f"{key}.bias"])
        preprocessor = checkpoint["observation_preprocessor"]
        self.register_buffer(
            "mean", preprocessor["running_mean"].float()
        )
        self.register_buffer(
            "variance", preprocessor["running_variance"].float()
        )
        self.to(device).eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (observation - self.mean)
            / (torch.sqrt(self.variance) + 1.0e-8),
            -5.0,
            5.0,
        )
        return self.net(normalized)


def load_library(
    start_name: str | None, action_name: str, device: torch.device
) -> ExemplarLibrary:
    with np.load(REPORT_ROOT / action_name) as action_data:
        selected = np.ones(
            len(action_data["candidate_action_first"]), dtype=bool
        )
        if "selection_kind" in action_data:
            selected = (
                np.asarray(action_data["selection_kind"]).astype(str)
                == "success"
            )
            if not selected.any():
                raise ValueError(
                    f"{action_name}: no strict-success action rows"
                )
        if start_name is None:
            start_qpos = np.asarray(
                action_data["action_start_qpos"], dtype=np.float32
            )
            bank_ids = np.arange(len(start_qpos), dtype=np.int64)
        else:
            bank_ids = np.asarray(
                action_data["source_bank_id"][selected], dtype=np.int64
            )
        first = np.asarray(
            action_data["candidate_action_first"][selected],
            dtype=np.float32,
        )
        second = np.asarray(
            action_data["candidate_action_second"][selected],
            dtype=np.float32,
        )
        first_duration = np.asarray(
            action_data["first_duration"][selected], dtype=np.int64
        )
        second_duration = np.asarray(
            action_data["second_duration"][selected], dtype=np.int64
        )
        stable_rows = None
        if (
            start_name is None
            and "max_contact_run" in action_data
        ):
            stable_rows = (
                np.asarray(action_data["max_contact_run"], dtype=np.int64)
                >= 10
            )
    if stable_rows is not None and stable_rows.any():
        start_qpos = start_qpos[stable_rows]
        bank_ids = np.arange(len(start_qpos), dtype=np.int64)
        first = first[stable_rows]
        second = second[stable_rows]
        first_duration = first_duration[stable_rows]
        second_duration = second_duration[stable_rows]
    if start_name is not None:
        with np.load(REPORT_ROOT / start_name) as start_data:
            start_qpos = np.asarray(
                start_data["source_qpos"], dtype=np.float32
            )
    if bank_ids.max(initial=-1) >= len(start_qpos):
        raise ValueError(
            f"{action_name}: source_bank_id exceeds {start_name}"
        )
    return ExemplarLibrary(
        start_qpos=torch.as_tensor(
            start_qpos[bank_ids], device=device
        ),
        first=torch.as_tensor(first, device=device),
        second=torch.as_tensor(second, device=device),
        first_duration=torch.as_tensor(
            first_duration, dtype=torch.long, device=device
        ),
        second_duration=torch.as_tensor(
            second_duration, dtype=torch.long, device=device
        ),
    )


def current_qpos(direct) -> torch.Tensor:
    return torch.cat(
        (direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1
    )


def nearest_rows(
    query: torch.Tensor, library: ExemplarLibrary
) -> torch.Tensor:
    scale = torch.as_tensor(
        [0.25] * 20 + [0.010] * 3 + [0.20] * 4,
        dtype=query.dtype,
        device=query.device,
    )
    delta = (
        query.unsqueeze(1) - library.start_qpos.unsqueeze(0)
    ) / scale
    return torch.argmin(torch.mean(delta.square(), dim=-1), dim=1)


def main() -> None:
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    if args.bundle is not None:
        cfg.bundle_path = str(args.bundle.resolve())
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        (args.steps + 2) * cfg.decimation * cfg.sim.dt,
    )
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    policy = Policy(args.early_checkpoint, direct.device)
    libraries = {
        phase: load_library(start, action, direct.device)
        for phase, (start, action) in LIBRARY_FILES.items()
    }
    observation, _ = env.reset()
    phase = direct.chain_phase_ids.clone()
    previous_phase = torch.full_like(phase, -1)
    exemplar_row = torch.zeros_like(phase)
    action_age = torch.zeros_like(phase)
    phase7_hold_latched = torch.zeros(
        direct.num_envs, dtype=torch.bool, device=direct.device
    )
    max_phase = phase.clone()
    global_deepest_phase = int(phase.max().item())
    max_penetration_m = 0.0
    transition_counts = torch.zeros(
        18, dtype=torch.long, device=direct.device
    )
    action_jump_sum = torch.zeros(
        18, dtype=torch.float32, device=direct.device
    )
    action_jump_max = torch.zeros_like(action_jump_sum)
    previous_applied_action = direct.actions.clone()
    completed = 0
    successes = 0
    drops = 0
    success_serial = direct.latest_success_serial.clone()
    trace_env: int | None = None
    trace_finished = False
    trace_qpos: list[np.ndarray] = []
    trace_qvel: list[np.ndarray] = []
    trace_phase: list[int] = []
    trace_penetration: list[float] = []
    pending_qpos: list[np.ndarray] = []
    pending_qvel: list[np.ndarray] = []
    pending_phase: list[np.ndarray] = []
    pending_penetration: list[np.ndarray] = []
    phase4_min_ring_gap_m = float("inf")
    phase4_ring_contact_steps = 0
    phase4_min_contact_rmse_rad = float("inf")
    phase4_nearest_distance_sum = 0.0
    phase4_nearest_distance_count = 0
    phase4_contact_pose_steps = 0
    phase4_contact_pose_safe_component_steps = np.zeros(5, dtype=np.int64)
    phase4_contact_pose_safe_steps = 0
    phase4_contact_pose_min_position_error_m = float("inf")
    phase4_contact_pose_min_axis_error_rad = float("inf")
    phase4_contact_pose_min_speed_rad_s = float("inf")
    phase4_contact_pose_min_penetration_m = float("inf")
    phase5_cache: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "source_qpos",
            "source_qvel",
            "source_target_qpos",
            "source_target_hand_qpos",
            "source_closure_origin_hand",
            "source_closure_brace_hand",
            "source_closure_origin_pen_pos",
            "source_closure_origin_pen_quat",
            "source_init_pen_pos",
            "source_init_pen_axis",
            "source_max_penetration_m",
            "source_contact_force_n",
            "source_policy_action",
        )
    }
    phase6_cache: dict[str, list[np.ndarray]] = {
        key: [] for key in phase5_cache
    }
    phase7_cache: dict[str, list[np.ndarray]] = {
        key: [] for key in phase5_cache
    }
    phase8_cache: dict[str, list[np.ndarray]] = {
        key: [] for key in phase5_cache
    }

    for _ in range(args.steps):
        phase = direct.chain_phase_ids.clone()
        entered = phase != previous_phase
        for phase_id, library in libraries.items():
            active_entry = entered & (phase == phase_id)
            if torch.any(active_entry):
                ids = torch.nonzero(
                    active_entry, as_tuple=False
                ).squeeze(-1)
                if phase_id == 7:
                    exemplar_row[ids] = (
                        ids % library.first.shape[0]
                    )
                else:
                    exemplar_row[ids] = nearest_rows(
                        current_qpos(direct)[ids], library
                    )
                if phase_id == 4:
                    scale = torch.as_tensor(
                        [0.25] * 20
                        + [0.010] * 3
                        + [0.20] * 4,
                        device=direct.device,
                    )
                    selected_start = library.start_qpos[
                        exemplar_row[ids]
                    ]
                    distance = torch.sqrt(
                        torch.mean(
                            (
                                (
                                    current_qpos(direct)[ids]
                                    - selected_start
                                )
                                / scale
                            ).square(),
                            dim=-1,
                        )
                    )
                    phase4_nearest_distance_sum += float(
                        distance.sum().item()
                    )
                    phase4_nearest_distance_count += int(ids.numel())
                action_age[ids] = 0
                if phase_id == 7:
                    phase7_hold_latched[ids] = False

        with torch.inference_mode():
            actions = policy(observation["policy"])
            phase7_hold_latched |= (
                (phase == 7) & (direct.chain_phase_hold > 0)
            )
            for phase_id, library in libraries.items():
                active = phase == phase_id
                if not torch.any(active):
                    continue
                ids = torch.nonzero(
                    active, as_tuple=False
                ).squeeze(-1)
                rows = exemplar_row[ids]
                age = action_age[ids]
                first_end = library.first_duration[rows]
                second_end = (
                    first_end + library.second_duration[rows]
                )
                local_action = torch.where(
                    (age < first_end).unsqueeze(-1),
                    library.first[rows],
                    torch.where(
                        (age < second_end).unsqueeze(-1),
                        library.second[rows],
                        torch.zeros_like(library.first[rows]),
                    ),
                )
                if phase_id == 7:
                    local_action = torch.where(
                        phase7_hold_latched[ids].unsqueeze(-1),
                        library.second[rows],
                        local_action,
                    )
                actions[ids] = local_action
            actions = torch.where(
                (phase == 17).unsqueeze(-1),
                torch.zeros_like(actions),
                actions,
            )
            observation, _, terminated, truncated, _ = env.step(actions)

        direct._compute_intermediate_values()
        phase4_active = direct.chain_phase_ids == 4
        if torch.any(phase4_active):
            ring_gap = (
                direct.tip_surface_dist[:, 3]
                - direct.fingertip_radii[:, 3]
            )
            phase4_min_ring_gap_m = min(
                phase4_min_ring_gap_m,
                float(ring_gap[phase4_active].min().item()),
            )
            contacts = (
                direct.contact_force_norm
                > direct.cfg.p1_contact_force_n
            )
            contact_active = (
                phase4_active & contacts[:, 1] & contacts[:, 3]
            )
            phase4_ring_contact_steps += int(
                contact_active.sum().item()
            )
            if torch.any(contact_active):
                index_ring_error = torch.cat(
                    (
                        direct.hand_qpos[:, 0:4]
                        - direct.target_hand_qpos[:, 0:4],
                        direct.hand_qpos[:, 8:12]
                        - direct.target_hand_qpos[:, 8:12],
                    ),
                    dim=-1,
                )
                index_ring_rmse = torch.sqrt(
                    torch.mean(index_ring_error.square(), dim=-1)
                )
                phase4_min_contact_rmse_rad = min(
                    phase4_min_contact_rmse_rad,
                    float(
                        index_ring_rmse[contact_active].min().item()
                    ),
                )
                pose_active = contact_active & (
                    index_ring_rmse
                    <= float(direct.cfg.closure_stage_rmse_thresholds[0])
                )
                if torch.any(pose_active):
                    target_axis = quat_apply(
                        direct.target_pen_quat, direct.pen_x_axis
                    )
                    position_error = torch.linalg.norm(
                        direct.pen_pos - direct.target_pen_pos, dim=-1
                    )
                    axis_dot = torch.sum(
                        direct.pen_axis * target_axis, dim=-1
                    ).clamp(-1.0, 1.0)
                    axis_error = torch.acos(axis_dot.abs())
                    speed = torch.linalg.norm(
                        direct.pen_angvel, dim=-1
                    )
                    peak_force = direct.contact_force_norm.max(
                        dim=-1
                    ).values
                    components = torch.stack(
                        (
                            position_error
                            <= float(
                                direct.cfg
                                .thumb_wrap_brace_pen_position_tolerance_m
                            ),
                            axis_error
                            <= torch.deg2rad(
                                torch.full_like(
                                    axis_error,
                                    float(
                                        direct.cfg
                                        .thumb_wrap_acquisition_pen_axis_tolerance_deg
                                    ),
                                )
                            ),
                            speed
                            <= float(
                                direct.cfg.p5_max_angular_speed_rad_s
                            ),
                            direct.episode_max_penetration
                            <= float(direct.cfg.penetration_threshold),
                            peak_force <= float(direct.cfg.p1_max_force_n),
                        ),
                        dim=-1,
                    )
                    phase4_contact_pose_steps += int(
                        pose_active.sum().item()
                    )
                    phase4_contact_pose_safe_component_steps += (
                        components[pose_active]
                        .sum(dim=0)
                        .cpu()
                        .numpy()
                    )
                    phase4_contact_pose_safe_steps += int(
                        (
                            pose_active
                            & components.all(dim=-1)
                        ).sum().item()
                    )
                    phase4_contact_pose_min_position_error_m = min(
                        phase4_contact_pose_min_position_error_m,
                        float(position_error[pose_active].min().item()),
                    )
                    phase4_contact_pose_min_axis_error_rad = min(
                        phase4_contact_pose_min_axis_error_rad,
                        float(axis_error[pose_active].min().item()),
                    )
                    phase4_contact_pose_min_speed_rad_s = min(
                        phase4_contact_pose_min_speed_rad_s,
                        float(speed[pose_active].min().item()),
                    )
                    phase4_contact_pose_min_penetration_m = min(
                        phase4_contact_pose_min_penetration_m,
                        float(
                            direct.episode_max_penetration[
                                pose_active
                            ].min().item()
                        ),
                    )
        applied = direct.actions.clone()
        transitioned = direct.chain_phase_ids == (phase + 1)
        entered_phase5 = transitioned & (phase == 4)
        cached_count = sum(
            len(batch) for batch in phase5_cache["source_qpos"]
        )
        if (
            args.phase5_cache_output is not None
            and torch.any(entered_phase5)
            and cached_count < args.max_cache
        ):
            ids = torch.nonzero(
                entered_phase5, as_tuple=False
            ).squeeze(-1)[: args.max_cache - cached_count]
            state_batches = {
                "source_qpos": current_qpos(direct)[ids],
                "source_qvel": torch.cat(
                    (
                        direct.hand_qvel,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[ids],
                "source_target_qpos": direct.policy_targets[ids],
                "source_target_hand_qpos": direct.target_hand_qpos[ids],
                "source_closure_origin_hand": (
                    direct.closure_origin_hand[ids]
                ),
                "source_closure_brace_hand": (
                    direct.closure_brace_hand[ids]
                ),
                "source_closure_origin_pen_pos": (
                    direct.closure_origin_pen_pos[ids]
                ),
                "source_closure_origin_pen_quat": (
                    direct.closure_origin_pen_quat[ids]
                ),
                "source_init_pen_pos": direct.init_pen_pos[ids],
                "source_init_pen_axis": direct.init_pen_axis[ids],
                "source_max_penetration_m": (
                    direct.episode_max_penetration[ids]
                ),
                "source_contact_force_n": (
                    direct.contact_force_norm[ids]
                ),
                "source_policy_action": direct.actions[ids],
            }
            for key, value in state_batches.items():
                phase5_cache[key].append(
                    value.detach().cpu().numpy().copy()
                )
        entered_phase6 = transitioned & (phase == 5)
        cached_phase6_count = sum(
            len(batch) for batch in phase6_cache["source_qpos"]
        )
        if (
            args.phase6_cache_output is not None
            and torch.any(entered_phase6)
            and cached_phase6_count < args.max_cache
        ):
            ids = torch.nonzero(
                entered_phase6, as_tuple=False
            ).squeeze(-1)[: args.max_cache - cached_phase6_count]
            state_batches = {
                "source_qpos": current_qpos(direct)[ids],
                "source_qvel": torch.cat(
                    (
                        direct.hand_qvel,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[ids],
                "source_target_qpos": direct.policy_targets[ids],
                "source_target_hand_qpos": direct.target_hand_qpos[ids],
                "source_closure_origin_hand": (
                    direct.closure_origin_hand[ids]
                ),
                "source_closure_brace_hand": (
                    direct.closure_brace_hand[ids]
                ),
                "source_closure_origin_pen_pos": (
                    direct.closure_origin_pen_pos[ids]
                ),
                "source_closure_origin_pen_quat": (
                    direct.closure_origin_pen_quat[ids]
                ),
                "source_init_pen_pos": direct.init_pen_pos[ids],
                "source_init_pen_axis": direct.init_pen_axis[ids],
                "source_max_penetration_m": (
                    direct.episode_max_penetration[ids]
                ),
                "source_contact_force_n": (
                    direct.contact_force_norm[ids]
                ),
                "source_policy_action": direct.actions[ids],
            }
            for key, value in state_batches.items():
                phase6_cache[key].append(
                    value.detach().cpu().numpy().copy()
                )
        entered_phase7 = transitioned & (phase == 6)
        cached_phase7_count = sum(
            len(batch) for batch in phase7_cache["source_qpos"]
        )
        if (
            args.phase7_cache_output is not None
            and torch.any(entered_phase7)
            and cached_phase7_count < args.max_cache
        ):
            ids = torch.nonzero(
                entered_phase7, as_tuple=False
            ).squeeze(-1)[: args.max_cache - cached_phase7_count]
            state_batches = {
                "source_qpos": current_qpos(direct)[ids],
                "source_qvel": torch.cat(
                    (
                        direct.hand_qvel,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[ids],
                "source_target_qpos": direct.policy_targets[ids],
                "source_target_hand_qpos": direct.target_hand_qpos[ids],
                "source_closure_origin_hand": (
                    direct.closure_origin_hand[ids]
                ),
                "source_closure_brace_hand": (
                    direct.closure_brace_hand[ids]
                ),
                "source_closure_origin_pen_pos": (
                    direct.closure_origin_pen_pos[ids]
                ),
                "source_closure_origin_pen_quat": (
                    direct.closure_origin_pen_quat[ids]
                ),
                "source_init_pen_pos": direct.init_pen_pos[ids],
                "source_init_pen_axis": direct.init_pen_axis[ids],
                "source_max_penetration_m": (
                    direct.episode_max_penetration[ids]
                ),
                "source_contact_force_n": (
                    direct.contact_force_norm[ids]
                ),
                "source_policy_action": direct.actions[ids],
            }
            for key, value in state_batches.items():
                phase7_cache[key].append(
                    value.detach().cpu().numpy().copy()
                )
        entered_phase8 = transitioned & (phase == 7)
        cached_phase8_count = sum(
            len(batch) for batch in phase8_cache["source_qpos"]
        )
        if (
            args.phase8_cache_output is not None
            and torch.any(entered_phase8)
            and cached_phase8_count < args.max_cache
        ):
            ids = torch.nonzero(
                entered_phase8, as_tuple=False
            ).squeeze(-1)[: args.max_cache - cached_phase8_count]
            state_batches = {
                "source_qpos": current_qpos(direct)[ids],
                "source_qvel": torch.cat(
                    (
                        direct.hand_qvel,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[ids],
                "source_target_qpos": direct.policy_targets[ids],
                "source_target_hand_qpos": direct.target_hand_qpos[ids],
                "source_closure_origin_hand": (
                    direct.closure_origin_hand[ids]
                ),
                "source_closure_brace_hand": (
                    direct.closure_brace_hand[ids]
                ),
                "source_closure_origin_pen_pos": (
                    direct.closure_origin_pen_pos[ids]
                ),
                "source_closure_origin_pen_quat": (
                    direct.closure_origin_pen_quat[ids]
                ),
                "source_init_pen_pos": direct.init_pen_pos[ids],
                "source_init_pen_axis": direct.init_pen_axis[ids],
                "source_max_penetration_m": (
                    direct.episode_max_penetration[ids]
                ),
                "source_contact_force_n": (
                    direct.contact_force_norm[ids]
                ),
                "source_policy_action": direct.actions[ids],
            }
            for key, value in state_batches.items():
                phase8_cache[key].append(
                    value.detach().cpu().numpy().copy()
                )
        if torch.any(transitioned):
            source = phase[transitioned].clamp(0, 17)
            jump = torch.max(
                torch.abs(
                    applied[transitioned]
                    - previous_applied_action[transitioned]
                ),
                dim=-1,
            ).values
            for source_phase in torch.unique(source).tolist():
                source_active = source == source_phase
                count = int(source_active.sum().item())
                transition_counts[source_phase] += count
                action_jump_sum[source_phase] += jump[
                    source_active
                ].sum()
                action_jump_max[source_phase] = torch.maximum(
                    action_jump_max[source_phase],
                    jump[source_active].max(),
                )
        previous_applied_action = applied
        action_age += 1
        max_phase = torch.maximum(max_phase, direct.chain_phase_ids)
        global_deepest_phase = max(
            global_deepest_phase,
            int(max_phase.max().item()),
        )
        max_penetration_m = max(
            max_penetration_m,
            float(direct.episode_max_penetration.max().item()),
        )

        new_success = direct.latest_success_serial > success_serial
        successes += int(new_success.sum().item())
        success_serial = direct.latest_success_serial.clone()
        done = (terminated | truncated).reshape(-1)
        completed += int(done.sum().item())
        if torch.any(done):
            height_drop = (
                direct.init_pen_pos[:, 2] - direct.pen_pos[:, 2]
            )
            drops += int((done & (height_drop > 0.035)).sum().item())
            action_age = torch.where(
                done, torch.zeros_like(action_age), action_age
            )
            exemplar_row = torch.where(
                done, torch.zeros_like(exemplar_row), exemplar_row
            )
            max_phase = torch.where(
                done,
                direct.chain_phase_ids,
                max_phase,
            )

        if args.trajectory_output is not None and not trace_finished:
            state_qpos = current_qpos(direct).detach().cpu().numpy()
            state_qvel = torch.cat(
                (
                    direct.hand_qvel,
                    direct.pen_linvel,
                    direct.pen_angvel,
                ),
                dim=-1,
            ).detach().cpu().numpy()
            state_phase = (
                direct.chain_phase_ids.detach().cpu().numpy()
            )
            state_penetration = (
                direct.episode_max_penetration.detach().cpu().numpy()
            )
            if trace_env is None:
                pending_qpos.append(state_qpos.copy())
                pending_qvel.append(state_qvel.copy())
                pending_phase.append(state_phase.copy())
                pending_penetration.append(state_penetration.copy())
                entered_trace_target = transitioned & (
                    phase == int(args.trace_transition_source)
                )
                if torch.any(entered_trace_target):
                    trace_env = int(
                        torch.nonzero(
                            entered_trace_target, as_tuple=False
                        )[0].item()
                    )
                    trace_qpos = [
                        row[trace_env] for row in pending_qpos
                    ]
                    trace_qvel = [
                        row[trace_env] for row in pending_qvel
                    ]
                    trace_phase = [
                        int(row[trace_env]) for row in pending_phase
                    ]
                    trace_penetration = [
                        float(row[trace_env])
                        for row in pending_penetration
                    ]
                    pending_qpos.clear()
                    pending_qvel.clear()
                    pending_phase.clear()
                    pending_penetration.clear()
            else:
                trace_qpos.append(state_qpos[trace_env].copy())
                trace_qvel.append(state_qvel[trace_env].copy())
                trace_phase.append(int(state_phase[trace_env]))
                trace_penetration.append(
                    float(state_penetration[trace_env])
                )
                trace_finished = bool(done[trace_env].item())
        previous_phase = phase

    if args.trajectory_output is not None and trace_env is None:
        trace_env = int(torch.argmax(max_phase).item())
        trace_qpos = [row[trace_env] for row in pending_qpos]
        trace_qvel = [row[trace_env] for row in pending_qvel]
        trace_phase = [
            int(row[trace_env]) for row in pending_phase
        ]
        trace_penetration = [
            float(row[trace_env]) for row in pending_penetration
        ]

    transition_count_cpu = transition_counts.cpu().numpy()
    mean_jump = np.divide(
        action_jump_sum.cpu().numpy(),
        np.maximum(transition_count_cpu, 1),
    )
    result = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "completed_episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "drop_count": drops,
        "drop_rate": drops / max(completed, 1),
        "transition_counts": transition_count_cpu.tolist(),
        "mean_transition_action_jump": mean_jump.tolist(),
        "max_transition_action_jump": (
            action_jump_max.cpu().numpy().tolist()
        ),
        "deepest_phase": global_deepest_phase,
        "max_penetration_mm": max_penetration_m * 1000.0,
        "handoff": "same PhysX episode; no hand or pen state reset",
        "blend_steps": int(cfg.chain_blend_steps),
        "library_phases": sorted(libraries),
        "phase4_diagnostics": {
            "min_ring_gap_mm": (
                phase4_min_ring_gap_m * 1000.0
                if np.isfinite(phase4_min_ring_gap_m)
                else None
            ),
            "index_ring_contact_env_steps": (
                phase4_ring_contact_steps
            ),
            "min_contact_rmse_rad": (
                phase4_min_contact_rmse_rad
                if np.isfinite(phase4_min_contact_rmse_rad)
                else None
            ),
            "mean_nearest_normalized_distance": (
                phase4_nearest_distance_sum
                / max(phase4_nearest_distance_count, 1)
            ),
            "entry_count": phase4_nearest_distance_count,
            "contact_pose_env_steps": phase4_contact_pose_steps,
            "contact_pose_safe_component_steps": (
                phase4_contact_pose_safe_component_steps.tolist()
            ),
            "safe_component_order": [
                "position",
                "axis",
                "angular_speed",
                "penetration",
                "force",
            ],
            "contact_pose_safe_env_steps": (
                phase4_contact_pose_safe_steps
            ),
            "contact_pose_min_position_error_mm": (
                phase4_contact_pose_min_position_error_m * 1000.0
                if np.isfinite(
                    phase4_contact_pose_min_position_error_m
                )
                else None
            ),
            "contact_pose_min_axis_error_deg": (
                np.rad2deg(phase4_contact_pose_min_axis_error_rad)
                if np.isfinite(phase4_contact_pose_min_axis_error_rad)
                else None
            ),
            "contact_pose_min_angular_speed_rad_s": (
                phase4_contact_pose_min_speed_rad_s
                if np.isfinite(phase4_contact_pose_min_speed_rad_s)
                else None
            ),
            "contact_pose_min_penetration_mm": (
                phase4_contact_pose_min_penetration_m * 1000.0
                if np.isfinite(phase4_contact_pose_min_penetration_m)
                else None
            ),
        },
    }
    if hasattr(direct, "closure_stage_step_counts"):
        result["closure_stage_diagnostics"] = {
            "steps": (
                direct.closure_stage_step_counts.detach()
                .cpu().tolist()
            ),
            "contact_steps": (
                direct.closure_stage_triad_step_counts.detach()
                .cpu().tolist()
            ),
            "contact_pose_steps": (
                direct.closure_stage_contact_pose_step_counts.detach()
                .cpu().tolist()
            ),
            "safe_steps": (
                direct.closure_stage_safe_step_counts.detach()
                .cpu().tolist()
            ),
            "safe_component_steps": (
                direct.closure_stage_safe_component_step_counts.detach()
                .cpu().tolist()
            ),
            "eligible_steps": (
                direct.closure_stage_eligible_step_counts.detach()
                .cpu().tolist()
            ),
            "min_target_rmse_rad": (
                direct.closure_stage_min_target_rmse.detach()
                .cpu().tolist()
            ),
            "best_hold_steps": (
                direct.closure_stage_best_hold.detach()
                .cpu().tolist()
            ),
        }
    output = (
        args.output
        if args.output.is_absolute()
        else ROOT / args.output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if (
        args.phase5_cache_output is not None
        and phase5_cache["source_qpos"]
    ):
        cache = {
            key: np.concatenate(value, axis=0)
            for key, value in phase5_cache.items()
        }
        count = len(cache["source_qpos"])
        cache["source_chain_phase_id"] = np.full(
            count, 5, dtype=np.int64
        )
        cache["target_qpos"] = cache["source_qpos"]
        cache["target_qvel"] = np.zeros_like(cache["source_qvel"])
        cache["phase_ids"] = np.zeros(count, dtype=np.int64)
        cache["source"] = np.asarray(
            "same-episode phase4-to-phase5 stable brace entries",
            dtype=np.str_,
        )
        cache_output = (
            args.phase5_cache_output
            if args.phase5_cache_output.is_absolute()
            else ROOT / args.phase5_cache_output
        )
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_output, **cache)
        result["phase5_cache_output"] = str(cache_output)
        result["phase5_cache_count"] = count
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if (
        args.phase6_cache_output is not None
        and phase6_cache["source_qpos"]
    ):
        cache = {
            key: np.concatenate(value, axis=0)
            for key, value in phase6_cache.items()
        }
        count = len(cache["source_qpos"])
        cache["source_chain_phase_id"] = np.full(
            count, 6, dtype=np.int64
        )
        cache["target_qpos"] = cache["source_qpos"]
        cache["target_qvel"] = np.zeros_like(cache["source_qvel"])
        cache["phase_ids"] = np.zeros(count, dtype=np.int64)
        cache["source"] = np.asarray(
            "same-episode phase5-to-phase6 thumb-under entries",
            dtype=np.str_,
        )
        cache_output = (
            args.phase6_cache_output
            if args.phase6_cache_output.is_absolute()
            else ROOT / args.phase6_cache_output
        )
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_output, **cache)
        result["phase6_cache_output"] = str(cache_output)
        result["phase6_cache_count"] = count
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if (
        args.phase7_cache_output is not None
        and phase7_cache["source_qpos"]
    ):
        cache = {
            key: np.concatenate(value, axis=0)
            for key, value in phase7_cache.items()
        }
        count = len(cache["source_qpos"])
        cache["source_chain_phase_id"] = np.full(
            count, 7, dtype=np.int64
        )
        cache["target_qpos"] = cache["source_qpos"]
        cache["target_qvel"] = np.zeros_like(cache["source_qvel"])
        cache["phase_ids"] = np.zeros(count, dtype=np.int64)
        cache["source"] = np.asarray(
            "same-episode phase6-to-phase7 thumb-contact entries",
            dtype=np.str_,
        )
        cache_output = (
            args.phase7_cache_output
            if args.phase7_cache_output.is_absolute()
            else ROOT / args.phase7_cache_output
        )
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_output, **cache)
        result["phase7_cache_output"] = str(cache_output)
        result["phase7_cache_count"] = count
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if (
        args.phase8_cache_output is not None
        and phase8_cache["source_qpos"]
    ):
        cache = {
            key: np.concatenate(value, axis=0)
            for key, value in phase8_cache.items()
        }
        count = len(cache["source_qpos"])
        cache["source_chain_phase_id"] = np.full(
            count, 8, dtype=np.int64
        )
        cache["target_qpos"] = cache["source_qpos"]
        cache["target_qvel"] = np.zeros_like(cache["source_qvel"])
        cache["phase_ids"] = np.zeros(count, dtype=np.int64)
        cache["source"] = np.asarray(
            "same-episode true phase7-to-phase8 thumb-under entries",
            dtype=np.str_,
        )
        cache_output = (
            args.phase8_cache_output
            if args.phase8_cache_output.is_absolute()
            else ROOT / args.phase8_cache_output
        )
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_output, **cache)
        result["phase8_cache_output"] = str(cache_output)
        result["phase8_cache_count"] = count
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.trajectory_output is not None:
        trajectory_output = (
            args.trajectory_output
            if args.trajectory_output.is_absolute()
            else ROOT / args.trajectory_output
        )
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_output,
            qpos=np.asarray(trace_qpos, dtype=np.float32),
            qvel=np.asarray(trace_qvel, dtype=np.float32),
            phase_ids=np.asarray(trace_phase, dtype=np.int64),
            max_penetration_m=np.asarray(
                trace_penetration, dtype=np.float32
            ),
            control_dt_s=np.asarray(
                cfg.decimation * cfg.sim.dt, dtype=np.float32
            ),
            trajectory_kind=np.asarray(
                "phase_conditioned_exemplar_teacher", dtype=np.str_
            ),
            deepest_phase=np.asarray(
                max(trace_phase, default=0), dtype=np.int64
            ),
        )
        result["trajectory_output"] = str(trajectory_output)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
