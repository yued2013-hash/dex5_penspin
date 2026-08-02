from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass

from .canonical_reset_physics_env import (
    Dex5CanonicalResetPhysicsEnv,
    Dex5CanonicalResetPhysicsEnvCfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ORACLE_BANK = PROJECT_ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
)
FROZEN_BANK_SHA256 = "73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65"
CANONICAL_IDS = ("P1", "P2", "P3", "P4", "P5")
FAILURE_TYPES = (
    "contact_loss",
    "ballistic",
    "penetration",
    "throw",
    "drop",
    "drift",
    "invalid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@configclass
class Dex5PrivilegedOraclePhysicsEnvCfg(Dex5CanonicalResetPhysicsEnvCfg):
    """O1 privileged-oracle reachability environment."""

    episode_length_s = 2.0
    action_scale = 0.025
    observation_space = 152
    bundle_path = str(DEFAULT_ORACLE_BANK)
    expected_bundle_sha256 = FROZEN_BANK_SHA256
    reset_split = "train"
    reset_mixture = (0.40, 0.20, 0.20, 0.15, 0.05)

    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    canonical_reset_pen_rot_noise_rad = 0.0

    penetration_threshold = 0.0010
    drop_threshold = 0.040
    drift_threshold = 0.080
    no_contact_termination_steps = 4
    throw_linear_speed_m_s = 1.0
    throw_angular_speed_rad_s = 60.0
    ballistic_no_contact_steps = 2
    ballistic_linear_speed_m_s = 0.20
    ballistic_angular_speed_rad_s = 12.0
    terminate_on_ballistic = False
    axis_alignment_min_angular_speed_rad_s = 0.50
    axis_alignment_episode_threshold = 0.70
    contact_force_threshold_n = 0.03
    contact_debounce_steps = 3
    target_rotation_direction = 1.0

    rotation_reward_scale = 0.12
    horizontal_reward_scale = 0.15
    linear_speed_penalty_scale = 0.15
    wrong_axis_penalty_scale = 0.03
    slip_penalty_scale = 0.05
    axis_aligned_rotation_reward_scale = 0.0
    axis_alignment_quality_reward_scale = 0.0
    ballistic_penalty_scale = 0.0
    high_angular_speed_penalty_scale = 0.0
    height_drop_penalty_scale = 0.0
    contact_gate_positive_rotation = False
    include_gait_history_observation = False
    include_temporal_support_observation = False
    temporal_support_ema_alpha = 0.20
    support_handoff_min_overlap_ratio = 0.80
    support_handoff_max_zero_tim_steps = 2
    support_target_force_n = 0.08
    support_margin_penalty_scale = 0.0
    support_drop_penalty_scale = 0.0
    support_force_excess_threshold_n = 6.0
    support_force_excess_penalty_scale = 0.0
    recovery_window_steps = 12
    recovery_height_deficit_penalty_scale = 0.0
    recovery_downward_speed_penalty_scale = 0.0
    recovery_progress_reward_scale = 0.0
    termination_mode = "early"
    terminal_failure_penalty_scale = 0.0
    failure_grace_steps = 12
    height_drop_diagnostic_m = 0.005
    height_recovery_diagnostic_m = 0.002


@configclass
class Dex5PrivilegedOracleAlignedCfg(Dex5PrivilegedOraclePhysicsEnvCfg):
    """O1.2 objective alignment shared by the reset-mixture experiments."""

    # Penalize brief flight during learning so the policy can still discover recovery.
    # Deterministic evaluation can independently enable the hard safety guard.
    terminate_on_ballistic = False
    contact_gate_positive_rotation = True
    axis_aligned_rotation_reward_scale = 0.10
    ballistic_penalty_scale = 2.0
    linear_speed_penalty_scale = 0.75
    wrong_axis_penalty_scale = 0.12
    high_angular_speed_penalty_scale = 0.05


@configclass
class Dex5PrivilegedOracleAlignedMixtureACfg(Dex5PrivilegedOracleAlignedCfg):
    reset_mixture = (0.50, 0.20, 0.30, 0.00, 0.00)


@configclass
class Dex5PrivilegedOracleAlignedMixtureBCfg(Dex5PrivilegedOracleAlignedCfg):
    reset_mixture = (0.30, 0.20, 0.40, 0.10, 0.00)


@configclass
class Dex5PrivilegedOracleAlignedP3OnlyCfg(Dex5PrivilegedOracleAlignedCfg):
    reset_mixture = (0.00, 0.00, 1.00, 0.00, 0.00)


@configclass
class Dex5PrivilegedOracleAlignedP3AxisCfg(
    Dex5PrivilegedOracleAlignedP3OnlyCfg
):
    """P3-only axis and height ablation after the O1.2 recovery smoke."""

    axis_aligned_rotation_reward_scale = 0.20
    axis_alignment_quality_reward_scale = 0.30
    wrong_axis_penalty_scale = 0.20
    height_drop_penalty_scale = 20.0


@configclass
class Dex5PrivilegedOracleP3StateCfg(Dex5PrivilegedOracleAlignedP3OnlyCfg):
    """P3-only O1.3 state augmentation with the O1.2 reward frozen."""

    observation_space = 158
    include_gait_history_observation = True


@configclass
class Dex5PrivilegedOracleP3TemporalCfg(Dex5PrivilegedOracleAlignedP3OnlyCfg):
    """O1.5 generic temporal support state with the O1.2 reward frozen."""

    observation_space = 172
    include_temporal_support_observation = True


@configclass
class Dex5PrivilegedOracleP3EntryCurriculumCfg(
    Dex5PrivilegedOracleP3TemporalCfg
):
    """O1.22 early reset curriculum with the O1.5 contract frozen."""

    curriculum_candidate_bank_ids = (1034, 1176, 1146, 1045, 1227, 852, 1149)
    curriculum_candidate_probability_start = 0.75
    curriculum_hold_control_steps = 480
    curriculum_decay_end_control_steps = 960


@configclass
class Dex5PrivilegedOracleP3SupportMarginCfg(Dex5PrivilegedOracleP3TemporalCfg):
    """O1.6-B weak sequence-agnostic support-margin objective."""

    support_margin_penalty_scale = 2.0
    support_drop_penalty_scale = 0.5
    support_force_excess_penalty_scale = 0.01


@configclass
class Dex5PrivilegedOracleP3RecoverabilityCfg(
    Dex5PrivilegedOracleP3SupportMarginCfg
):
    """O1.6-C support margin plus a finite height-recovery window."""

    recovery_height_deficit_penalty_scale = 20.0
    recovery_downward_speed_penalty_scale = 0.5
    recovery_progress_reward_scale = 10.0


@configclass
class Dex5PrivilegedOracleP3FixedHorizonCfg(
    Dex5PrivilegedOracleP3TemporalCfg
):
    """O1.7-A fixed horizon with the O1.5 baseline reward unchanged."""

    termination_mode = "fixed_horizon"


@configclass
class Dex5PrivilegedOracleP3FixedHorizonTerminalCfg(
    Dex5PrivilegedOracleP3FixedHorizonCfg
):
    """O1.7-B fixed horizon plus one terminal penalty for any failure."""

    terminal_failure_penalty_scale = 20.0


@configclass
class Dex5PrivilegedOracleP3RecoveryGraceCfg(
    Dex5PrivilegedOracleP3TemporalCfg
):
    """O1.7-C allow transient physical failures to recover for 200 ms."""

    termination_mode = "recovery_grace"
    failure_grace_steps = 12


class Dex5PrivilegedOraclePhysicsEnv(Dex5CanonicalResetPhysicsEnv):
    """Privileged PPO environment for emergent pen rotation and finger gait."""

    cfg: Dex5PrivilegedOraclePhysicsEnvCfg

    def _load_bundle(self) -> None:
        bank_path = Path(self.cfg.bundle_path).resolve()
        actual_hash = _sha256(bank_path)
        if actual_hash != self.cfg.expected_bundle_sha256:
            raise RuntimeError(
                "O1 frozen-bank hash mismatch: "
                f"expected {self.cfg.expected_bundle_sha256}, got {actual_hash}"
            )
        super()._load_bundle()
        with np.load(bank_path, allow_pickle=False) as data:
            split_bank = np.asarray(data["metadata_split"])
            contact_bank = np.asarray(data["contact_state"], dtype=np.bool_)

        canonical_to_index = {name: index for index, name in enumerate(CANONICAL_IDS)}
        canonical_indices = np.asarray(
            [canonical_to_index[str(value)] for value in self.canonical_id_bank],
            dtype=np.int64,
        )
        self.canonical_index_bank = torch.as_tensor(
            canonical_indices, dtype=torch.long, device=self.device
        )
        self.source_contact_state_bank = torch.as_tensor(
            contact_bank, dtype=torch.bool, device=self.device
        )
        self.reset_rows_by_canonical: list[torch.Tensor] = []
        configured_mixture = np.asarray(self.cfg.reset_mixture, dtype=np.float64)
        for canonical_index, canonical_id in enumerate(CANONICAL_IDS):
            rows = np.flatnonzero(
                (canonical_indices == canonical_index)
                & (split_bank == self.cfg.reset_split)
            )
            if rows.size == 0 and configured_mixture[canonical_index] > 0.0:
                raise RuntimeError(
                    f"Frozen bank has no {canonical_id} rows in split={self.cfg.reset_split}"
                )
            self.reset_rows_by_canonical.append(
                torch.as_tensor(rows, dtype=torch.long, device=self.device)
            )

        configured_candidates = tuple(
            int(value)
            for value in getattr(self.cfg, "curriculum_candidate_bank_ids", ())
        )
        self.curriculum_candidate_rows = torch.as_tensor(
            configured_candidates, dtype=torch.long, device=self.device
        )
        if self.curriculum_candidate_rows.numel():
            if torch.any(self.curriculum_candidate_rows < 0) or torch.any(
                self.curriculum_candidate_rows >= self.bank_size
            ):
                raise ValueError("curriculum candidate bank ID is out of range")
            if torch.any(
                self.canonical_index_bank[self.curriculum_candidate_rows]
                != CANONICAL_IDS.index("P3")
            ):
                raise ValueError("all curriculum candidates must be frozen P3 rows")
            p3_rows = self.reset_rows_by_canonical[CANONICAL_IDS.index("P3")]
            if not bool(
                torch.all(torch.isin(self.curriculum_candidate_rows, p3_rows)).item()
            ):
                raise ValueError("curriculum candidates must belong to the P3 train split")
            if self.curriculum_candidate_rows.unique().numel() != len(
                configured_candidates
            ):
                raise ValueError("curriculum candidate bank IDs must be unique")
            self.curriculum_non_candidate_rows = p3_rows[
                ~torch.isin(p3_rows, self.curriculum_candidate_rows)
            ]
            if not self.curriculum_non_candidate_rows.numel():
                raise ValueError("curriculum requires non-candidate P3 rows")
        else:
            self.curriculum_non_candidate_rows = torch.empty(
                0, dtype=torch.long, device=self.device
            )

        mixture = torch.as_tensor(
            self.cfg.reset_mixture, dtype=torch.float32, device=self.device
        )
        if mixture.shape != (len(CANONICAL_IDS),) or torch.any(mixture < 0.0):
            raise ValueError("reset_mixture must contain five non-negative weights")
        self.reset_mixture = mixture / mixture.sum().clamp_min(1.0e-9)
        self.canonical_indices_env = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.no_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.debounced_contacts = torch.zeros(
            (self.num_envs, 5), dtype=torch.bool, device=self.device
        )
        self.contact_candidates = torch.zeros_like(self.debounced_contacts)
        self.contact_candidate_steps = torch.zeros(
            (self.num_envs, 5), dtype=torch.long, device=self.device
        )
        self.episode_release_seen = torch.zeros_like(self.debounced_contacts)
        self.episode_recontact_seen = torch.zeros_like(self.debounced_contacts)
        self.episode_release_count = torch.zeros(
            (self.num_envs, 5), dtype=torch.long, device=self.device
        )
        self.episode_recontact_count = torch.zeros_like(self.episode_release_count)
        self.episode_thumb_attempt_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_first_thumb_break_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.episode_first_thumb_attempt_step = torch.full_like(
            self.episode_first_thumb_break_step, -1
        )
        self.episode_first_thumb_recontact_step = torch.full_like(
            self.episode_first_thumb_break_step, -1
        )
        self.episode_min_height_delta = torch.zeros(self.num_envs, device=self.device)
        self.episode_height_drop_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_height_recovered_seen = torch.zeros_like(
            self.episode_height_drop_seen
        )
        self.episode_first_height_drop_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.episode_tim_overlap_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_zero_tim_run_steps = torch.zeros_like(
            self.episode_tim_overlap_steps
        )
        self.episode_max_zero_tim_run_steps = torch.zeros_like(
            self.episode_tim_overlap_steps
        )
        self.episode_target_rotation = torch.zeros(self.num_envs, device=self.device)
        self.episode_peak_target_rotation = torch.zeros(
            self.num_envs, device=self.device
        )
        self.episode_axis_alignment_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self.episode_absolute_axis_alignment_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self.episode_axis_alignment_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_ballistic_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_ballistic_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_max_no_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_failure_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_failure_type_seen = torch.zeros(
            (self.num_envs, len(FAILURE_TYPES)),
            dtype=torch.bool,
            device=self.device,
        )
        self.episode_primary_failure = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.failure_grace_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.completed_gait_events = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_episode_length_sum = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_over_180deg = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_failures = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_throws = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_contact_losses = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_penetrations = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_ballistic = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_by_canonical = torch.zeros(
            len(CANONICAL_IDS), dtype=torch.long, device=self.device
        )
        self.completed_episode_length_sum_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_gait_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_over_180deg_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_failure_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_timeout_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_axis_aligned_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_qualified_gait_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_qualified_over_180deg_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_qualified_gait = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_qualified_over_180deg = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_axis_alignment_sum_by_canonical = torch.zeros(
            len(CANONICAL_IDS), device=self.device
        )
        self.completed_absolute_axis_alignment_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_max_no_contact_steps_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_ballistic_steps_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_thumb_release_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_thumb_attempt_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_thumb_recontact_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_height_drop_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_height_recovery_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_support_handoff = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_support_handoff_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_tim_gait = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_tim_gait_by_canonical = torch.zeros_like(
            self.completed_by_canonical
        )
        self.completed_tim_overlap_ratio_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_max_zero_tim_run_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_min_height_delta_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_thumb_break_step_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_thumb_attempt_step_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_thumb_recontact_step_sum_by_canonical = torch.zeros_like(
            self.completed_axis_alignment_sum_by_canonical
        )
        self.completed_release_episode_by_canonical_finger = torch.zeros(
            (len(CANONICAL_IDS), 5), dtype=torch.long, device=self.device
        )
        self.completed_recontact_episode_by_canonical_finger = torch.zeros_like(
            self.completed_release_episode_by_canonical_finger
        )
        self.completed_gait_episode_by_canonical_finger = torch.zeros_like(
            self.completed_release_episode_by_canonical_finger
        )
        self.completed_release_events_by_canonical_finger = torch.zeros_like(
            self.completed_release_episode_by_canonical_finger
        )
        self.completed_recontact_events_by_canonical_finger = torch.zeros_like(
            self.completed_release_episode_by_canonical_finger
        )
        self.completed_failure_type_by_canonical = torch.zeros(
            (len(CANONICAL_IDS), len(FAILURE_TYPES)),
            dtype=torch.long,
            device=self.device,
        )
        self.completed_primary_failure_by_canonical = torch.zeros_like(
            self.completed_failure_type_by_canonical
        )
        self.sampled_reset_counts = torch.zeros(
            len(CANONICAL_IDS), dtype=torch.long, device=self.device
        )
        self.sampled_curriculum_candidate_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.sampled_curriculum_non_candidate_count = torch.zeros_like(
            self.sampled_curriculum_candidate_count
        )
        self.current_failed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_throw = torch.zeros_like(self.current_failed)
        self.current_contact_loss = torch.zeros_like(self.current_failed)
        self.current_penetration_failure = torch.zeros_like(self.current_failed)
        self.current_ballistic = torch.zeros_like(self.current_failed)
        self.current_drop = torch.zeros_like(self.current_failed)
        self.current_drift = torch.zeros_like(self.current_failed)
        self.current_invalid = torch.zeros_like(self.current_failed)
        self.current_timeout = torch.zeros_like(self.current_failed)
        self.current_primary_failure = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.temporal_contact_duty = torch.zeros(
            (self.num_envs, 5), device=self.device
        )
        self.temporal_force_ema = torch.zeros_like(self.temporal_contact_duty)
        self.temporal_force_delta = torch.zeros_like(self.temporal_contact_duty)
        self.temporal_previous_force = torch.zeros_like(self.temporal_contact_duty)
        self.temporal_height_ema = torch.zeros(self.num_envs, device=self.device)
        self.temporal_height_delta = torch.zeros_like(self.temporal_height_ema)
        self.temporal_previous_height = torch.zeros_like(self.temporal_height_ema)
        self.temporal_support_ema = torch.zeros_like(self.temporal_height_ema)
        self.temporal_support_delta = torch.zeros_like(self.temporal_height_ema)
        self.temporal_previous_support = torch.zeros_like(self.temporal_height_ema)

    def _curriculum_candidate_probability(self) -> float:
        if not self.curriculum_candidate_rows.numel():
            return 0.0
        p3_rows = self.reset_rows_by_canonical[CANONICAL_IDS.index("P3")]
        natural = self.curriculum_candidate_rows.numel() / p3_rows.numel()
        start = float(self.cfg.curriculum_candidate_probability_start)
        hold = int(self.cfg.curriculum_hold_control_steps)
        decay_end = int(self.cfg.curriculum_decay_end_control_steps)
        step = int(self.common_step_counter)
        if step <= hold:
            return start
        if step >= decay_end:
            return natural
        fraction = (step - hold) / max(decay_end - hold, 1)
        return start + fraction * (natural - start)

    def _sample_mixture_bank_ids(self, count: int) -> torch.Tensor:
        canonical_indices = torch.multinomial(
            self.reset_mixture, count, replacement=True
        )
        bank_ids = torch.empty(count, dtype=torch.long, device=self.device)
        for canonical_index, rows in enumerate(self.reset_rows_by_canonical):
            mask = canonical_indices == canonical_index
            row_count = int(mask.sum().item())
            if row_count:
                if (
                    canonical_index == CANONICAL_IDS.index("P3")
                    and self.curriculum_candidate_rows.numel()
                ):
                    probability = self._curriculum_candidate_probability()
                    use_candidate = (
                        torch.rand(row_count, device=self.device) < probability
                    )
                    selected = torch.empty(
                        row_count, dtype=torch.long, device=self.device
                    )
                    candidate_count = int(use_candidate.sum().item())
                    other_count = row_count - candidate_count
                    if candidate_count:
                        choices = torch.randint(
                            0,
                            self.curriculum_candidate_rows.numel(),
                            (candidate_count,),
                            device=self.device,
                        )
                        selected[use_candidate] = self.curriculum_candidate_rows[
                            choices
                        ]
                    if other_count:
                        choices = torch.randint(
                            0,
                            self.curriculum_non_candidate_rows.numel(),
                            (other_count,),
                            device=self.device,
                        )
                        selected[~use_candidate] = self.curriculum_non_candidate_rows[
                            choices
                        ]
                    bank_ids[mask] = selected
                    self.sampled_curriculum_candidate_count += candidate_count
                    self.sampled_curriculum_non_candidate_count += other_count
                else:
                    choices = torch.randint(
                        0, rows.numel(), (row_count,), device=self.device
                    )
                    bank_ids[mask] = rows[choices]
        return bank_ids

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )

        if hasattr(self, "episode_release_seen"):
            completed = self.episode_length_buf[env_ids_tensor] > 0
            old_canonical = self.canonical_indices_env[env_ids_tensor]
            gait_by_finger = (
                self.episode_release_seen[env_ids_tensor]
                & self.episode_recontact_seen[env_ids_tensor]
            )
            gait = gait_by_finger.any(dim=-1)
            tim_gait = gait_by_finger[:, :3].any(dim=-1)
            alignment_steps = self.episode_axis_alignment_steps[
                env_ids_tensor
            ].clamp_min(1)
            axis_alignment = self.episode_axis_alignment_sum[
                env_ids_tensor
            ] / alignment_steps
            absolute_axis_alignment = self.episode_absolute_axis_alignment_sum[
                env_ids_tensor
            ] / alignment_steps
            axis_aligned = (
                axis_alignment >= self.cfg.axis_alignment_episode_threshold
            ) & (self.episode_axis_alignment_steps[env_ids_tensor] > 0)
            stable_episode = (
                completed
                & axis_aligned
                & ~self.episode_ballistic_seen[env_ids_tensor]
                & ~self.episode_failure_type_seen[env_ids_tensor, 0]
                & ~self.episode_failure_seen[env_ids_tensor]
            )
            qualified_gait = stable_episode & gait
            episode_steps = self.episode_length_buf[env_ids_tensor].clamp_min(1)
            tim_overlap_ratio = (
                self.episode_tim_overlap_steps[env_ids_tensor].float()
                / episode_steps.float()
            )
            support_handoff = (
                completed
                & tim_gait
                & (
                    tim_overlap_ratio
                    >= self.cfg.support_handoff_min_overlap_ratio
                )
                & (
                    self.episode_max_zero_tim_run_steps[env_ids_tensor]
                    <= self.cfg.support_handoff_max_zero_tim_steps
                )
                & (
                    ~self.episode_height_drop_seen[env_ids_tensor]
                    | self.episode_height_recovered_seen[env_ids_tensor]
                )
                & ~self.episode_ballistic_seen[env_ids_tensor]
                & ~self.episode_failure_seen[env_ids_tensor]
            )
            qualified_over_180deg = stable_episode & (
                self.episode_peak_target_rotation[env_ids_tensor] > torch.pi
            )
            self.completed_episodes += completed.sum()
            self.completed_episode_length_sum += torch.where(
                completed, episode_steps, 0
            ).sum()
            self.completed_gait_events += (completed & gait).sum()
            self.completed_over_180deg += (
                completed
                & (self.episode_peak_target_rotation[env_ids_tensor] > torch.pi)
            ).sum()
            self.completed_failures += (
                completed & self.episode_failure_seen[env_ids_tensor]
            ).sum()
            self.completed_throws += (
                completed & self.episode_failure_type_seen[env_ids_tensor, 3]
            ).sum()
            self.completed_contact_losses += (
                completed & self.episode_failure_type_seen[env_ids_tensor, 0]
            ).sum()
            self.completed_penetrations += (
                completed & self.episode_failure_type_seen[env_ids_tensor, 2]
            ).sum()
            self.completed_ballistic += (
                completed & self.episode_ballistic_seen[env_ids_tensor]
            ).sum()
            self.completed_qualified_gait += qualified_gait.sum()
            self.completed_qualified_over_180deg += qualified_over_180deg.sum()
            self.completed_support_handoff += support_handoff.sum()
            self.completed_tim_gait += (completed & tim_gait).sum()

            completed_long = completed.long()
            self.completed_by_canonical.index_add_(
                0, old_canonical, completed_long
            )
            self.completed_episode_length_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(completed, episode_steps, 0),
            )
            self.completed_gait_by_canonical.index_add_(
                0, old_canonical, (completed & gait).long()
            )
            self.completed_over_180deg_by_canonical.index_add_(
                0,
                old_canonical,
                (
                    completed
                    & (self.episode_peak_target_rotation[env_ids_tensor] > torch.pi)
                ).long(),
            )
            self.completed_failure_by_canonical.index_add_(
                0,
                old_canonical,
                (completed & self.episode_failure_seen[env_ids_tensor]).long(),
            )
            self.completed_timeout_by_canonical.index_add_(
                0, old_canonical, (completed & self.current_timeout[env_ids_tensor]).long()
            )
            self.completed_axis_aligned_by_canonical.index_add_(
                0, old_canonical, (completed & axis_aligned).long()
            )
            self.completed_qualified_gait_by_canonical.index_add_(
                0, old_canonical, qualified_gait.long()
            )
            self.completed_qualified_over_180deg_by_canonical.index_add_(
                0, old_canonical, qualified_over_180deg.long()
            )
            self.completed_axis_alignment_sum_by_canonical.index_add_(
                0, old_canonical, torch.where(completed, axis_alignment, 0.0)
            )
            self.completed_absolute_axis_alignment_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(completed, absolute_axis_alignment, 0.0),
            )
            self.completed_max_no_contact_steps_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(
                    completed,
                    self.episode_max_no_contact_steps[env_ids_tensor].float(),
                    0.0,
                ),
            )
            self.completed_ballistic_steps_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(
                    completed,
                    self.episode_ballistic_steps[env_ids_tensor].float(),
                    0.0,
                ),
            )
            self.completed_thumb_release_by_canonical.index_add_(
                0,
                old_canonical,
                (completed & self.episode_release_seen[env_ids_tensor, 0]).long(),
            )
            self.completed_thumb_attempt_by_canonical.index_add_(
                0,
                old_canonical,
                (completed & self.episode_thumb_attempt_seen[env_ids_tensor]).long(),
            )
            self.completed_thumb_recontact_by_canonical.index_add_(
                0,
                old_canonical,
                (completed & self.episode_recontact_seen[env_ids_tensor, 0]).long(),
            )
            self.completed_height_drop_by_canonical.index_add_(
                0,
                old_canonical,
                (completed & self.episode_height_drop_seen[env_ids_tensor]).long(),
            )
            self.completed_height_recovery_by_canonical.index_add_(
                0,
                old_canonical,
                (completed & self.episode_height_recovered_seen[env_ids_tensor]).long(),
            )
            self.completed_support_handoff_by_canonical.index_add_(
                0, old_canonical, support_handoff.long()
            )
            self.completed_tim_gait_by_canonical.index_add_(
                0, old_canonical, (completed & tim_gait).long()
            )
            self.completed_tim_overlap_ratio_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(completed, tim_overlap_ratio, 0.0),
            )
            self.completed_max_zero_tim_run_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(
                    completed,
                    self.episode_max_zero_tim_run_steps[env_ids_tensor].float(),
                    0.0,
                ),
            )
            self.completed_min_height_delta_sum_by_canonical.index_add_(
                0,
                old_canonical,
                torch.where(
                    completed,
                    self.episode_min_height_delta[env_ids_tensor],
                    0.0,
                ),
            )
            for target, values in (
                (
                    self.completed_thumb_break_step_sum_by_canonical,
                    self.episode_first_thumb_break_step[env_ids_tensor],
                ),
                (
                    self.completed_thumb_attempt_step_sum_by_canonical,
                    self.episode_first_thumb_attempt_step[env_ids_tensor],
                ),
                (
                    self.completed_thumb_recontact_step_sum_by_canonical,
                    self.episode_first_thumb_recontact_step[env_ids_tensor],
                ),
            ):
                target.index_add_(
                    0,
                    old_canonical,
                    torch.where(completed & (values >= 0), values.float(), 0.0),
                )
            release_seen = self.episode_release_seen[env_ids_tensor]
            recontact_seen = self.episode_recontact_seen[env_ids_tensor]
            for canonical_index in range(len(CANONICAL_IDS)):
                canonical_mask = completed & (old_canonical == canonical_index)
                if torch.any(canonical_mask):
                    self.completed_release_episode_by_canonical_finger[
                        canonical_index
                    ] += (release_seen & canonical_mask.unsqueeze(-1)).sum(dim=0)
                    self.completed_recontact_episode_by_canonical_finger[
                        canonical_index
                    ] += (recontact_seen & canonical_mask.unsqueeze(-1)).sum(dim=0)
                    self.completed_gait_episode_by_canonical_finger[
                        canonical_index
                    ] += (gait_by_finger & canonical_mask.unsqueeze(-1)).sum(dim=0)
                    self.completed_release_events_by_canonical_finger[
                        canonical_index
                    ] += torch.where(
                        canonical_mask.unsqueeze(-1),
                        self.episode_release_count[env_ids_tensor],
                        0,
                    ).sum(dim=0)
                    self.completed_recontact_events_by_canonical_finger[
                        canonical_index
                    ] += torch.where(
                        canonical_mask.unsqueeze(-1),
                        self.episode_recontact_count[env_ids_tensor],
                        0,
                    ).sum(dim=0)

            failure_flags = self.episode_failure_type_seen[
                env_ids_tensor
            ] & completed.unsqueeze(-1)
            for canonical_index in range(len(CANONICAL_IDS)):
                canonical_mask = old_canonical == canonical_index
                self.completed_failure_type_by_canonical[canonical_index] += (
                    failure_flags & canonical_mask.unsqueeze(-1)
                ).sum(dim=0)
                primary = self.episode_primary_failure[env_ids_tensor]
                for failure_index in range(len(FAILURE_TYPES)):
                    self.completed_primary_failure_by_canonical[
                        canonical_index, failure_index
                    ] += (
                        completed
                        & canonical_mask
                        & (primary == failure_index)
                    ).sum()

        injected_ids = False
        if int(self.cfg.fixed_reset_bank_id) < 0 and not hasattr(self, "forced_bank_ids"):
            self.forced_bank_ids = self._sample_mixture_bank_ids(len(env_ids_tensor))
            injected_ids = True
        try:
            super()._reset_idx(env_ids_tensor)
        finally:
            if injected_ids:
                del self.forced_bank_ids

        bank_ids = self.current_bank_ids[env_ids_tensor]
        canonical_indices = self.canonical_index_bank[bank_ids]
        self.canonical_indices_env[env_ids_tensor] = canonical_indices
        self.sampled_reset_counts += torch.bincount(
            canonical_indices, minlength=len(CANONICAL_IDS)
        )
        initial_contacts = self.source_contact_state_bank[bank_ids]
        self.debounced_contacts[env_ids_tensor] = initial_contacts
        self.contact_candidates[env_ids_tensor] = initial_contacts
        self.contact_candidate_steps[env_ids_tensor] = 0
        self.episode_release_seen[env_ids_tensor] = False
        self.episode_recontact_seen[env_ids_tensor] = False
        self.episode_release_count[env_ids_tensor] = 0
        self.episode_recontact_count[env_ids_tensor] = 0
        self.episode_thumb_attempt_seen[env_ids_tensor] = False
        self.episode_first_thumb_break_step[env_ids_tensor] = -1
        self.episode_first_thumb_attempt_step[env_ids_tensor] = -1
        self.episode_first_thumb_recontact_step[env_ids_tensor] = -1
        self.episode_min_height_delta[env_ids_tensor] = 0.0
        self.episode_height_drop_seen[env_ids_tensor] = False
        self.episode_height_recovered_seen[env_ids_tensor] = False
        self.episode_first_height_drop_step[env_ids_tensor] = -1
        self.episode_tim_overlap_steps[env_ids_tensor] = 0
        self.episode_zero_tim_run_steps[env_ids_tensor] = 0
        self.episode_max_zero_tim_run_steps[env_ids_tensor] = 0
        self.no_contact_steps[env_ids_tensor] = 0
        self.episode_target_rotation[env_ids_tensor] = 0.0
        self.episode_peak_target_rotation[env_ids_tensor] = 0.0
        self.episode_axis_alignment_sum[env_ids_tensor] = 0.0
        self.episode_absolute_axis_alignment_sum[env_ids_tensor] = 0.0
        self.episode_axis_alignment_steps[env_ids_tensor] = 0
        self.episode_ballistic_seen[env_ids_tensor] = False
        self.episode_ballistic_steps[env_ids_tensor] = 0
        self.episode_max_no_contact_steps[env_ids_tensor] = 0
        self.episode_failure_seen[env_ids_tensor] = False
        self.episode_failure_type_seen[env_ids_tensor] = False
        self.episode_primary_failure[env_ids_tensor] = -1
        self.failure_grace_counter[env_ids_tensor] = 0
        self.current_failed[env_ids_tensor] = False
        self.current_throw[env_ids_tensor] = False
        self.current_contact_loss[env_ids_tensor] = False
        self.current_penetration_failure[env_ids_tensor] = False
        self.current_ballistic[env_ids_tensor] = False
        self.current_drop[env_ids_tensor] = False
        self.current_drift[env_ids_tensor] = False
        self.current_invalid[env_ids_tensor] = False
        self.current_timeout[env_ids_tensor] = False
        self.current_primary_failure[env_ids_tensor] = -1
        self.temporal_contact_duty[env_ids_tensor] = initial_contacts.float()
        self.temporal_force_ema[env_ids_tensor] = 0.0
        self.temporal_force_delta[env_ids_tensor] = 0.0
        self.temporal_previous_force[env_ids_tensor] = 0.0
        self.temporal_height_ema[env_ids_tensor] = 0.0
        self.temporal_height_delta[env_ids_tensor] = 0.0
        self.temporal_previous_height[env_ids_tensor] = 0.0
        self.temporal_support_ema[env_ids_tensor] = 0.0
        self.temporal_support_delta[env_ids_tensor] = 0.0
        self.temporal_previous_support[env_ids_tensor] = 0.0

    def _apply_action(self) -> None:
        self.processed_targets = self.processed_targets + self.cfg.action_scale * self.actions
        self.processed_targets = torch.maximum(
            torch.minimum(self.processed_targets, self.joint_upper), self.joint_lower
        )
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._compute_intermediate_values()
        bank_ids = self.current_bank_ids
        contacts = (
            self.contact_force_norm > self.cfg.contact_force_threshold_n
        ).float()
        phase_one_hot = torch.nn.functional.one_hot(
            self.canonical_indices_env, num_classes=len(CANONICAL_IDS)
        ).float()
        physics = torch.cat(
            (
                (self.mass_bank[bank_ids] / 0.0094).unsqueeze(-1),
                self.static_friction_bank[bank_ids].unsqueeze(-1),
                self.dynamic_friction_bank[bank_ids].unsqueeze(-1),
                100.0 * self.com_offset_bank[bank_ids],
            ),
            dim=-1,
        )
        observation_parts = [
            self.hand_qpos,
            0.2 * self.hand_qvel,
            self.processed_targets,
            self.previous_actions,
            self.pen_pos,
            self.pen_quat,
            self.pen_linvel,
            0.1 * self.pen_angvel,
            self.pen_axis,
            (self.fingertip_pos_local - self.pen_pos.unsqueeze(1)).reshape(
                self.num_envs, -1
            ),
            contacts,
            0.1 * self.contact_force_w.reshape(self.num_envs, -1),
            0.1 * self.contact_force_norm,
            self.contact_tangential_slip_speed,
            physics,
            phase_one_hot,
        ]
        if self.cfg.include_gait_history_observation:
            rotation_phase = self.episode_target_rotation
            relative_height = 20.0 * (self.pen_pos[:, 2] - self.init_pen_pos[:, 2])
            observation_parts.append(
                torch.stack(
                    (
                        torch.sin(rotation_phase),
                        torch.cos(rotation_phase),
                        torch.clamp(rotation_phase / (2.0 * torch.pi), -2.0, 2.0),
                        relative_height,
                        self.episode_release_seen[:, 0].float(),
                        self.episode_recontact_seen[:, 0].float(),
                    ),
                    dim=-1,
                )
            )
        if self.cfg.include_temporal_support_observation:
            relative_height = self.pen_pos[:, 2] - self.init_pen_pos[:, 2]
            support_force_z = -self.contact_force_w.sum(dim=1)[:, 2]
            observation_parts.extend(
                (
                    self.temporal_contact_duty,
                    0.1 * self.temporal_force_ema,
                    0.1 * self.temporal_force_delta,
                    torch.stack(
                        (20.0 * relative_height, 20.0 * self.temporal_height_ema),
                        dim=-1,
                    ),
                    0.1
                    * torch.stack(
                        (
                            support_force_z,
                            self.temporal_support_ema,
                            self.temporal_support_delta,
                        ),
                        dim=-1,
                    ),
                )
            )
        obs = torch.cat(tuple(observation_parts), dim=-1)
        if obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Oracle observation mismatch: {obs.shape[-1]} != {self.cfg.observation_space}"
            )
        return {"policy": obs}

    def _update_contact_transitions(self) -> torch.Tensor:
        raw = self.contact_force_norm > self.cfg.contact_force_threshold_n
        candidate_changed = raw != self.contact_candidates
        self.contact_candidates = torch.where(
            candidate_changed, raw, self.contact_candidates
        )
        self.contact_candidate_steps = torch.where(
            candidate_changed,
            torch.ones_like(self.contact_candidate_steps),
            self.contact_candidate_steps + 1,
        )
        commit = (
            self.contact_candidate_steps >= self.cfg.contact_debounce_steps
        ) & (self.contact_candidates != self.debounced_contacts)
        release = commit & self.debounced_contacts & ~self.contact_candidates
        recontact = commit & ~self.debounced_contacts & self.contact_candidates
        prior_release = self.episode_release_seen.clone()
        first_thumb_break = release[:, 0] & (self.episode_first_thumb_break_step < 0)
        self.episode_first_thumb_break_step = torch.where(
            first_thumb_break,
            self.episode_length_buf,
            self.episode_first_thumb_break_step,
        )
        self.episode_release_seen |= release
        self.episode_recontact_seen |= recontact & prior_release
        thumb_attempt = (
            self.episode_release_seen[:, 0]
            & raw[:, 0]
            & ~self.debounced_contacts[:, 0]
        )
        first_thumb_attempt = thumb_attempt & ~self.episode_thumb_attempt_seen
        self.episode_first_thumb_attempt_step = torch.where(
            first_thumb_attempt,
            self.episode_length_buf,
            self.episode_first_thumb_attempt_step,
        )
        first_thumb_recontact = (
            recontact[:, 0]
            & prior_release[:, 0]
            & (self.episode_first_thumb_recontact_step < 0)
        )
        self.episode_first_thumb_recontact_step = torch.where(
            first_thumb_recontact,
            self.episode_length_buf,
            self.episode_first_thumb_recontact_step,
        )
        self.episode_thumb_attempt_seen |= thumb_attempt
        self.episode_release_count += release.long()
        self.episode_recontact_count += recontact.long()
        self.debounced_contacts = torch.where(
            commit, self.contact_candidates, self.debounced_contacts
        )
        return (self.episode_release_seen & self.episode_recontact_seen).any(dim=-1)

    def _compute_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        height_drop = self.init_pen_pos[:, 2] - self.pen_pos[:, 2]
        drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        contact = (
            self.contact_force_norm > self.cfg.contact_force_threshold_n
        ).any(dim=-1)
        self.no_contact_steps = torch.where(
            contact, torch.zeros_like(self.no_contact_steps), self.no_contact_steps + 1
        )
        self.episode_max_no_contact_steps = torch.maximum(
            self.episode_max_no_contact_steps, self.no_contact_steps
        )
        linear_speed = torch.linalg.norm(self.pen_linvel, dim=-1)
        angular_speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        invalid = (
            ~torch.isfinite(self.pen_pos).all(dim=-1)
            | ~torch.isfinite(self.pen_quat).all(dim=-1)
            | ~torch.isfinite(self.hand_qpos).all(dim=-1)
        )
        self.current_throw = (
            linear_speed > self.cfg.throw_linear_speed_m_s
        ) | (angular_speed > self.cfg.throw_angular_speed_rad_s)
        self.current_contact_loss = (
            self.no_contact_steps >= self.cfg.no_contact_termination_steps
        )
        self.current_ballistic = (
            self.no_contact_steps >= self.cfg.ballistic_no_contact_steps
        ) & (
            (linear_speed > self.cfg.ballistic_linear_speed_m_s)
            | (angular_speed > self.cfg.ballistic_angular_speed_rad_s)
        )
        self.episode_ballistic_seen |= self.current_ballistic
        self.episode_ballistic_steps += self.current_ballistic.long()
        self.current_penetration_failure = (
            self.episode_max_penetration > self.cfg.penetration_threshold
        )
        self.current_drop = height_drop > self.cfg.drop_threshold
        self.current_drift = drift > self.cfg.drift_threshold
        self.current_invalid = invalid
        failed = (
            self.current_drop
            | self.current_drift
            | self.current_penetration_failure
            | self.current_contact_loss
            | self.current_throw
            | (self.cfg.terminate_on_ballistic & self.current_ballistic)
            | invalid
        )
        self.current_failed = failed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.current_timeout = time_out
        primary = torch.full_like(self.current_primary_failure, -1)
        # Low-priority causes are assigned first; later assignments win.
        for failure_index, flag in (
            (5, self.current_drift),
            (4, self.current_drop),
            (0, self.current_contact_loss),
            (1, self.episode_ballistic_seen),
            (3, self.current_throw),
            (2, self.current_penetration_failure),
            (6, self.current_invalid),
        ):
            primary = torch.where(flag, failure_index, primary)
        self.current_primary_failure = torch.where(failed, primary, -1)
        failure_flags = torch.stack(
            (
                self.current_contact_loss,
                self.current_ballistic,
                self.current_penetration_failure,
                self.current_throw,
                self.current_drop,
                self.current_drift,
                self.current_invalid,
            ),
            dim=-1,
        )
        first_failure = failed & ~self.episode_failure_seen
        self.episode_primary_failure = torch.where(
            first_failure, primary, self.episode_primary_failure
        )
        self.episode_failure_seen |= failed
        self.episode_failure_type_seen |= failure_flags

        if self.cfg.termination_mode == "early":
            terminated = failed
        elif self.cfg.termination_mode == "fixed_horizon":
            # Invalid numerical state is the only safety exception to fixed horizon.
            terminated = invalid
        elif self.cfg.termination_mode == "recovery_grace":
            recoverable_failure = failed & ~invalid
            self.failure_grace_counter = torch.where(
                recoverable_failure,
                self.failure_grace_counter + 1,
                torch.zeros_like(self.failure_grace_counter),
            )
            terminated = invalid | (
                self.failure_grace_counter >= self.cfg.failure_grace_steps
            )
        else:
            raise ValueError(f"Unknown termination_mode={self.cfg.termination_mode!r}")
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed = self.current_failed
        gait_event = self._update_contact_transitions()

        tim_count = self.debounced_contacts[:, :3].sum(dim=-1)
        self.episode_tim_overlap_steps += (tim_count >= 2).long()
        self.episode_zero_tim_run_steps = torch.where(
            tim_count == 0,
            self.episode_zero_tim_run_steps + 1,
            torch.zeros_like(self.episode_zero_tim_run_steps),
        )
        self.episode_max_zero_tim_run_steps = torch.maximum(
            self.episode_max_zero_tim_run_steps,
            self.episode_zero_tim_run_steps,
        )

        alpha = self.cfg.temporal_support_ema_alpha
        raw_contacts = (
            self.contact_force_norm > self.cfg.contact_force_threshold_n
        ).float()
        relative_height = self.pen_pos[:, 2] - self.init_pen_pos[:, 2]
        support_force_z = -self.contact_force_w.sum(dim=1)[:, 2]
        self.temporal_contact_duty.lerp_(raw_contacts, alpha)
        self.temporal_force_delta = (
            self.contact_force_norm - self.temporal_previous_force
        )
        self.temporal_force_ema.lerp_(self.contact_force_norm, alpha)
        self.temporal_previous_force.copy_(self.contact_force_norm)
        self.temporal_height_delta = relative_height - self.temporal_previous_height
        self.temporal_height_ema.lerp_(relative_height, alpha)
        self.temporal_previous_height.copy_(relative_height)
        self.temporal_support_delta = support_force_z - self.temporal_previous_support
        self.temporal_support_ema.lerp_(support_force_z, alpha)
        self.temporal_previous_support.copy_(support_force_z)

        target_angular_velocity = (
            self.cfg.target_rotation_direction * self.pen_angvel[:, 2]
        )
        self.episode_target_rotation += target_angular_velocity * (
            self.cfg.sim.dt * self.cfg.decimation
        )
        self.episode_peak_target_rotation = torch.maximum(
            self.episode_peak_target_rotation, self.episode_target_rotation
        )
        height_delta = self.pen_pos[:, 2] - self.init_pen_pos[:, 2]
        prior_height_drop = self.episode_height_drop_seen.clone()
        self.episode_min_height_delta = torch.minimum(
            self.episode_min_height_delta, height_delta
        )
        new_height_drop = (~prior_height_drop) & (
            height_delta <= -self.cfg.height_drop_diagnostic_m
        )
        self.episode_first_height_drop_step = torch.where(
            new_height_drop,
            self.episode_length_buf,
            self.episode_first_height_drop_step,
        )
        self.episode_height_drop_seen |= new_height_drop
        self.episode_height_recovered_seen |= prior_height_drop & (
            height_delta >= -self.cfg.height_recovery_diagnostic_m
        )
        angular_speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        moving = angular_speed >= self.cfg.axis_alignment_min_angular_speed_rad_s
        directional_axis_alignment = target_angular_velocity / angular_speed.clamp_min(
            1.0e-6
        )
        absolute_axis_alignment = torch.abs(self.pen_angvel[:, 2]) / angular_speed.clamp_min(
            1.0e-6
        )
        self.episode_axis_alignment_sum += torch.where(
            moving, directional_axis_alignment, 0.0
        )
        self.episode_absolute_axis_alignment_sum += torch.where(
            moving, absolute_axis_alignment, 0.0
        )
        self.episode_axis_alignment_steps += moving.long()
        horizontal_error = torch.abs(self.pen_axis[:, 2])
        linear_speed = torch.linalg.norm(self.pen_linvel, dim=-1)
        wrong_axis_speed = torch.linalg.norm(self.pen_angvel[:, :2], dim=-1)
        contacts = self.contact_force_norm > self.cfg.contact_force_threshold_n
        contact_count = contacts.float().sum(dim=-1)
        has_contact = contact_count > 0.0
        slip = (
            self.contact_tangential_slip_speed * contacts.float()
        ).sum(dim=-1) / contact_count.clamp_min(1.0)
        penetration_excess = torch.relu(
            self.episode_max_penetration - 0.0005
        ) / 0.0005
        applied_torque = self.hand.data.applied_torque[:, self.dex5_joint_ids]
        mechanical_power = torch.mean(
            torch.abs(applied_torque * self.hand_qvel), dim=-1
        )

        positive_rotation = torch.clamp(target_angular_velocity, 0.0, 20.0)
        if self.cfg.contact_gate_positive_rotation:
            positive_rotation = positive_rotation * has_contact.float()
        signed_rotation = positive_rotation + torch.clamp(
            target_angular_velocity, -10.0, 0.0
        )
        aligned_rotation = (
            torch.clamp(target_angular_velocity, 0.0, 12.0)
            * torch.relu(directional_axis_alignment)
            * has_contact.float()
        )
        high_angular_speed = torch.relu(angular_speed - 12.0)
        axis_alignment_quality = (
            torch.relu((directional_axis_alignment - 0.5) / 0.5)
            .clamp_max(1.0)
            * moving.float()
            * has_contact.float()
        )
        height_drop = torch.relu(self.init_pen_pos[:, 2] - self.pen_pos[:, 2])
        support_deficit = torch.relu(
            self.cfg.support_target_force_n - support_force_z
        ).clamp_max(0.16)
        support_drop = torch.relu(-self.temporal_support_delta).clamp_max(0.25)
        total_contact_force = self.contact_force_norm.sum(dim=-1)
        support_force_excess = torch.relu(
            total_contact_force - self.cfg.support_force_excess_threshold_n
        )
        steps_since_height_drop = (
            self.episode_length_buf - self.episode_first_height_drop_step
        )
        recovery_active = (
            self.episode_height_drop_seen
            & ~self.episode_height_recovered_seen
            & (self.episode_first_height_drop_step >= 0)
            & (steps_since_height_drop <= self.cfg.recovery_window_steps)
        )
        recovery_height_deficit = (
            torch.relu(-height_delta - self.cfg.height_recovery_diagnostic_m)
            .clamp_max(0.02)
            * recovery_active.float()
        )
        recovery_downward_speed = (
            torch.relu(-self.pen_linvel[:, 2]).clamp_max(0.30)
            * recovery_active.float()
        )
        recovery_progress = (
            torch.relu(self.temporal_height_delta).clamp_max(0.003)
            * recovery_active.float()
        )

        terms = {
            "rotation": self.cfg.rotation_reward_scale * signed_rotation,
            "axis_aligned_rotation": self.cfg.axis_aligned_rotation_reward_scale
            * aligned_rotation,
            "axis_alignment_quality": self.cfg.axis_alignment_quality_reward_scale
            * axis_alignment_quality,
            "horizontal": self.cfg.horizontal_reward_scale
            * (1.0 - horizontal_error.clamp(0.0, 1.0)),
            "linear_speed": -self.cfg.linear_speed_penalty_scale * linear_speed,
            "wrong_axis_rotation": -self.cfg.wrong_axis_penalty_scale
            * wrong_axis_speed,
            "high_angular_speed": -self.cfg.high_angular_speed_penalty_scale
            * high_angular_speed,
            "height_drop": -self.cfg.height_drop_penalty_scale * height_drop,
            "support_margin": -self.cfg.support_margin_penalty_scale
            * support_deficit,
            "support_drop": -self.cfg.support_drop_penalty_scale * support_drop,
            "support_force_excess": -self.cfg.support_force_excess_penalty_scale
            * support_force_excess,
            "recovery_height_deficit": -self.cfg.recovery_height_deficit_penalty_scale
            * recovery_height_deficit,
            "recovery_downward_speed": -self.cfg.recovery_downward_speed_penalty_scale
            * recovery_downward_speed,
            "recovery_progress": self.cfg.recovery_progress_reward_scale
            * recovery_progress,
            "ballistic": -self.cfg.ballistic_penalty_scale
            * self.current_ballistic.float(),
            "slip": -self.cfg.slip_penalty_scale * slip,
            "penetration": -1.50 * penetration_excess.clamp_max(3.0),
            "action_l2": -0.01 * torch.mean(self.actions**2, dim=-1),
            "action_smoothness": -0.02
            * torch.mean((self.actions - self.previous_actions) ** 2, dim=-1),
            "mechanical_power": -0.002 * mechanical_power,
            "failure": -5.0 * failed.float(),
            "terminal_failure": -self.cfg.terminal_failure_penalty_scale
            * (self.current_timeout & self.episode_failure_seen).float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.extras["log"] = {
            f"reward/{name}": value.mean() for name, value in terms.items()
        }
        self.extras["log"].update(
            {
                "metric/target_angular_velocity_rad_s": target_angular_velocity.mean(),
                "metric/positive_rotation_ratio": (target_angular_velocity > 0.5).float().mean(),
                "metric/rotation_over_180deg_ratio": (
                    self.episode_target_rotation > torch.pi
                ).float().mean(),
                "metric/pen_linear_speed_m_s": linear_speed.mean(),
                "metric/wrong_axis_angular_speed_rad_s": wrong_axis_speed.mean(),
                "metric/directional_axis_alignment": torch.where(
                    moving, directional_axis_alignment, 0.0
                ).sum()
                / moving.float().sum().clamp_min(1.0),
                "metric/absolute_axis_alignment": torch.where(
                    moving, absolute_axis_alignment, 0.0
                ).sum()
                / moving.float().sum().clamp_min(1.0),
                "metric/horizontal_error": horizontal_error.mean(),
                "metric/contact_count": contact_count.mean(),
                "metric/gait_event_ratio": gait_event.float().mean(),
                "metric/release_count": self.episode_release_count.float().sum(dim=-1).mean(),
                "metric/recontact_count": self.episode_recontact_count.float().sum(dim=-1).mean(),
                "metric/slip_speed_m_s": slip.mean(),
                "metric/max_penetration_mm": self.episode_max_penetration.mean() * 1000.0,
                "metric/failure_ratio": failed.float().mean(),
                "metric/ballistic_ratio": self.current_ballistic.float().mean(),
                "metric/TIM_overlap_ratio": (tim_count >= 2).float().mean(),
                "metric/zero_TIM_ratio": (tim_count == 0).float().mean(),
                "metric/support_force_z_n": support_force_z.mean(),
                "metric/support_deficit_n": support_deficit.mean(),
                "metric/total_contact_force_n": total_contact_force.mean(),
                "metric/recovery_active_ratio": recovery_active.float().mean(),
                "metric/failure_seen_ratio": self.episode_failure_seen.float().mean(),
                "metric/failure_grace_counter": self.failure_grace_counter.float().mean(),
            }
        )
        for canonical_index, canonical_id in enumerate(CANONICAL_IDS):
            self.extras["log"][f"reset/{canonical_id}_ratio"] = (
                self.canonical_indices_env == canonical_index
            ).float().mean()
        return reward
