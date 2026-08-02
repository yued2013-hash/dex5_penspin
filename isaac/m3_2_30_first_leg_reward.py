from __future__ import annotations

import torch


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_index_near_surface_reward(
    *,
    index_tip_dist: torch.Tensor,
    initial_index_tip_dist: torch.Tensor,
    obj_pos: torch.Tensor,
    init_obj_pos: torch.Tensor,
    target_obj_pos: torch.Tensor,
    max_penetration: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: int,
    failed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vectorized reward for the M3.2.30 approach-only first leg.

    Shapes:
        index_tip_dist: [N], signed or unsigned fingertip-to-pen surface distance in meters.
        initial_index_tip_dist: [N], reset-time fingertip distance in meters.
        obj_pos/init_obj_pos/target_obj_pos: [N, 3].
        max_penetration: [N], max penetration depth in meters.
        actions/prev_actions: [N, A].
        progress_buf: [N].

    Success is approach-only: final index_tip_dist < 14mm. Index contact is not required.
    """

    tau = torch.clamp(progress_buf.float() / float(max_episode_length), 0.0, 1.0)
    terminal_gate = smoothstep((tau - 0.45) / 0.40)

    positive_dist = torch.clamp(index_tip_dist, min=0.0)
    initial = torch.clamp(initial_index_tip_dist, min=0.004)
    approach = 1.0 - torch.clamp(positive_dist / 0.030, 0.0, 1.0)
    near = 1.0 - torch.clamp(positive_dist / 0.014, 0.0, 1.0)
    near_hit = (index_tip_dist <= 0.014).float()
    progress = torch.clamp((initial - index_tip_dist) / torch.clamp(initial - 0.003, min=0.006), 0.0, 1.0)

    center_dist = torch.linalg.norm(obj_pos - target_obj_pos, dim=-1)
    xy_drift = torch.linalg.norm((obj_pos - init_obj_pos)[..., :2], dim=-1)
    height_drop = torch.clamp(init_obj_pos[..., 2] - obj_pos[..., 2], min=0.0)
    penetration_excess = torch.clamp((max_penetration - 0.0015) / 0.003, min=0.0, max=1.0)
    miss_14 = torch.clamp((positive_dist - 0.014) / 0.025, min=0.0, max=1.0)
    action_l2 = torch.linalg.norm(actions, dim=-1)
    action_smooth = torch.linalg.norm(actions - prev_actions, dim=-1)
    failed_f = torch.zeros_like(index_tip_dist) if failed is None else failed.float()

    terms = {
        "alive": torch.full_like(index_tip_dist, 0.10),
        "index_tip_approach": 1.70 * approach,
        "index_tip_progress": 0.95 * progress,
        "index_tip_near_terminal": 1.80 * terminal_gate * near_hit + 1.20 * terminal_gate * near,
        "miss_14mm": -2.40 * terminal_gate * miss_14,
        "center_guard": -0.70 * torch.clamp((center_dist - 0.020) / 0.030, min=0.0, max=1.0),
        "height_guard": -0.80 * torch.clamp((height_drop + xy_drift) / 0.035, min=0.0, max=1.0),
        "penetration": -2.00 * penetration_excess,
        "action_l2": -0.06 * action_l2,
        "action_smoothness": -0.08 * action_smooth,
        "failure": -5.00 * failed_f,
    }
    reward = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    return reward, terms
