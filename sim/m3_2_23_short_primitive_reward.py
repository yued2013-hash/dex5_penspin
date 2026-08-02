from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CORE_SUPPORT = ("thumb", "index", "middle", "ring")


@dataclass
class ShortPrimitiveWeights:
    alive: float = 0.10
    support_contact: float = 0.85
    desired_contact: float = 1.15
    terminal_contact: float = 2.75
    thumb_support: float = 0.65
    index_capture: float = 0.75
    rotation_progress: float = 0.55
    target_axis: float = 0.50
    target_center: float = 0.45
    height_guard: float = 0.80
    ring_guard_contact: float = 0.15
    ring_overload: float = 1.25
    pinky: float = 0.90
    penetration: float = 1.40
    action_l2: float = 0.06
    action_smoothness: float = 0.08
    joint_limit: float = 0.40
    middle_proximity: float = 0.0
    terminal_linear_speed: float = 0.0
    terminal_angular_speed: float = 0.0
    failure: float = 5.00


@dataclass
class PrimitiveSpec:
    label: str
    start_seed: str
    target_seed: str
    duration_s: float
    target_contacts: list[str]
    support_contacts: list[str] = field(default_factory=lambda: list(CORE_SUPPORT))
    allowed_extra_contacts: list[str] = field(default_factory=list)
    forbidden_contacts: list[str] = field(default_factory=lambda: ["pinky"])
    reset_mode: str = "static_keypose"
    source_summary_csv: str = ""
    source_summary_label: str = ""
    phase_direction: float = 0.0
    target_phase_delta: float = 0.0
    priority: str = "normal"
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def smoothstep(x: float) -> float:
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _finger_dict(values: dict[str, Any] | None) -> dict[str, float]:
    values = values or {}
    return {finger: float(values.get(finger, 0.0)) for finger in FINGERS}


def _as_contacts(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, set):
        return set(values)
    if isinstance(values, (list, tuple)):
        return {str(x) for x in values}
    if isinstance(values, str):
        if not values or values == "none":
            return set()
        return {x for x in values.split(",") if x}
    return set()


def terminal_gate(tau: float) -> float:
    return smoothstep((float(tau) - 0.55) / 0.45)


def contact_role_terms(
    *,
    contacts: set[str],
    force_share: dict[str, float],
    spec: PrimitiveSpec,
    tau: float,
) -> dict[str, float]:
    desired = set(spec.target_contacts)
    support = set(spec.support_contacts)
    forbidden = set(spec.forbidden_contacts)
    gate = terminal_gate(tau)
    shares = _finger_dict(force_share)

    desired_ratio = 0.0
    if desired:
        desired_ratio = sum(float(f in contacts) for f in desired) / len(desired)
    all_desired = float(bool(desired) and desired.issubset(contacts))
    support_count = len(support.intersection(contacts))
    support_ratio = min(1.0, support_count / 2.0)

    thumb = float("thumb" in contacts)
    index = float("index" in contacts)
    ring = float("ring" in contacts)
    pinky = float("pinky" in contacts)

    ring_allowed = "ring" in desired or "ring" in spec.allowed_extra_contacts
    ring_overload = max(0.0, shares["ring"] - 0.35)
    if not ring_allowed:
        ring_overload += 0.35 * ring

    forbidden_ratio = 0.0
    if forbidden:
        forbidden_ratio = sum(float(f in contacts) for f in forbidden) / len(forbidden)

    return {
        "desired_contact": desired_ratio,
        "terminal_contact": gate * all_desired,
        "support_contact": support_ratio,
        "thumb_support": thumb * min(1.0, shares["thumb"] / 0.30 if shares["thumb"] > 0.0 else 0.75),
        "index_capture": gate * index,
        "ring_guard_contact": ring * float(ring_allowed) * (1.0 - min(1.0, ring_overload / 0.45)),
        "ring_overload": ring_overload,
        "pinky": pinky + 0.5 * forbidden_ratio,
    }


def step_reward_terms(
    *,
    contacts: set[str],
    force_share: dict[str, float],
    pen_center: np.ndarray,
    initial_center: np.ndarray,
    target_center: np.ndarray,
    pen_axis: np.ndarray,
    target_axis: np.ndarray,
    z0: float,
    phase_delta: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    max_penetration_m: float,
    joint_limit_margin: float,
    middle_tip_distance: float,
    pen_linear_speed: float,
    pen_angular_speed: float,
    tau: float,
    spec: PrimitiveSpec,
    weights: ShortPrimitiveWeights | None = None,
    failed: bool = False,
) -> dict[str, float]:
    weights = weights or ShortPrimitiveWeights()
    role = contact_role_terms(contacts=contacts, force_share=force_share, spec=spec, tau=tau)

    center_dist = float(np.linalg.norm(np.asarray(pen_center) - np.asarray(target_center)))
    xy_drift = float(np.linalg.norm((np.asarray(pen_center) - np.asarray(initial_center))[:2]))
    height_drop = max(0.0, float(z0) - float(np.asarray(pen_center)[2]))
    axis_cos = float(
        np.clip(
            np.dot(pen_axis, target_axis) / (np.linalg.norm(pen_axis) * np.linalg.norm(target_axis) + 1e-9),
            -1.0,
            1.0,
        )
    )
    axis_error = float(np.arccos(axis_cos))
    progress = max(0.0, float(spec.phase_direction) * float(phase_delta)) if spec.phase_direction else 0.0
    action_l2 = float(np.linalg.norm(action))
    action_smooth = float(np.linalg.norm(np.asarray(action) - np.asarray(prev_action)))
    penetration_excess = max(0.0, float(max_penetration_m) - 0.006) / 0.006
    joint_limit_excess = max(0.0, 0.035 - float(joint_limit_margin)) / 0.035
    gate = terminal_gate(tau)
    middle_near = 1.0 - min(1.0, max(0.0, float(middle_tip_distance)) / 0.025)
    linear_speed_excess = min(
        1.0,
        max(0.0, float(pen_linear_speed) - 0.12) / 0.30,
    )
    angular_speed_excess = min(
        1.0,
        max(0.0, float(pen_angular_speed) - 12.0) / 25.0,
    )

    reward_terms = {
        "alive": weights.alive,
        "support_contact": weights.support_contact * role["support_contact"],
        "desired_contact": weights.desired_contact * role["desired_contact"],
        "terminal_contact": weights.terminal_contact * role["terminal_contact"],
        "thumb_support": weights.thumb_support * role["thumb_support"],
        "index_capture": weights.index_capture * role["index_capture"],
        "ring_guard_contact": weights.ring_guard_contact * role["ring_guard_contact"],
        "rotation_progress": weights.rotation_progress * np.clip(progress, 0.0, 0.08) / 0.08,
        "target_axis": -weights.target_axis * min(1.0, axis_error / 0.65),
        "target_center": -weights.target_center * min(1.0, center_dist / 0.030),
        "height_guard": -weights.height_guard * min(1.0, (height_drop + xy_drift) / 0.035),
        "ring_overload": -weights.ring_overload * min(1.0, role["ring_overload"] / 0.45),
        "pinky": -weights.pinky * min(1.0, role["pinky"]),
        "penetration": -weights.penetration * min(1.0, penetration_excess),
        "action_l2": -weights.action_l2 * action_l2,
        "action_smoothness": -weights.action_smoothness * action_smooth,
        "joint_limit": -weights.joint_limit * min(1.0, joint_limit_excess),
        "middle_proximity": weights.middle_proximity * gate * middle_near,
        "terminal_linear_speed": -weights.terminal_linear_speed * gate * linear_speed_excess,
        "terminal_angular_speed": -weights.terminal_angular_speed * gate * angular_speed_excess,
        "failure": -weights.failure * float(failed),
    }
    reward_terms["total"] = float(sum(reward_terms.values()))
    return {key: float(value) for key, value in reward_terms.items()}


def summary_reward_terms(
    row: dict[str, Any],
    spec: PrimitiveSpec,
    weights: ShortPrimitiveWeights | None = None,
) -> dict[str, float]:
    weights = weights or ShortPrimitiveWeights()
    duration = max(float(spec.duration_s), 1e-6)
    survival = min(1.0, float(row.get("survival_s", row.get("survival_seconds", 0.0))) / duration)
    desired_ratio = float(row.get("desired_ratio", 0.0))
    active = float(row.get("active_contact_ratio", row.get("tripod_ratio", 0.0)))
    target_final = float(row.get("target_final_window_ratio", desired_ratio))
    support_final = float(row.get("support_final_window_ratio", active))
    contacts = _as_contacts(row.get("final_contacts", []))
    contact_ratio = _finger_dict(row.get("contact_ratio"))
    shares = _finger_dict(row.get("force_share_mean"))
    terminated = bool(row.get("terminated", False))

    target_contacts = set(spec.target_contacts)
    terminal_hit = float(bool(target_contacts) and target_contacts.issubset(contacts))
    pinky_ratio = contact_ratio["pinky"]
    ring_allowed = "ring" in target_contacts or "ring" in set(spec.allowed_extra_contacts)
    ring_overload = max(0.0, shares["ring"] - 0.35)
    if not ring_allowed:
        ring_overload += 0.35 * contact_ratio["ring"]

    penetration_mm = float(row.get("max_penetration_mm", 0.0))
    penetration_excess = max(0.0, penetration_mm - 6.0) / 6.0
    actuator_force = float(row.get("max_actuator_force", 0.0))
    actuator_excess = max(0.0, actuator_force - 14.0) / 14.0

    terms = {
        "survival": 30.0 * survival,
        "active_support": 12.0 * active,
        "desired_ratio": 16.0 * desired_ratio,
        "target_final": 26.0 * target_final,
        "support_final": 14.0 * support_final,
        "terminal_hit": 18.0 * terminal_hit,
        "thumb_contact": 8.0 * contact_ratio["thumb"],
        "index_contact": 9.0 * contact_ratio["index"],
        "thumb_share": 5.0 * min(1.0, shares["thumb"] / 0.30 if shares["thumb"] > 0.0 else 0.0),
        "ring_guard": 3.0 * contact_ratio["ring"] * float(ring_allowed),
        "pinky_penalty": -24.0 * pinky_ratio,
        "ring_overload_penalty": -14.0 * min(1.0, ring_overload / 0.45),
        "penetration_penalty": -10.0 * min(1.0, penetration_excess),
        "actuator_penalty": -4.0 * min(1.0, actuator_excess),
        "termination_penalty": -25.0 * float(terminated),
    }
    terms["total"] = float(sum(terms.values()))
    return {key: float(value) for key, value in terms.items()}


def readiness_class(total_score: float) -> str:
    if total_score >= 95.0:
        return "ready_teacher"
    if total_score >= 70.0:
        return "rl_trainable"
    if total_score >= 45.0:
        return "needs_focused_reward"
    return "needs_pose_or_reset_fix"
