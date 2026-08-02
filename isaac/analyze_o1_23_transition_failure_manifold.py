from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PHASES = (
    "release",
    "same_finger_recontact",
    "support_continuity",
    "height_recovery",
    "axis_alignment",
    "safe_transition",
)
FAILURES = (
    "no_TIM_release",
    "release_without_recontact",
    "support_gap",
    "height_not_recovered",
    "axis_misalignment",
    "unsafe_transition",
    "strict_transition",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="O1.23 offline transition-phase and failure-manifold audit."
    )
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _first_true(values: np.ndarray) -> int:
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else -1


def _max_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _quantiles(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": None, "p10": None, "median": None, "p90": None}
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _load(path: str) -> dict[str, np.ndarray]:
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {
            "bank_ids",
            "initial_contacts",
            "debounced_contact",
            "height_delta_m",
            "net_force_proxy_n",
            "pen_linear_velocity_m_s",
            "pen_angular_velocity_rad_s",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{source} is missing O1.23 fields: {missing}")
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["_source"] = np.asarray(str(source))
    return payload


def _episode(data: dict[str, np.ndarray], env: int) -> dict:
    contact = data["debounced_contact"][:, env]
    initial = data["initial_contacts"][env]
    transitions = np.diff(
        np.concatenate((initial[None].astype(np.int8), contact.astype(np.int8)), axis=0),
        axis=0,
    )
    release_steps = [_first_true(transitions[:, finger] < 0) for finger in range(3)]
    release_step = min((step for step in release_steps if step >= 0), default=-1)

    gait_finger = -1
    recontact_step = -1
    for finger, step in enumerate(release_steps):
        if step < 0:
            continue
        candidates = np.flatnonzero(
            (transitions[:, finger] > 0) & (np.arange(transitions.shape[0]) > step)
        )
        if candidates.size and (recontact_step < 0 or candidates[0] < recontact_step):
            gait_finger = finger
            recontact_step = int(candidates[0])

    horizon = contact.shape[0]
    window_start = max(0, release_step)
    window_end = min(
        horizon,
        (recontact_step + 7) if recontact_step >= 0 else (release_step + 13),
    )
    valid_window = release_step >= 0 and window_end > window_start
    tim_count = contact[:, :3].sum(axis=-1)
    all_count = contact.sum(axis=-1)
    height = data["height_delta_m"][:, env]
    support_z = data["net_force_proxy_n"][:, env, 2]
    tim_normal_force = data["contact_force_norm_n"][:, env, :3].sum(axis=-1)
    angular = data["pen_angular_velocity_rad_s"][:, env]
    linear = data["pen_linear_velocity_m_s"][:, env]
    angular_norm = np.linalg.norm(angular, axis=-1)
    axis = np.divide(
        angular[:, 2],
        angular_norm,
        out=np.zeros_like(angular_norm),
        where=angular_norm > 0.5,
    )

    if valid_window:
        overlap_ratio = float(np.mean(tim_count[window_start:window_end] >= 2))
        max_zero_tim = _max_true_run(tim_count[window_start:window_end] == 0)
        pre = tim_normal_force[max(0, release_step - 3) : release_step]
        pre_support = float(np.median(pre)) if pre.size else float(tim_normal_force[release_step])
        minimum_support = float(np.min(tim_normal_force[window_start:window_end]))
        moving = angular_norm[window_start:window_end] > 0.5
        moving_axis = axis[window_start:window_end][moving]
        axis_alignment = float(np.mean(moving_axis)) if moving_axis.size else float("nan")
    else:
        overlap_ratio = float("nan")
        max_zero_tim = horizon
        pre_support = minimum_support = axis_alignment = float("nan")

    height_drop_step = _first_true(height <= -0.005)
    if height_drop_step < 0:
        height_recovered = True
    else:
        recovery_start = max(height_drop_step + 1, recontact_step + 1)
        height_recovered = bool(np.any(height[recovery_start:] >= -0.002))
    support_continuity = bool(
        recontact_step >= 0 and overlap_ratio >= 0.80 and max_zero_tim <= 2
    )
    axis_aligned = bool(np.isfinite(axis_alignment) and axis_alignment >= 0.70)
    ballistic = bool(
        np.any(
            (all_count == 0)
            & ((np.linalg.norm(linear, axis=-1) > 0.20) | (angular_norm > 12.0))
        )
    )
    contact_loss = _max_true_run(all_count == 0) >= 4
    unsafe = ballistic or contact_loss or float(np.min(height)) < -0.040
    gates = {
        "release": release_step >= 0,
        "same_finger_recontact": recontact_step >= 0,
        "support_continuity": support_continuity,
        "height_recovery": height_recovered,
        "axis_alignment": axis_aligned,
        "safe_transition": not unsafe,
    }
    if not gates["release"]:
        failure = "no_TIM_release"
    elif not gates["same_finger_recontact"]:
        failure = "release_without_recontact"
    elif not gates["support_continuity"]:
        failure = "support_gap"
    elif not gates["height_recovery"]:
        failure = "height_not_recovered"
    elif not gates["axis_alignment"]:
        failure = "axis_misalignment"
    elif not gates["safe_transition"]:
        failure = "unsafe_transition"
    else:
        failure = "strict_transition"

    action_norm = tracking_error = float("nan")
    if "action" in data and valid_window:
        action_norm = float(
            np.mean(np.linalg.norm(data["action"][window_start:window_end, env], axis=-1))
        )
    if "processed_targets" in data and "hand_qpos" in data and valid_window:
        tracking_error = float(
            np.mean(
                np.linalg.norm(
                    data["processed_targets"][window_start:window_end, env]
                    - data["hand_qpos"][window_start:window_end, env],
                    axis=-1,
                )
            )
        )
    return {
        "env_id": env,
        "bank_id": int(data["bank_ids"][env]),
        "release_step": release_step,
        "recontact_step": recontact_step,
        "gait_finger": FINGERS[gait_finger] if gait_finger >= 0 else None,
        "phase_gates": gates,
        "first_failed_gate": failure,
        "transition_window_steps": [window_start, window_end],
        "TIM_overlap_ratio": overlap_ratio,
        "max_zero_TIM_steps": max_zero_tim,
        "pre_release_TIM_normal_force_n": pre_support,
        "minimum_transition_TIM_normal_force_n": minimum_support,
        "support_retention_ratio": (
            minimum_support / max(pre_support, 1.0e-4)
            if np.isfinite(minimum_support) and np.isfinite(pre_support)
            else float("nan")
        ),
        "minimum_height_delta_mm": float(1000.0 * np.min(height)),
        "height_drop_step": height_drop_step,
        "height_recovered": height_recovered,
        "transition_axis_alignment": axis_alignment,
        "ballistic": ballistic,
        "contact_loss": contact_loss,
        "action_norm_during_transition": action_norm,
        "target_tracking_error_l2_rad": tracking_error,
    }


def _summarize(data: dict[str, np.ndarray], label: str) -> dict:
    episodes = [_episode(data, env) for env in range(data["bank_ids"].size)]
    count = len(episodes)
    phase_counts = {}
    for phase_index, phase in enumerate(PHASES):
        prerequisites = PHASES[: phase_index + 1]
        phase_counts[phase] = sum(
            all(ep["phase_gates"][required] for required in prerequisites)
            for ep in episodes
        )
    failure_counts = {
        failure: sum(ep["first_failed_gate"] == failure for ep in episodes)
        for failure in FAILURES
    }
    transitioned = [ep for ep in episodes if ep["phase_gates"]["same_finger_recontact"]]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in transitioned]

    summary = {
        "label": label,
        "episodes": count,
        "unique_bank_rows": int(np.unique(data["bank_ids"]).size),
        "phase_reach_count": phase_counts,
        "phase_reach_rate": {key: value / count for key, value in phase_counts.items()},
        "first_failed_gate_count": failure_counts,
        "first_failed_gate_rate": {key: value / count for key, value in failure_counts.items()},
        "transition_subset": {
            "episodes": len(transitioned),
            "TIM_overlap_ratio": _quantiles(values("TIM_overlap_ratio")),
            "max_zero_TIM_steps": _quantiles(values("max_zero_TIM_steps")),
            "support_retention_ratio": _quantiles(values("support_retention_ratio")),
            "minimum_height_delta_mm": _quantiles(values("minimum_height_delta_mm")),
            "transition_axis_alignment": _quantiles(values("transition_axis_alignment")),
            "action_norm": _quantiles(values("action_norm_during_transition")),
            "target_tracking_error_l2_rad": _quantiles(values("target_tracking_error_l2_rad")),
        },
        "gait_finger_count": {
            finger: sum(ep["gait_finger"] == finger for ep in episodes)
            for finger in FINGERS[:3]
        },
    }
    return {"summary": summary, "episodes": episodes}


def _decision(uniform: dict, curriculum: dict) -> dict:
    a = uniform["summary"]
    b = curriculum["summary"]
    transition_rate = a["phase_reach_rate"]["same_finger_recontact"]
    support_rate = a["phase_reach_rate"]["support_continuity"]
    recovery_rate = a["phase_reach_rate"]["height_recovery"]
    dominant = max(
        FAILURES[:-1], key=lambda key: a["first_failed_gate_count"][key]
    )
    phase_data_exists = transition_rate >= 0.10
    downstream_boundary_exists = support_rate > 0.0 or recovery_rate > 0.0
    # User-confirmed V9 semantics place this motion inside P4. The generic
    # same-finger gait funnel cannot authorize a curriculum until the P4->P5
    # thumb-index capture window and penetration are audited explicitly.
    authorize = False
    return {
        "state_curriculum_conclusion": "REJECTED_BY_O1_22",
        "uniform_policy_dominant_failure_phase": dominant,
        "transition_phase_data_exists": phase_data_exists,
        "nonzero_downstream_boundary_exists": downstream_boundary_exists,
        "transition_curriculum_design_authorized": authorize,
        "p4_p5_capture_window_audit_required": True,
        "penetration_audit_required": True,
        "skill_pretraining_or_hierarchy_authorized": False,
        "interpretation": (
            "The observed motion is a V9 P4 dynamic rotation attempt. Do not start the "
            "previously proposed generic transition curriculum. First locate the geometric "
            "P4->P5 capture window, verify that the pen does not cross the thumb through "
            "penetration, and measure whether thumb/index establish a pinch before the "
            "index/middle drive continues."
        ),
        "curriculum_checkpoint_effect": {
            "recontact_rate_delta": b["phase_reach_rate"]["same_finger_recontact"] - transition_rate,
            "support_continuity_rate_delta": b["phase_reach_rate"]["support_continuity"] - support_rate,
            "height_recovery_rate_delta": b["phase_reach_rate"]["height_recovery"] - recovery_rate,
        },
    }


def _plot(report: dict, output: Path) -> None:
    labels = ("A uniform", "B state curriculum")
    colors = ("#276FBF", "#D1495B")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.36
    x = np.arange(len(PHASES))
    for index, key in enumerate(("uniform", "curriculum")):
        rates = report[key]["summary"]["phase_reach_rate"]
        axes[0].bar(
            x + (index - 0.5) * width,
            [100.0 * rates[phase] for phase in PHASES],
            width,
            label=labels[index],
            color=colors[index],
        )
    axes[0].set_xticks(x, [phase.replace("_", "\n") for phase in PHASES], fontsize=8)
    axes[0].set_ylabel("Episode reach rate (%)")
    axes[0].set_title("Transition phase reachability")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    failures = FAILURES[:-1]
    y = np.arange(len(failures))
    for index, key in enumerate(("uniform", "curriculum")):
        rates = report[key]["summary"]["first_failed_gate_rate"]
        axes[1].barh(
            y + (index - 0.5) * width,
            [100.0 * rates[failure] for failure in failures],
            width,
            label=labels[index],
            color=colors[index],
        )
    axes[1].set_yticks(y, [failure.replace("_", " ") for failure in failures], fontsize=8)
    axes[1].set_xlabel("First-failure share (%)")
    axes[1].set_title("Failure manifold by first unmet gate")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    uniform_data = _load(args.uniform)
    curriculum_data = _load(args.curriculum)
    if not np.array_equal(uniform_data["bank_ids"], curriculum_data["bank_ids"]):
        raise ValueError("A/B diagnostics do not use the same ordered P3 bank rows")
    if str(uniform_data["bank_sha256"]) != str(curriculum_data["bank_sha256"]):
        raise ValueError("A/B diagnostics use different O0 banks")
    uniform = _summarize(uniform_data, "A_uniform")
    curriculum = _summarize(curriculum_data, "B_state_curriculum")
    report = {
        "status": "PASS_DIAGNOSTIC",
        "phase": "O1.23 Failure Manifold / Transition Phase Analysis",
        "artifact_role": "diagnostic_only_not_teacher_not_reset",
        "semantic_correction": {
            "confirmed_motion": "V9 P4 dynamic rotation around the thumb",
            "expected_switch": "when the pen end reaches the thumb-index side, switch to P5 thumb-index pinch",
            "observed_failure": "index-middle drive continues without P5 capture; possible penetration-assisted crossing",
            "generic_handoff_metrics_are_not_P5_success": True,
        },
        "matched_contract": {
            "same_ordered_bank_rows": True,
            "same_bank_hash": True,
            "same_horizon_steps": int(uniform_data["debounced_contact"].shape[0]),
            "uniform_checkpoint": str(uniform_data["checkpoint"]),
            "curriculum_checkpoint": str(curriculum_data["checkpoint"]),
        },
        "phase_contract": {
            "release": "first debounced release of thumb/index/middle",
            "same_finger_recontact": "a released TIM finger later completes debounced re-contact",
            "support_continuity": "TIM overlap >=80% and zero-TIM run <=2 steps in transition window",
            "height_recovery": "no 5mm loss, or recovery to within 2mm of reset height",
            "axis_alignment": "mean directional axis alignment >=0.70 while moving",
            "safe_transition": "no ballistic, four-step all-contact loss, or 40mm drop",
            "phase_reach_semantics": "cumulative: each phase requires every preceding gate",
        },
        "uniform": uniform,
        "curriculum": curriculum,
    }
    report["decision_gate"] = _decision(uniform, curriculum)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(report, output.with_suffix(".png"))
    print(json.dumps(report["decision_gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
