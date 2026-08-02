from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from build_m3_2_9_photo_five_phase_debug import PHASE_SPECS, normalize, set_state, smoothstep
from dex5_penspin_env import Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


def load_keyframes(seed_dir: Path) -> list[dict[str, Any]]:
    keyframes: list[dict[str, Any]] = []
    for spec in PHASE_SPECS:
        path = seed_dir / f"P{spec['phase']}_{spec['name']}_hand_pen_tuned_seed.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        keyframes.append(data)
    return keyframes


def make_env(
    keyframe: dict[str, Any],
    *,
    duration: float,
    render: bool,
    action_delta_limit: float,
    hard_hold_hand: bool,
) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=keyframe["model_path"],
        pen_init_pos=keyframe["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=duration,
        action_delta_limit=action_delta_limit,
        actuator_kp=keyframe.get("actuator_kp", 400.0),
        actuator_kv=keyframe.get("actuator_kv", 20.0),
        thumb_force_limit=keyframe.get("thumb_force_limit", 0.0),
        hard_hold_hand=hard_hold_hand,
        visual_only_markers=bool(keyframe.get("visual_only_markers", True)),
        disable_thumb_mesh_collision=bool(keyframe.get("disable_thumb_mesh_collision", True)),
        disable_distal_mesh_collision=bool(keyframe.get("disable_distal_mesh_collision", True)),
        render_mode="rgb_array" if render else None,
        seed=0,
    )


def reset_to_keyframe(env: Dex5PenSpinMinimalEnv, keyframe: dict[str, Any]) -> None:
    set_state(
        env,
        np.asarray(keyframe["hand_qpos"], dtype=np.float64),
        np.asarray(keyframe["pen_pos"], dtype=np.float64),
        np.asarray(keyframe["pen_quat"], dtype=np.float64),
    )


def quat_to_pen_axis(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(np.asarray(quat, dtype=np.float64))
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )


def keyframe_target_phases(keyframes: list[dict[str, Any]]) -> np.ndarray:
    angles = []
    for keyframe in keyframes:
        axis = quat_to_pen_axis(np.asarray(keyframe["pen_quat"], dtype=np.float64))
        angles.append(float(np.arctan2(axis[1], axis[0])))
    return np.unwrap(np.asarray(angles, dtype=np.float64))


def interpolate_qpos(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    t = smoothstep(float(t))
    return (1.0 - t) * q0 + t * q1


def target_for_step(
    keyframes: list[dict[str, Any]],
    *,
    step: int,
    control_hz: float,
    start_hold_seconds: float,
    segment_seconds: float,
    end_hold_seconds: float,
) -> tuple[np.ndarray, int, float, str]:
    qposes = [np.asarray(k["hand_qpos"], dtype=np.float64) for k in keyframes]
    start_hold_steps = int(round(start_hold_seconds * control_hz))
    segment_steps = max(1, int(round(segment_seconds * control_hz)))
    end_hold_steps = int(round(end_hold_seconds * control_hz))

    if step < start_hold_steps:
        return qposes[0], 0, 0.0, f"P{keyframes[0]['phase']}_hold"

    local = step - start_hold_steps
    total_segments = len(keyframes) - 1
    if local >= total_segments * segment_steps:
        return qposes[-1], len(keyframes) - 1, 1.0, f"P{keyframes[-1]['phase']}_hold"

    seg = int(local // segment_steps)
    phase_t = float((local % segment_steps) / segment_steps)
    target = interpolate_qpos(qposes[seg], qposes[seg + 1], phase_t)
    return target, seg, phase_t, f"P{keyframes[seg]['phase']}->P{keyframes[seg + 1]['phase']}"


def desired_contacts_for_segment(keyframes: list[dict[str, Any]], seg_index: int, phase_t: float) -> set[str]:
    # During a handover, judge contact against the upcoming keyframe after the
    # halfway point. This keeps the metric aligned with "relay" rather than
    # static hold at the previous pose.
    idx = min(seg_index + (1 if phase_t >= 0.5 else 0), len(keyframes) - 1)
    return set(keyframes[idx].get("desired_contacts", []))


def axis_angle_xy(axis: np.ndarray) -> float:
    return float(np.arctan2(axis[1], axis[0]))


def phase_metrics(angles: np.ndarray, target_phases: np.ndarray) -> dict[str, float]:
    if len(angles) == 0:
        return {
            "final_axis_progress": 0.0,
            "peak_axis_progress": 0.0,
            "target_progress": float(abs(target_phases[-1] - target_phases[0])),
            "progress_ratio": 0.0,
            "monotonic_ratio": 0.0,
            "negative_progress": 0.0,
            "max_drawdown": 0.0,
        }
    unwrapped = np.unwrap(angles)
    target_delta = float(target_phases[-1] - target_phases[0])
    sign = float(np.sign(target_delta) or 1.0)
    directed = sign * (unwrapped - unwrapped[0])
    deltas = np.diff(directed)
    positive = float(np.sum(np.clip(deltas, 0.0, None)))
    negative = float(np.sum(np.clip(-deltas, 0.0, None)))
    running_max = np.maximum.accumulate(directed)
    drawdown = running_max - directed
    target_abs = abs(target_delta)
    return {
        "final_axis_progress": float(directed[-1]),
        "peak_axis_progress": float(np.max(directed)),
        "target_progress": float(target_abs),
        "progress_ratio": float(np.clip(directed[-1] / (target_abs + 1e-9), -2.0, 2.0)),
        "peak_progress_ratio": float(np.clip(np.max(directed) / (target_abs + 1e-9), -2.0, 2.0)),
        "monotonic_ratio": float(positive / (positive + negative + 1e-9)),
        "negative_progress": negative,
        "max_drawdown": float(np.max(drawdown)),
    }


def evaluate_relay(
    keyframes: list[dict[str, Any]],
    config: dict[str, float],
    *,
    render: bool,
    render_every: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    total_duration = (
        config["start_hold_seconds"]
        + (len(keyframes) - 1) * config["segment_seconds"]
        + config["end_hold_seconds"]
    )
    env = make_env(
        keyframes[0],
        duration=total_duration,
        render=render,
        action_delta_limit=config["action_delta_limit"],
        hard_hold_hand=bool(config.get("hard_hold_hand", False)),
    )
    env.reset(seed=0)
    reset_to_keyframe(env, keyframes[0])

    target_phases = keyframe_target_phases(keyframes)
    center0 = env.get_state().pen_center.copy()
    desired_frames = 0
    active_contact_frames = 0
    segment_steps: Counter[str] = Counter()
    segment_desired_hits: Counter[str] = Counter()
    segment_exact: dict[str, Counter[str]] = defaultdict(Counter)
    contact_counter: Counter[str] = Counter()
    exact_counter: Counter[str] = Counter()
    max_penetration = 0.0
    min_height = float(center0[2])
    returns = 0.0
    terminated = False
    truncated = False
    frames: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    axis_angles: list[float] = []
    action_norms: list[float] = []
    desired_ratios: list[float] = []
    phase_labels: list[str] = []

    if render:
        frames.append(env.render())

    max_steps = int(round(total_duration * env.control_hz))
    for step in range(max_steps):
        target, seg, phase_t, label = target_for_step(
            keyframes,
            step=step,
            control_hz=env.control_hz,
            start_hold_seconds=config["start_hold_seconds"],
            segment_seconds=config["segment_seconds"],
            end_hold_seconds=config["end_hold_seconds"],
        )
        delta = config["tracking_gain"] * (target - env.target_qpos.astype(np.float64))
        action = np.clip(delta, -config["action_delta_limit"], config["action_delta_limit"]).astype(np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)

        fingers, geoms, penetration = contact_snapshot(env)
        desired = desired_contacts_for_segment(keyframes, seg, phase_t)
        exact = ",".join(sorted(fingers)) or "none"
        hit = int(bool(desired and desired.issubset(fingers)))
        desired_frames += hit
        active_contact_frames += int(len(fingers) >= 2)
        segment_steps[label] += 1
        segment_desired_hits[label] += hit
        segment_exact[label][exact] += 1
        exact_counter[exact] += 1
        for finger in fingers:
            contact_counter[finger] += 1
        max_penetration = max(max_penetration, float(penetration))

        state = env.get_state()
        centers.append(state.pen_center.copy())
        axes.append(state.pen_axis.copy())
        axis_angles.append(axis_angle_xy(state.pen_axis))
        action_norms.append(float(np.linalg.norm(action)))
        desired_ratios.append(float(hit))
        phase_labels.append(label)
        min_height = min(min_height, float(state.pen_center[2]))

        if render and step % render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    if render and (not frames or (len(centers) % render_every != 0)):
        frames.append(env.render())

    steps = max(1, len(centers))
    centers_np = np.asarray(centers, dtype=np.float64) if centers else np.zeros((1, 3), dtype=np.float64)
    axes_np = np.asarray(axes, dtype=np.float64) if axes else np.zeros((1, 3), dtype=np.float64)
    angle_np = np.asarray(axis_angles, dtype=np.float64)
    metrics = phase_metrics(angle_np, target_phases)
    drift = float(np.linalg.norm(centers_np[-1] - center0))
    xy_drift = float(np.linalg.norm(centers_np[-1, :2] - center0[:2]))
    z_delta = float(centers_np[-1, 2] - center0[2])
    survival_seconds = float(steps / env.control_hz)
    desired_ratio = float(desired_frames / steps)
    active_contact_ratio = float(active_contact_frames / steps)
    score = (
        900.0 * survival_seconds / max(total_duration, 1e-9)
        + 500.0 * metrics["peak_progress_ratio"]
        + 180.0 * active_contact_ratio
        + 140.0 * desired_ratio
        + 80.0 * metrics["monotonic_ratio"]
        - 400.0 * max(0.0, -metrics["final_axis_progress"])
        - 260.0 * metrics["negative_progress"]
        - 180.0 * metrics["max_drawdown"]
        - 900.0 * xy_drift
        - 300.0 * max(0.0, 0.050 - min_height)
        - 240.0 * max(0.0, max_penetration - 0.014)
        - (250.0 if terminated else 0.0)
    )
    row = {
        **config,
        "score": float(score),
        "return": float(returns),
        "total_duration": float(total_duration),
        "steps": int(steps),
        "survival_seconds": survival_seconds,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "desired_contact_ratio": desired_ratio,
        "active_contact_ratio": active_contact_ratio,
        "drift": drift,
        "xy_drift": xy_drift,
        "z_delta": z_delta,
        "min_height": min_height,
        "max_penetration": float(max_penetration),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_norm_max": float(np.max(action_norms)) if action_norms else 0.0,
        "contact_counter": dict(contact_counter),
        "exact_contact_counter": dict(exact_counter),
        "segment_desired_ratio": {
            label: float(segment_desired_hits[label] / max(1, segment_steps[label]))
            for label in segment_steps
        },
        "segment_top_contacts": {
            label: segment_exact[label].most_common(5)
            for label in segment_steps
        },
        **metrics,
    }
    traces = None
    if render:
        traces = {
            "frames": np.asarray(frames, dtype=np.uint8),
            "centers": centers_np,
            "axes": axes_np,
            "axis_angles": angle_np,
            "action_norms": np.asarray(action_norms, dtype=np.float64),
            "desired_hits": np.asarray(desired_ratios, dtype=np.float64),
            "phase_labels": np.asarray(phase_labels, dtype=object),
        }
    env.close()
    return row, traces


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "rank",
        "score",
        "segment_seconds",
        "start_hold_seconds",
        "end_hold_seconds",
        "tracking_gain",
        "action_delta_limit",
        "hard_hold_hand",
        "survival_seconds",
        "total_duration",
        "terminated",
        "progress_ratio",
        "peak_progress_ratio",
        "final_axis_progress",
        "peak_axis_progress",
        "target_progress",
        "monotonic_ratio",
        "negative_progress",
        "max_drawdown",
        "desired_contact_ratio",
        "active_contact_ratio",
        "xy_drift",
        "z_delta",
        "min_height",
        "max_penetration",
        "action_norm_mean",
        "segment_desired_ratio",
        "segment_top_contacts",
        "contact_counter",
        "exact_contact_counter",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {"rank": rank, **row}
            writer.writerow({field: out.get(field, "") for field in fields})


def plot_trace(path: Path, traces: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    centers = traces["centers"]
    angle = np.unwrap(traces["axis_angles"])
    action_norms = traces["action_norms"]
    desired_hits = traces["desired_hits"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(centers[:, 0], label="x")
    axes[0].plot(centers[:, 1], label="y")
    axes[0].plot(centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].legend()
    axes[1].plot(angle - angle[0])
    axes[1].set_ylabel("axis phase")
    axes[2].plot(action_norms)
    axes[2].set_ylabel("action norm")
    axes[3].plot(desired_hits)
    axes[3].set_ylabel("desired hit")
    axes[3].set_xlabel("step")
    fig.suptitle(f"M3.2.11 keyframe relay score={row['score']:.1f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_best_artifacts(output_dir: Path, row: dict[str, Any], traces: dict[str, np.ndarray] | None) -> None:
    if traces is None:
        return
    imageio.mimsave(output_dir / "best_keyframe_relay.gif", list(traces["frames"]), fps=25)
    plot_trace(output_dir / "best_keyframe_relay_trace.png", traces, row)
    np.savez_compressed(
        output_dir / "best_keyframe_relay_rollout.npz",
        centers=traces["centers"],
        axes=traces["axes"],
        axis_angles=traces["axis_angles"],
        action_norms=traces["action_norms"],
        desired_hits=traces["desired_hits"],
        phase_labels=traces["phase_labels"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/keyframe_relay_m3_2_11",
    )
    parser.add_argument("--render-every", type=int, default=5)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes = load_keyframes(seed_dir)

    rows: list[dict[str, Any]] = []
    configs: list[dict[str, float]] = []
    for segment_seconds in [0.16, 0.22, 0.30, 0.42, 0.60]:
        for start_hold_seconds in [0.00, 0.08, 0.16]:
            for tracking_gain in [0.45, 0.75, 1.10, 1.60]:
                for action_delta_limit in [0.010, 0.018, 0.026, 0.035]:
                    configs.append(
                        {
                            "segment_seconds": float(segment_seconds),
                            "start_hold_seconds": float(start_hold_seconds),
                            "end_hold_seconds": 0.10,
                            "tracking_gain": float(tracking_gain),
        "action_delta_limit": float(action_delta_limit),
                            "hard_hold_hand": False,
                        }
                    )
                    configs.append(
                        {
                            "segment_seconds": float(segment_seconds),
                            "start_hold_seconds": float(start_hold_seconds),
                            "end_hold_seconds": 0.10,
                            "tracking_gain": float(tracking_gain),
                            "action_delta_limit": float(action_delta_limit),
                            "hard_hold_hand": True,
                        }
                    )

    for config in configs:
        row, _ = evaluate_relay(keyframes, config, render=False, render_every=args.render_every)
        rows.append(row)
    rows.sort(key=lambda x: x["score"], reverse=True)

    best_row, best_trace = evaluate_relay(keyframes, rows[0], render=True, render_every=args.render_every)
    best_row["score"] = rows[0]["score"]
    save_best_artifacts(output_dir, best_row, best_trace)

    write_csv(output_dir / "keyframe_relay_sweep.csv", rows)
    summary = {
        "stage": "M3.2.11 scripted keyframe relay validation",
        "seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "num_configs": len(configs),
        "best": best_row,
        "top10": rows[:10],
        "artifacts": {
            "csv": str(output_dir / "keyframe_relay_sweep.csv"),
            "best_gif": str(output_dir / "best_keyframe_relay.gif"),
            "best_trace": str(output_dir / "best_keyframe_relay_trace.png"),
            "best_npz": str(output_dir / "best_keyframe_relay_rollout.npz"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.11 Scripted Keyframe Relay Validation",
        "",
        "This stage starts from P1 and only controls hand joint targets toward P2-P5. The pen is not teleported after reset.",
        "",
        "Best config:",
        "",
        "```json",
        json.dumps({k: best_row[k] for k in [
            "score",
            "segment_seconds",
            "start_hold_seconds",
            "tracking_gain",
            "action_delta_limit",
            "hard_hold_hand",
            "survival_seconds",
            "total_duration",
            "terminated",
            "progress_ratio",
            "peak_progress_ratio",
            "desired_contact_ratio",
            "active_contact_ratio",
            "xy_drift",
            "min_height",
            "max_penetration",
        ]}, indent=2, ensure_ascii=False),
        "```",
        "",
        "Interpretation:",
        "",
        "- If survival is short, the current keyframes are still mostly geometric seeds.",
        "- If peak_progress_ratio is positive, the relay can move the pen in the intended direction before failure.",
        "- If desired_contact_ratio is low, the handover contact schedule is not yet physically reliable.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
