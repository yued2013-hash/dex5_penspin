from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from build_m3_2_9_photo_five_phase_debug import smoothstep
from run_m3_2_11_keyframe_relay import (
    axis_angle_xy,
    keyframe_target_phases,
    load_keyframes,
    make_env,
    reset_to_keyframe,
)
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


def interp_qpos(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    t = smoothstep(float(np.clip(t, 0.0, 1.0)))
    return (1.0 - t) * q0 + t * q1


def target_from_cursor(keyframes: list[dict[str, Any]], cursor: float) -> tuple[np.ndarray, int, float, str]:
    qposes = [np.asarray(k["hand_qpos"], dtype=np.float64) for k in keyframes]
    max_cursor = len(keyframes) - 1
    cursor = float(np.clip(cursor, 0.0, max_cursor))
    if cursor >= max_cursor:
        idx = len(keyframes) - 1
        return qposes[idx], idx, 1.0, f"P{keyframes[idx]['phase']}_hold"
    seg = int(np.floor(cursor))
    phase_t = float(cursor - seg)
    target = interp_qpos(qposes[seg], qposes[seg + 1], phase_t)
    return target, seg, phase_t, f"P{keyframes[seg]['phase']}->P{keyframes[seg + 1]['phase']}"


def desired_for_cursor(keyframes: list[dict[str, Any]], seg: int, phase_t: float) -> set[str]:
    idx = min(seg + (1 if phase_t >= 0.5 else 0), len(keyframes) - 1)
    return set(keyframes[idx].get("desired_contacts", []))


def contact_quality(fingers: set[str], desired: set[str]) -> dict[str, Any]:
    desired_hits = len(fingers.intersection(desired))
    undesired_hits = len(fingers.difference(desired))
    missing = sorted(desired.difference(fingers))
    active = len(fingers) >= 2
    exact = bool(desired and desired.issubset(fingers))
    if len(desired) >= 3:
        near = bool(active and desired_hits >= len(desired) - 1)
    else:
        near = bool(desired and desired_hits >= len(desired))
    support = bool(active and desired_hits >= 2)
    if exact:
        gate_weight = 1.0
        gate_label = "exact"
    elif near:
        gate_weight = 0.65
        gate_label = "near"
    elif support:
        gate_weight = 0.25
        gate_label = "support"
    elif active:
        gate_weight = 0.05
        gate_label = "active_wrong"
    else:
        gate_weight = 0.0
        gate_label = "lost"
    return {
        "desired_hits": desired_hits,
        "undesired_hits": undesired_hits,
        "missing": missing,
        "active": active,
        "exact": exact,
        "near": near,
        "support": support,
        "gate_weight": gate_weight,
        "gate_label": gate_label,
    }


def phase_delta_metrics(prev_angle: float | None, angle: float, direction: float) -> tuple[float, float, float]:
    if prev_angle is None:
        return 0.0, 0.0, 0.0
    delta = float(np.arctan2(np.sin(angle - prev_angle), np.cos(angle - prev_angle)))
    directed = float(direction * delta)
    return directed, max(0.0, directed), max(0.0, -directed)


def evaluate_contact_gated_relay(
    keyframes: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    render: bool,
    render_every: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    max_seconds = float(config["max_seconds"])
    env = make_env(
        keyframes[0],
        duration=max_seconds,
        render=render,
        action_delta_limit=float(config["action_delta_limit"]),
        hard_hold_hand=bool(config["hard_hold_hand"]),
    )
    env.reset(seed=0)
    reset_to_keyframe(env, keyframes[0])

    target_phases = keyframe_target_phases(keyframes)
    target_delta = float(target_phases[-1] - target_phases[0])
    direction = float(np.sign(target_delta) or 1.0)
    target_abs = abs(target_delta)
    center0 = env.get_state().pen_center.copy()

    cursor = 0.0
    start_hold_steps = int(round(float(config["start_hold_seconds"]) * env.control_hz))
    base_cursor_inc = 1.0 / max(1.0, float(config["segment_seconds"]) * env.control_hz)

    exact_frames = 0
    near_frames = 0
    support_frames = 0
    active_frames = 0
    lost_frames = 0
    total_gate_weight = 0.0
    gated_positive = 0.0
    ungated_positive = 0.0
    negative_progress = 0.0
    invalid_positive = 0.0
    max_drawdown = 0.0
    running_progress = 0.0
    running_max = 0.0
    max_penetration = 0.0
    min_height = float(center0[2])
    returns = 0.0
    terminated = False
    truncated = False

    contact_counter: Counter[str] = Counter()
    gate_counter: Counter[str] = Counter()
    segment_steps: Counter[str] = Counter()
    segment_gate_counter: dict[str, Counter[str]] = defaultdict(Counter)
    exact_contact_counter: Counter[str] = Counter()

    frames: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    axis_angles: list[float] = []
    cursor_trace: list[float] = []
    gate_weight_trace: list[float] = []
    gated_progress_trace: list[float] = []
    ungated_progress_trace: list[float] = []
    action_norms: list[float] = []
    gate_labels: list[str] = []
    phase_labels: list[str] = []

    if render:
        frames.append(env.render())

    prev_angle: float | None = axis_angle_xy(env.get_state().pen_axis)
    max_steps = int(round(max_seconds * env.control_hz))
    for step in range(max_steps):
        target, seg, phase_t, phase_label = target_from_cursor(keyframes, cursor)
        delta = float(config["tracking_gain"]) * (target - env.target_qpos.astype(np.float64))
        action = np.clip(delta, -float(config["action_delta_limit"]), float(config["action_delta_limit"])).astype(np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)

        fingers, _, penetration = contact_snapshot(env)
        desired = desired_for_cursor(keyframes, seg, phase_t)
        quality = contact_quality(fingers, desired)
        gate_weight = float(quality["gate_weight"])
        angle = axis_angle_xy(np.asarray(info["pen_axis"], dtype=np.float64))
        directed_delta, positive_delta, negative_delta = phase_delta_metrics(prev_angle, angle, direction)
        prev_angle = angle

        ungated_positive += positive_delta
        gated_positive += positive_delta * gate_weight
        invalid_positive += positive_delta * (1.0 - gate_weight)
        negative_progress += negative_delta
        running_progress += directed_delta
        running_max = max(running_max, running_progress)
        max_drawdown = max(max_drawdown, running_max - running_progress)

        if quality["exact"]:
            exact_frames += 1
        if quality["near"]:
            near_frames += 1
        if quality["support"]:
            support_frames += 1
        if quality["active"]:
            active_frames += 1
        if gate_weight <= 0.0:
            lost_frames += 1
        total_gate_weight += gate_weight
        gate_counter[str(quality["gate_label"])] += 1
        segment_gate_counter[phase_label][str(quality["gate_label"])] += 1
        segment_steps[phase_label] += 1
        exact_contact_counter[",".join(sorted(fingers)) or "none"] += 1
        for finger in fingers:
            contact_counter[finger] += 1
        max_penetration = max(max_penetration, float(penetration))
        center = np.asarray(info["pen_center"], dtype=np.float64).copy()
        min_height = min(min_height, float(center[2]))

        if step >= start_hold_steps and cursor < len(keyframes) - 1:
            if config["controller_mode"] == "timeline":
                advance_weight = 1.0
            else:
                if quality["exact"]:
                    advance_weight = 1.0
                elif quality["near"]:
                    advance_weight = float(config["near_advance"])
                elif quality["support"]:
                    advance_weight = float(config["support_advance"])
                elif quality["active"]:
                    advance_weight = float(config["wrong_contact_advance"])
                else:
                    advance_weight = float(config["lost_advance"])
            cursor += base_cursor_inc * advance_weight
            cursor = min(cursor, float(len(keyframes) - 1))

        centers.append(center)
        axis_angles.append(angle)
        cursor_trace.append(cursor)
        gate_weight_trace.append(gate_weight)
        gated_progress_trace.append(gated_positive)
        ungated_progress_trace.append(ungated_positive)
        action_norms.append(float(np.linalg.norm(action)))
        gate_labels.append(str(quality["gate_label"]))
        phase_labels.append(phase_label)

        if render and step % render_every == 0:
            frame = env.render()
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            text = (
                f"{phase_label} cursor={cursor:.2f} gate={quality['gate_label']} "
                f"contacts={','.join(sorted(fingers)) or 'none'}"
            )
            draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
            draw.text((8, 8), text, fill=(0, 0, 0))
            frames.append(np.asarray(image))
        if terminated or truncated:
            break

    steps = max(1, len(centers))
    centers_np = np.asarray(centers, dtype=np.float64) if centers else np.zeros((1, 3), dtype=np.float64)
    drift = float(np.linalg.norm(centers_np[-1] - center0))
    xy_drift = float(np.linalg.norm(centers_np[-1, :2] - center0[:2]))
    z_delta = float(centers_np[-1, 2] - center0[2])
    survival_seconds = float(steps / env.control_hz)
    gated_progress_ratio = float(gated_positive / (target_abs + 1e-9))
    ungated_progress_ratio = float(ungated_positive / (target_abs + 1e-9))
    invalid_progress_ratio = float(invalid_positive / (target_abs + 1e-9))
    final_progress_ratio = float(running_progress / (target_abs + 1e-9))
    score = (
        900.0 * survival_seconds / max(max_seconds, 1e-9)
        + 1200.0 * gated_progress_ratio
        + 360.0 * (near_frames / steps)
        + 260.0 * (support_frames / steps)
        + 120.0 * (exact_frames / steps)
        + 80.0 * (total_gate_weight / steps)
        - 950.0 * invalid_progress_ratio
        - 420.0 * (negative_progress / (target_abs + 1e-9))
        - 240.0 * (max_drawdown / (target_abs + 1e-9))
        - 900.0 * xy_drift
        - 360.0 * max(0.0, 0.050 - min_height)
        - 260.0 * max(0.0, max_penetration - 0.014)
        - (300.0 if terminated else 0.0)
    )
    if cursor < len(keyframes) - 1:
        score -= 120.0 * (len(keyframes) - 1 - cursor) / max(1.0, len(keyframes) - 1)

    row = {
        **config,
        "score": float(score),
        "return": float(returns),
        "steps": int(steps),
        "survival_seconds": survival_seconds,
        "max_seconds": max_seconds,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "cursor_final": float(cursor),
        "cursor_ratio": float(cursor / max(1.0, len(keyframes) - 1)),
        "gated_positive": float(gated_positive),
        "ungated_positive": float(ungated_positive),
        "invalid_positive": float(invalid_positive),
        "negative_progress": float(negative_progress),
        "running_progress": float(running_progress),
        "gated_progress_ratio": gated_progress_ratio,
        "ungated_progress_ratio": ungated_progress_ratio,
        "invalid_progress_ratio": invalid_progress_ratio,
        "final_progress_ratio": final_progress_ratio,
        "max_drawdown": float(max_drawdown),
        "exact_contact_ratio": float(exact_frames / steps),
        "near_contact_ratio": float(near_frames / steps),
        "support_contact_ratio": float(support_frames / steps),
        "active_contact_ratio": float(active_frames / steps),
        "lost_contact_ratio": float(lost_frames / steps),
        "mean_gate_weight": float(total_gate_weight / steps),
        "drift": drift,
        "xy_drift": xy_drift,
        "z_delta": z_delta,
        "min_height": min_height,
        "max_penetration": float(max_penetration),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_norm_max": float(np.max(action_norms)) if action_norms else 0.0,
        "gate_counter": dict(gate_counter),
        "segment_gate_counter": {label: dict(counter) for label, counter in segment_gate_counter.items()},
        "contact_counter": dict(contact_counter),
        "exact_contact_counter": dict(exact_contact_counter),
    }
    traces = None
    if render:
        traces = {
            "frames": np.asarray(frames, dtype=np.uint8),
            "centers": centers_np,
            "axis_angles": np.asarray(axis_angles, dtype=np.float64),
            "cursor": np.asarray(cursor_trace, dtype=np.float64),
            "gate_weights": np.asarray(gate_weight_trace, dtype=np.float64),
            "gated_progress": np.asarray(gated_progress_trace, dtype=np.float64),
            "ungated_progress": np.asarray(ungated_progress_trace, dtype=np.float64),
            "action_norms": np.asarray(action_norms, dtype=np.float64),
            "gate_labels": np.asarray(gate_labels, dtype=object),
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
        "controller_mode",
        "hard_hold_hand",
        "segment_seconds",
        "start_hold_seconds",
        "max_seconds",
        "tracking_gain",
        "action_delta_limit",
        "near_advance",
        "support_advance",
        "wrong_contact_advance",
        "lost_advance",
        "survival_seconds",
        "terminated",
        "cursor_final",
        "cursor_ratio",
        "gated_progress_ratio",
        "ungated_progress_ratio",
        "invalid_progress_ratio",
        "final_progress_ratio",
        "near_contact_ratio",
        "support_contact_ratio",
        "active_contact_ratio",
        "lost_contact_ratio",
        "mean_gate_weight",
        "xy_drift",
        "z_delta",
        "min_height",
        "max_penetration",
        "action_norm_mean",
        "gate_counter",
        "segment_gate_counter",
        "contact_counter",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {"rank": rank, **row}
            writer.writerow({field: out.get(field, "") for field in fields})


def plot_trace(path: Path, traces: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    centers = traces["centers"]
    axis = np.unwrap(traces["axis_angles"])
    cursor = traces["cursor"]
    gate = traces["gate_weights"]
    gated = traces["gated_progress"]
    ungated = traces["ungated_progress"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(centers[:, 0], label="x")
    axes[0].plot(centers[:, 1], label="y")
    axes[0].plot(centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].legend()
    axes[1].plot(axis - axis[0])
    axes[1].set_ylabel("axis phase")
    axes[2].plot(cursor)
    axes[2].set_ylabel("keyframe cursor")
    axes[3].plot(gate)
    axes[3].set_ylabel("contact gate")
    axes[4].plot(gated, label="gated")
    axes[4].plot(ungated, label="ungated", alpha=0.7)
    axes[4].set_ylabel("positive phase")
    axes[4].set_xlabel("step")
    axes[4].legend()
    fig.suptitle(f"M3.2.13 contact-gated relay score={row['score']:.1f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_best_artifacts(output_dir: Path, row: dict[str, Any], traces: dict[str, np.ndarray] | None) -> None:
    if traces is None:
        return
    imageio.mimsave(output_dir / "best_contact_gated_relay.gif", list(traces["frames"]), fps=25)
    plot_trace(output_dir / "best_contact_gated_relay_trace.png", traces, row)
    np.savez_compressed(
        output_dir / "best_contact_gated_relay_rollout.npz",
        centers=traces["centers"],
        axis_angles=traces["axis_angles"],
        cursor=traces["cursor"],
        gate_weights=traces["gate_weights"],
        gated_progress=traces["gated_progress"],
        ungated_progress=traces["ungated_progress"],
        action_norms=traces["action_norms"],
        gate_labels=traces["gate_labels"],
        phase_labels=traces["phase_labels"],
    )


def save_montage(gif_path: Path, output_path: Path) -> None:
    frames = imageio.mimread(gif_path)
    if not frames:
        return
    idxs = [0, len(frames) // 5, 2 * len(frames) // 5, 3 * len(frames) // 5, 4 * len(frames) // 5, len(frames) - 1]
    images = []
    for idx in idxs:
        image = Image.fromarray(frames[idx]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 120, 26], fill=(255, 255, 255))
        draw.text((6, 6), f"frame {idx}", fill=(0, 0, 0))
        images.append(image)
    w, h = images[0].size
    canvas = Image.new("RGB", (3 * w, 2 * h), (245, 245, 245))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % 3) * w, (i // 3) * h))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/contact_gated_relay_m3_2_13",
    )
    parser.add_argument("--render-every", type=int, default=5)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes = load_keyframes(seed_dir)

    configs: list[dict[str, Any]] = []
    for controller_mode in ["timeline", "contact_gated"]:
        for hard_hold_hand in [False, True]:
            for segment_seconds in [0.14, 0.18, 0.24, 0.32, 0.45]:
                for start_hold_seconds in [0.0, 0.08, 0.16]:
                    for tracking_gain in [0.45, 0.75, 1.10, 1.60]:
                        for action_delta_limit in [0.010, 0.018, 0.026]:
                            for near_advance, support_advance, wrong_advance, lost_advance in [
                                (1.0, 0.35, 0.05, 0.0),
                                (0.75, 0.25, 0.03, 0.0),
                                (0.55, 0.10, 0.00, 0.0),
                            ]:
                                configs.append(
                                    {
                                        "controller_mode": controller_mode,
                                        "hard_hold_hand": bool(hard_hold_hand),
                                        "segment_seconds": float(segment_seconds),
                                        "start_hold_seconds": float(start_hold_seconds),
                                        "max_seconds": float(start_hold_seconds + 4 * segment_seconds + 0.30),
                                        "tracking_gain": float(tracking_gain),
                                        "action_delta_limit": float(action_delta_limit),
                                        "near_advance": float(near_advance),
                                        "support_advance": float(support_advance),
                                        "wrong_contact_advance": float(wrong_advance),
                                        "lost_advance": float(lost_advance),
                                    }
                                )

    rows: list[dict[str, Any]] = []
    for config in configs:
        row, _ = evaluate_contact_gated_relay(keyframes, config, render=False, render_every=args.render_every)
        rows.append(row)
    rows.sort(key=lambda x: x["score"], reverse=True)

    best_row, best_trace = evaluate_contact_gated_relay(keyframes, rows[0], render=True, render_every=args.render_every)
    best_row["score"] = rows[0]["score"]
    write_csv(output_dir / "contact_gated_relay_sweep.csv", rows)
    save_best_artifacts(output_dir, best_row, best_trace)
    save_montage(output_dir / "best_contact_gated_relay.gif", output_dir / "best_contact_gated_relay_montage.png")

    summary = {
        "stage": "M3.2.13 contact-gated directed relay",
        "seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "num_configs": len(configs),
        "best": best_row,
        "top10": rows[:10],
        "artifacts": {
            "csv": str(output_dir / "contact_gated_relay_sweep.csv"),
            "best_gif": str(output_dir / "best_contact_gated_relay.gif"),
            "best_montage": str(output_dir / "best_contact_gated_relay_montage.png"),
            "best_trace": str(output_dir / "best_contact_gated_relay_trace.png"),
            "best_npz": str(output_dir / "best_contact_gated_relay_rollout.npz"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.13 Contact-Gated Directed Relay",
        "",
        "This stage scores phase progress only when contact quality is exact/near/support. Ungated phase progress is penalized as likely ejection.",
        "",
        "Best config:",
        "",
        "```json",
        json.dumps(
            {
                key: best_row[key]
                for key in [
                    "score",
                    "controller_mode",
                    "hard_hold_hand",
                    "segment_seconds",
                    "start_hold_seconds",
                    "tracking_gain",
                    "action_delta_limit",
                    "survival_seconds",
                    "terminated",
                    "cursor_final",
                    "cursor_ratio",
                    "gated_progress_ratio",
                    "ungated_progress_ratio",
                    "invalid_progress_ratio",
                    "near_contact_ratio",
                    "support_contact_ratio",
                    "active_contact_ratio",
                    "lost_contact_ratio",
                    "mean_gate_weight",
                    "xy_drift",
                    "min_height",
                    "max_penetration",
                ]
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "Gate labels:",
        "",
        "- exact: desired contacts are all present.",
        "- near: one desired contact may be missing for 3+ finger sets; 2-finger sets require both.",
        "- support: at least two desired contacts are present.",
        "- active_wrong/lost: not acceptable for oracle progress.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
