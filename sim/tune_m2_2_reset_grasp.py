from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv


DEFAULT_MODEL = "dex5_penspin/sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"


def evaluate_candidate(
    model_path: str,
    pen_init_pos: tuple[float, float, float],
    grasp_scale: float,
    episode_seconds: float,
    seed: int,
) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=pen_init_pos,
        grasp_scale=grasp_scale,
        init_noise_scale=0.0,
        episode_seconds=episode_seconds,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)
    centers = [info["pen_center"]]
    phases = [info["rotation_phase"]]
    contacts = []
    terminated = False
    truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        total_reward += reward
        centers.append(info["pen_center"])
        phases.append(info["rotation_phase"])
        contacts.append(int(env.data.ncon))
    env.close()

    centers_np = np.asarray(centers)
    contacts_np = np.asarray(contacts, dtype=np.int32) if contacts else np.zeros(1, dtype=np.int32)
    return {
        "x": float(pen_init_pos[0]),
        "y": float(pen_init_pos[1]),
        "z": float(pen_init_pos[2]),
        "grasp_scale": float(grasp_scale),
        "steps": int(len(centers_np) - 1),
        "survival_seconds": float((len(centers_np) - 1) / env.control_hz),
        "return": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "min_z": float(np.min(centers_np[:, 2])),
        "final_z": float(centers_np[-1, 2]),
        "xy_drift": float(np.linalg.norm(centers_np[-1, :2] - centers_np[0, :2])),
        "mean_contacts": float(np.mean(contacts_np)),
        "max_contacts": int(np.max(contacts_np)),
        "rotation_phase_abs": float(abs(phases[-1])),
    }


def rollout_for_visuals(
    model_path: str,
    pen_init_pos: tuple[float, float, float],
    grasp_scale: float,
    episode_seconds: float,
    seed: int,
    render_every: int,
) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=pen_init_pos,
        grasp_scale=grasp_scale,
        init_noise_scale=0.0,
        episode_seconds=episode_seconds,
        seed=seed,
    )
    _, info = env.reset(seed=seed)
    frames = [env.render()]
    centers = [info["pen_center"]]
    phases = [info["rotation_phase"]]
    contacts = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        centers.append(info["pen_center"])
        phases.append(info["rotation_phase"])
        contacts.append(int(env.data.ncon))
        step += 1
        if step % render_every == 0:
            frames.append(env.render())
    if step % render_every != 0:
        frames.append(env.render())
    env.close()
    return {
        "frames": frames,
        "centers": np.asarray(centers),
        "phases": np.asarray(phases),
        "contacts": np.asarray(contacts, dtype=np.int32),
        "steps": step,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "rank",
        "x",
        "y",
        "z",
        "grasp_scale",
        "steps",
        "survival_seconds",
        "return",
        "terminated",
        "truncated",
        "min_z",
        "final_z",
        "xy_drift",
        "mean_contacts",
        "max_contacts",
        "rotation_phase_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {"rank": i}
            out.update(row)
            writer.writerow(out)


def plot_rollout(rollout: dict, path: Path, title: str) -> None:
    centers = rollout["centers"]
    phases = rollout["phases"]
    contacts = rollout["contacts"]
    t = np.arange(len(centers))

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, centers[:, 0], label="x")
    axes[0].plot(t, centers[:, 1], label="y")
    axes[0].plot(t, centers[:, 2], label="z")
    axes[0].set_ylabel("pen center (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(t, phases, label="rotation_phase")
    axes[1].set_ylabel("phase (rad)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if len(contacts):
        axes[2].plot(np.arange(len(contacts)), contacts, label="ncon")
    axes[2].set_ylabel("contacts")
    axes[2].set_xlabel("step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_2_reset_grasp_tuning")
    parser.add_argument("--episode-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--render-every", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_values = [-0.045, -0.035, -0.025, -0.015, -0.005]
    y_values = [0.035, 0.045, 0.055, 0.065, 0.075]
    z_values = [0.080, 0.090, 0.100, 0.110, 0.120]
    grasp_scales = [1.0, 1.2, 1.4, 1.6]

    rows = []
    total = len(x_values) * len(y_values) * len(z_values) * len(grasp_scales)
    print(f"M2.2 reset/grasp/contact tuning")
    print(f"model_path: {args.model_path}")
    print(f"candidates: {total}")
    idx = 0
    for grasp_scale in grasp_scales:
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    idx += 1
                    row = evaluate_candidate(
                        args.model_path,
                        (x, y, z),
                        grasp_scale,
                        args.episode_seconds,
                        args.seed,
                    )
                    rows.append(row)
                    if idx % 50 == 0:
                        print(f"  evaluated {idx}/{total}")

    rows.sort(key=lambda r: (r["steps"], r["min_z"], -r["xy_drift"], r["mean_contacts"]), reverse=True)
    top_rows = rows[: args.top_k]
    write_csv(output_dir / "all_candidates.csv", rows)
    write_csv(output_dir / "top_candidates.csv", top_rows)

    best = top_rows[0]
    print("best:", best)
    visual = rollout_for_visuals(
        args.model_path,
        (best["x"], best["y"], best["z"]),
        best["grasp_scale"],
        args.episode_seconds,
        args.seed,
        args.render_every,
    )

    imageio.mimsave(output_dir / "best_hold_rollout.gif", visual["frames"], duration=0.08)
    imageio.imwrite(output_dir / "best_initial_frame.png", visual["frames"][0])
    imageio.imwrite(output_dir / "best_final_frame.png", visual["frames"][-1])
    plot_rollout(visual, output_dir / "best_hold_rollout_trace.png", "M2.2 best hold rollout")

    summary = {
        "model_path": args.model_path,
        "episode_seconds": args.episode_seconds,
        "num_candidates": total,
        "top_k": args.top_k,
        "best": best,
        "outputs": {
            "all_candidates": "all_candidates.csv",
            "top_candidates": "top_candidates.csv",
            "best_gif": "best_hold_rollout.gif",
            "best_initial_frame": "best_initial_frame.png",
            "best_final_frame": "best_final_frame.png",
            "best_trace": "best_hold_rollout_trace.png",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved results to: {output_dir}")


if __name__ == "__main__":
    main()
