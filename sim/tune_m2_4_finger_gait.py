from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from view_m2_mujoco import DEFAULT_MODEL, DEFAULT_PEN_INIT_POS, action_for_policy


PALM_LIKE_GEOMS = {"geom2", "geom10", "geom19"}
TIP_KEYWORDS = ("tip_proxy",)


def pen_contact_names(env: Dex5PenSpinMinimalEnv) -> list[str]:
    names: list[str] = []
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        if "pen" in g1 or "pen" in g2:
            names.append(g2 if "pen" in g1 else g1)
    return names


def evaluate_config(args: argparse.Namespace, action_scale: float, phase_preset: str, center_gain: float) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=args.grasp_scale,
        init_noise_scale=0.0,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
    )
    _, info = env.reset(seed=args.seed)
    center0 = np.asarray(info["pen_center"], dtype=np.float64)
    centers = [center0]
    phases = [float(info["rotation_phase"])]
    ncons = []
    counter: Counter[str] = Counter()
    terminated = False
    truncated = False
    step = 0
    total_reward = 0.0
    while not (terminated or truncated):
        action = action_for_policy(
            env,
            "finger_gait",
            step,
            action_scale,
            phase_preset=phase_preset,
            center_gain=center_gain,
            target_center=args.pen_init_pos,
            cycle_steps=args.cycle_steps,
        )
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        centers.append(info["pen_center"])
        phases.append(float(info["rotation_phase"]))
        contacts = pen_contact_names(env)
        counter.update(contacts)
        ncons.append(int(env.data.ncon))
        step += 1
    env.close()

    centers_np = np.asarray(centers, dtype=np.float64)
    ncon_np = np.asarray(ncons, dtype=np.float64) if ncons else np.zeros(1)
    total_pen_contacts = sum(counter.values())
    tip_contacts = sum(v for k, v in counter.items() if any(s in k for s in TIP_KEYWORDS))
    palm_contacts = sum(v for k, v in counter.items() if k in PALM_LIKE_GEOMS)
    phase_abs = abs(phases[-1])
    xy_drift = float(np.linalg.norm(centers_np[-1, :2] - centers_np[0, :2]))
    survival = float(step / env.control_hz)
    score = phase_abs - 55.0 * xy_drift + 0.35 * survival
    if terminated:
        score -= 2.0
    if total_pen_contacts:
        score += 0.8 * (tip_contacts / total_pen_contacts)
        score -= 0.4 * (palm_contacts / total_pen_contacts)

    return {
        "action_scale": action_scale,
        "phase_preset": phase_preset,
        "center_gain": center_gain,
        "cycle_steps": args.cycle_steps,
        "steps": int(step),
        "survival_seconds": survival,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "return": float(total_reward),
        "rotation_phase_abs": float(phase_abs),
        "rotation_rounds": float(phase_abs / (2.0 * np.pi)),
        "xy_drift": xy_drift,
        "min_z": float(np.min(centers_np[:, 2])),
        "final_z": float(centers_np[-1, 2]),
        "mean_contacts": float(np.mean(ncon_np)),
        "max_contacts": int(np.max(ncon_np)),
        "total_pen_contacts": int(total_pen_contacts),
        "tip_contact_ratio": float(tip_contacts / total_pen_contacts) if total_pen_contacts else 0.0,
        "palm_contact_ratio": float(palm_contacts / total_pen_contacts) if total_pen_contacts else 0.0,
        "top_pen_contact_geoms": counter.most_common(10),
        "score": float(score),
    }


def rollout_visual(args: argparse.Namespace, config: dict) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=args.grasp_scale,
        init_noise_scale=0.0,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
    )
    _, info = env.reset(seed=args.seed)
    frames = [env.render()]
    centers = [info["pen_center"]]
    phases = [info["rotation_phase"]]
    ncons = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        action = action_for_policy(
            env,
            "finger_gait",
            step,
            config["action_scale"],
            phase_preset=config["phase_preset"],
            center_gain=config["center_gain"],
            target_center=args.pen_init_pos,
            cycle_steps=args.cycle_steps,
        )
        _, _, terminated, truncated, info = env.step(action)
        centers.append(info["pen_center"])
        phases.append(info["rotation_phase"])
        ncons.append(int(env.data.ncon))
        step += 1
        if step % args.render_every == 0:
            frames.append(env.render())
    if step % args.render_every:
        frames.append(env.render())
    env.close()
    return {
        "frames": frames,
        "centers": np.asarray(centers, dtype=np.float32),
        "phases": np.asarray(phases, dtype=np.float32),
        "ncons": np.asarray(ncons, dtype=np.float32),
    }


def plot_trace(visual: dict, path: Path, title: str) -> None:
    centers = visual["centers"]
    phases = visual["phases"]
    ncons = visual["ncons"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    t = np.arange(len(centers))
    axes[0].plot(t, centers[:, 0], label="x")
    axes[0].plot(t, centers[:, 1], label="y")
    axes[0].plot(t, centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(np.arange(len(phases)), phases, label="phase")
    axes[1].plot(np.arange(len(phases)), phases / (2.0 * np.pi), label="rounds")
    axes[1].set_ylabel("rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(np.arange(len(ncons)), ncons, label="ncon")
    axes[2].set_ylabel("contacts")
    axes[2].set_xlabel("step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "rank",
        "score",
        "action_scale",
        "phase_preset",
        "center_gain",
        "cycle_steps",
        "steps",
        "survival_seconds",
        "terminated",
        "truncated",
        "rotation_phase_abs",
        "rotation_rounds",
        "xy_drift",
        "min_z",
        "final_z",
        "mean_contacts",
        "max_contacts",
        "total_pen_contacts",
        "tip_contact_ratio",
        "palm_contact_ratio",
        "return",
        "top_pen_contact_geoms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {"rank": i}
            out.update(row)
            out["top_pen_contact_geoms"] = json.dumps(row["top_pen_contact_geoms"], ensure_ascii=False)
            writer.writerow(out)


def viewer_command(args: argparse.Namespace, config: dict) -> str:
    return (
        "python dex5_penspin/sim/view_m2_mujoco.py "
        "--policy finger_gait "
        f"--action-scale {config['action_scale']} "
        f"--phase-preset {config['phase_preset']} "
        f"--center-gain {config['center_gain']} "
        f"--cycle-steps {config['cycle_steps']} "
        "--duration 20 --realtime"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_4_finger_gait_tuning")
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--action-scales", nargs="+", type=float, default=[0.001, 0.002, 0.003, 0.004])
    parser.add_argument("--phase-presets", nargs="+", default=["m2_3", "quadrature", "paired", "thumb_late"])
    parser.add_argument("--center-gains", nargs="+", type=float, default=[0.0, 0.02, 0.04])
    parser.add_argument("--cycle-steps", type=int, default=220)
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=DEFAULT_PEN_INIT_POS)
    parser.add_argument("--grasp-scale", type=float, default=1.4)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(args.action_scales) * len(args.phase_presets) * len(args.center_gains)
    print("M2.4 finger_gait tuning")
    print(f"model_path: {args.model_path}")
    print(f"candidate count: {total}")
    for action_scale in args.action_scales:
        for phase_preset in args.phase_presets:
            for center_gain in args.center_gains:
                row = evaluate_config(args, action_scale, phase_preset, center_gain)
                rows.append(row)
                print(
                    f"scale={action_scale:.3f} preset={phase_preset:10s} center={center_gain:.3f} "
                    f"steps={row['steps']:3d} phase={row['rotation_phase_abs']:.3f} "
                    f"drift={row['xy_drift']:.3f} score={row['score']:.3f}"
                )

    rows.sort(key=lambda r: r["score"], reverse=True)
    top_rows = rows[: args.top_k]
    write_csv(output_dir / "all_candidates.csv", rows)
    write_csv(output_dir / "top_candidates.csv", top_rows)

    visuals = {}
    for i, config in enumerate(top_rows):
        candidate_dir = output_dir / f"top_{i:02d}_{config['phase_preset']}_s{config['action_scale']}_cg{config['center_gain']}"
        candidate_dir.mkdir(exist_ok=True)
        visual = rollout_visual(args, config)
        imageio.mimsave(candidate_dir / "rollout.gif", visual["frames"], duration=0.08)
        imageio.imwrite(candidate_dir / "initial_frame.png", visual["frames"][0])
        imageio.imwrite(candidate_dir / "final_frame.png", visual["frames"][-1])
        plot_trace(visual, candidate_dir / "trace.png", f"M2.4 top {i}: {config['phase_preset']}")
        visuals[str(candidate_dir.name)] = {
            "summary": config,
            "viewer_command": viewer_command(args, config),
        }

    summary = {
        "model_path": args.model_path,
        "episode_seconds": args.episode_seconds,
        "pen_init_pos": args.pen_init_pos,
        "grasp_scale": args.grasp_scale,
        "candidate_count": total,
        "best": top_rows[0],
        "top": top_rows,
        "visuals": visuals,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    readme_lines = [
        "# M2.4 Finger Gait Tuning",
        "",
        "Each top candidate has a GIF and an equivalent MuJoCo viewer command.",
        "",
    ]
    for i, row in enumerate(top_rows):
        readme_lines.extend(
            [
                f"## Top {i}",
                "",
                "```text",
                json.dumps(row, indent=2, ensure_ascii=False),
                "```",
                "",
                "Viewer:",
                "",
                "```bash",
                viewer_command(args, row),
                "```",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"saved results to: {output_dir}")
    print("best viewer command:")
    print(viewer_command(args, top_rows[0]))


if __name__ == "__main__":
    main()
