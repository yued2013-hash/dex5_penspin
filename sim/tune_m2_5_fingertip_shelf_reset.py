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
from view_m2_mujoco import DEFAULT_MODEL


TIP_KEYWORDS = ("tip_proxy",)


def set_fingertip_shelf_qpos(
    env: Dex5PenSpinMinimalEnv,
    mcp_scale: float,
    pip_scale: float,
    dip_scale: float,
    thumb_scale: float,
    roll_spread: float,
) -> np.ndarray:
    q = np.zeros(20, dtype=np.float32)
    # Index/middle/ring/pinky: create a shelf with relatively high MCP/PIP
    # flexion and softer DIP. Roll offsets spread the fingertips under the pen.
    base_flex = {
        "mcp": 0.80 * mcp_scale,
        "pip": 0.82 * pip_scale,
        "dip": 0.52 * dip_scale,
    }
    rolls = [-roll_spread, -0.35 * roll_spread, 0.35 * roll_spread, roll_spread]
    for finger_i, base in enumerate([0, 4, 8, 12]):
        q[base + 0] = rolls[finger_i]
        q[base + 1] = base_flex["mcp"]
        q[base + 2] = base_flex["pip"]
        q[base + 3] = base_flex["dip"]

    # Thumb as opposing support, not a hard clamp.
    q[16] = 0.20
    q[17] = 0.78 * thumb_scale
    q[18] = 0.68 * thumb_scale
    q[19] = 0.48 * thumb_scale

    q = np.clip(q, env.joint_low, env.joint_high)
    env.target_qpos = q.astype(np.float32)
    env.data.qpos[env.qpos_adr] = env.target_qpos
    env.data.ctrl[:] = env.target_qpos
    mujoco.mj_forward(env.model, env.data)
    return env.target_qpos.copy()


def pen_contact_names(env: Dex5PenSpinMinimalEnv) -> list[str]:
    names: list[str] = []
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        if "pen" in g1 or "pen" in g2:
            names.append(g2 if "pen" in g1 else g1)
    return names


def evaluate_candidate(args: argparse.Namespace, cfg: dict) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=(cfg["x"], cfg["y"], cfg["z"]),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
    )
    _, info = env.reset(seed=args.seed)
    qpos = set_fingertip_shelf_qpos(
        env,
        cfg["mcp_scale"],
        cfg["pip_scale"],
        cfg["dip_scale"],
        cfg["thumb_scale"],
        cfg["roll_spread"],
    )
    center0 = env.get_state().pen_center.copy()
    centers = [center0]
    counter: Counter[str] = Counter()
    ncons = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        centers.append(info["pen_center"])
        contacts = pen_contact_names(env)
        counter.update(contacts)
        ncons.append(int(env.data.ncon))
        step += 1
    env.close()

    centers_np = np.asarray(centers, dtype=np.float64)
    total_pen_contacts = sum(counter.values())
    tip_contacts = sum(v for k, v in counter.items() if any(s in k for s in TIP_KEYWORDS))
    palm_contacts = total_pen_contacts - tip_contacts
    xy_drift = float(np.linalg.norm(centers_np[-1, :2] - centers_np[0, :2]))
    ncon_np = np.asarray(ncons, dtype=np.float64) if ncons else np.zeros(1)
    tip_ratio = float(tip_contacts / total_pen_contacts) if total_pen_contacts else 0.0
    palm_ratio = float(palm_contacts / total_pen_contacts) if total_pen_contacts else 0.0
    survival = float(step / 100.0)
    score = 7.0 * tip_ratio - 3.0 * palm_ratio + 0.8 * survival - 35.0 * xy_drift
    if terminated:
        score -= 1.0
    return {
        **cfg,
        "steps": int(step),
        "survival_seconds": survival,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "xy_drift": xy_drift,
        "min_z": float(np.min(centers_np[:, 2])),
        "final_z": float(centers_np[-1, 2]),
        "mean_contacts": float(np.mean(ncon_np)),
        "max_contacts": int(np.max(ncon_np)),
        "total_pen_contacts": int(total_pen_contacts),
        "tip_contact_ratio": tip_ratio,
        "palm_contact_ratio": palm_ratio,
        "top_pen_contact_geoms": counter.most_common(10),
        "qpos": qpos.tolist(),
        "score": float(score),
    }


def rollout_visual(args: argparse.Namespace, row: dict) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=(row["x"], row["y"], row["z"]),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
    )
    _, _ = env.reset(seed=args.seed)
    qpos = set_fingertip_shelf_qpos(
        env,
        row["mcp_scale"],
        row["pip_scale"],
        row["dip_scale"],
        row["thumb_scale"],
        row["roll_spread"],
    )
    frames = [env.render()]
    centers = [env.get_state().pen_center.copy()]
    ncons = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        centers.append(info["pen_center"])
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
        "ncons": np.asarray(ncons, dtype=np.float32),
        "qpos": qpos,
    }


def plot_trace(visual: dict, path: Path, title: str) -> None:
    centers = visual["centers"]
    ncons = visual["ncons"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t = np.arange(len(centers))
    axes[0].plot(t, centers[:, 0], label="x")
    axes[0].plot(t, centers[:, 1], label="y")
    axes[0].plot(t, centers[:, 2], label="z")
    axes[0].set_title(title)
    axes[0].set_ylabel("pen center")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(np.arange(len(ncons)), ncons, label="ncon")
    axes[1].set_ylabel("contacts")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "rank",
        "score",
        "x",
        "y",
        "z",
        "mcp_scale",
        "pip_scale",
        "dip_scale",
        "thumb_scale",
        "roll_spread",
        "steps",
        "survival_seconds",
        "terminated",
        "truncated",
        "xy_drift",
        "min_z",
        "final_z",
        "mean_contacts",
        "max_contacts",
        "total_pen_contacts",
        "tip_contact_ratio",
        "palm_contact_ratio",
        "top_pen_contact_geoms",
        "qpos",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {"rank": i}
            out.update(row)
            out["top_pen_contact_geoms"] = json.dumps(row["top_pen_contact_geoms"], ensure_ascii=False)
            out["qpos"] = json.dumps(row["qpos"])
            writer.writerow(out)


def viewer_command(row: dict) -> str:
    qpos_str = " ".join(f"{v:.6f}" for v in row["qpos"])
    return (
        "python dex5_penspin/sim/view_m2_mujoco.py "
        "--policy hold --duration 20 --realtime "
        f"--pen-init-pos {row['x']} {row['y']} {row['z']} "
        f"--custom-qpos {qpos_str}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_5_fingertip_shelf_reset")
    parser.add_argument("--episode-seconds", type=float, default=2.0)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.search == "full":
        grid = {
            "x": [-0.040, -0.030, -0.020],
            "y": [0.055, 0.065, 0.075, 0.085],
            "z": [0.105, 0.115, 0.125, 0.135],
            "mcp_scale": [1.0, 1.2, 1.4],
            "pip_scale": [1.0, 1.2, 1.4],
            "dip_scale": [0.7, 1.0],
            "thumb_scale": [0.9, 1.1],
            "roll_spread": [0.0, 0.12],
        }
    else:
        grid = {
            "x": [-0.035, -0.025, -0.015],
            "y": [0.065, 0.075, 0.085],
            "z": [0.110, 0.120, 0.130],
            "mcp_scale": [1.0, 1.3, 1.6],
            "pip_scale": [1.0, 1.3],
            "dip_scale": [0.6, 0.9],
            "thumb_scale": [0.9, 1.1],
            "roll_spread": [0.0, 0.12],
        }

    configs = []
    for x in grid["x"]:
        for y in grid["y"]:
            for z in grid["z"]:
                for mcp_scale in grid["mcp_scale"]:
                    for pip_scale in grid["pip_scale"]:
                        for dip_scale in grid["dip_scale"]:
                            for thumb_scale in grid["thumb_scale"]:
                                for roll_spread in grid["roll_spread"]:
                                    configs.append(
                                        {
                                            "x": x,
                                            "y": y,
                                            "z": z,
                                            "mcp_scale": mcp_scale,
                                            "pip_scale": pip_scale,
                                            "dip_scale": dip_scale,
                                            "thumb_scale": thumb_scale,
                                            "roll_spread": roll_spread,
                                        }
                                    )
    print("M2.5 fingertip shelf reset tuning")
    print(f"search: {args.search}")
    print(f"candidates: {len(configs)}")
    rows = []
    for i, cfg in enumerate(configs):
        rows.append(evaluate_candidate(args, cfg))
        if (i + 1) % 250 == 0:
            print(f"  evaluated {i + 1}/{len(configs)}", flush=True)
    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[: args.top_k]
    write_csv(output_dir / "all_candidates.csv", rows)
    write_csv(output_dir / "top_candidates.csv", top)

    visuals = {}
    for i, row in enumerate(top):
        candidate_dir = output_dir / f"top_{i:02d}_tip{row['tip_contact_ratio']:.2f}_palm{row['palm_contact_ratio']:.2f}"
        candidate_dir.mkdir(exist_ok=True)
        visual = rollout_visual(args, row)
        imageio.mimsave(candidate_dir / "rollout.gif", visual["frames"], duration=0.08)
        imageio.imwrite(candidate_dir / "initial_frame.png", visual["frames"][0])
        imageio.imwrite(candidate_dir / "final_frame.png", visual["frames"][-1])
        plot_trace(visual, candidate_dir / "trace.png", f"M2.5 top {i}")
        visuals[candidate_dir.name] = {
            "summary": row,
            "viewer_command": viewer_command(row),
        }
    summary = {
        "model_path": args.model_path,
        "episode_seconds": args.episode_seconds,
        "candidate_count": len(configs),
        "best": top[0],
        "top": top,
        "visuals": visuals,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = ["# M2.5 Fingertip Shelf Reset", ""]
    for i, row in enumerate(top):
        readme += [
            f"## Top {i}",
            "",
            "```text",
            json.dumps(row, indent=2, ensure_ascii=False),
            "```",
            "",
            "Viewer:",
            "",
            "```bash",
            viewer_command(row),
            "```",
            "",
        ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("best:", json.dumps(top[0], indent=2, ensure_ascii=False))
    print(f"saved results to: {output_dir}")
    print("best viewer command:")
    print(viewer_command(top[0]))


if __name__ == "__main__":
    main()
