from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from view_m2_mujoco import action_for_policy


DEFAULT_MODEL = "dex5_penspin/sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"
DEFAULT_PEN_INIT_POS = (-0.025, 0.065, 0.120)


def contact_names(env: Dex5PenSpinMinimalEnv) -> list[str]:
    names: list[str] = []
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        if "pen" in g1 or "pen" in g2:
            other = g2 if "pen" in g1 else g1
            names.append(other)
    return names


def rollout(policy: str, args: argparse.Namespace, save_frames: bool = False) -> dict:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=args.grasp_scale,
        init_noise_scale=0.0,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
    )
    _, info = env.reset(seed=args.seed)
    frames = [env.render()] if save_frames else []
    centers = [info["pen_center"]]
    phases = [info["rotation_phase"]]
    contacts = []
    contact_counter: Counter[str] = Counter()
    actions = []
    total_reward = 0.0
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        action = action_for_policy(env, policy, step, args.action_scale).astype(np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        centers.append(info["pen_center"])
        phases.append(info["rotation_phase"])
        actions.append(action)
        names = contact_names(env)
        contact_counter.update(names)
        contacts.append({"ncon": int(env.data.ncon), "pen_contacts": names})
        step += 1
        if save_frames and step % args.render_every == 0:
            frames.append(env.render())
    if save_frames and (not frames or step % args.render_every != 0):
        frames.append(env.render())

    centers_np = np.asarray(centers, dtype=np.float32)
    phases_np = np.asarray(phases, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)
    ncon = np.asarray([c["ncon"] for c in contacts], dtype=np.float32) if contacts else np.zeros(1)
    summary = {
        "policy": policy,
        "steps": int(step),
        "survival_seconds": float(step / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "return": float(total_reward),
        "min_z": float(np.min(centers_np[:, 2])),
        "final_z": float(centers_np[-1, 2]),
        "xy_drift": float(np.linalg.norm(centers_np[-1, :2] - centers_np[0, :2])),
        "rotation_phase_abs": float(abs(phases_np[-1])),
        "rotation_rounds": float(abs(phases_np[-1]) / (2.0 * np.pi)),
        "mean_contacts": float(np.mean(ncon)),
        "max_contacts": int(np.max(ncon)),
        "top_pen_contact_geoms": contact_counter.most_common(12),
    }
    env.close()
    return {
        "summary": summary,
        "frames": frames,
        "centers": centers_np,
        "phases": phases_np,
        "actions": actions_np,
        "contacts": contacts,
    }


def plot_trace(result: dict, path: Path, title: str) -> None:
    centers = result["centers"]
    phases = result["phases"]
    ncon = np.asarray([c["ncon"] for c in result["contacts"]], dtype=np.float32)
    t = np.arange(len(centers))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, centers[:, 0], label="x")
    axes[0].plot(t, centers[:, 1], label="y")
    axes[0].plot(t, centers[:, 2], label="z")
    axes[0].set_ylabel("pen center (m)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(np.arange(len(phases)), phases, label="phase")
    axes[1].plot(np.arange(len(phases)), phases / (2.0 * np.pi), label="rounds")
    axes[1].set_ylabel("rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if len(ncon):
        axes[2].plot(np.arange(len(ncon)), ncon, label="ncon")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("contacts")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_3_scripted_baselines")
    parser.add_argument("--policies", nargs="+", default=["hold", "active_grasp", "finger_gait"])
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--action-scale", type=float, default=0.003)
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=DEFAULT_PEN_INIT_POS)
    parser.add_argument("--grasp-scale", type=float, default=1.4)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for policy in args.policies:
        print(f"running policy: {policy}")
        result = rollout(policy, args, save_frames=True)
        all_summaries[policy] = result["summary"]
        policy_dir = output_dir / policy
        policy_dir.mkdir(exist_ok=True)
        imageio.mimsave(policy_dir / "rollout.gif", result["frames"], duration=0.08)
        imageio.imwrite(policy_dir / "initial_frame.png", result["frames"][0])
        imageio.imwrite(policy_dir / "final_frame.png", result["frames"][-1])
        plot_trace(result, policy_dir / "trace.png", f"M2.3 {policy}")
        np.savez_compressed(
            policy_dir / "rollout.npz",
            centers=result["centers"],
            phases=result["phases"],
            actions=result["actions"],
        )
        with (policy_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(result["summary"], f, indent=2, ensure_ascii=False)
        print(result["summary"])

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "pen_init_pos": args.pen_init_pos,
                "grasp_scale": args.grasp_scale,
                "action_scale": args.action_scale,
                "episode_seconds": args.episode_seconds,
                "summaries": all_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved results to: {output_dir}")


if __name__ == "__main__":
    main()
