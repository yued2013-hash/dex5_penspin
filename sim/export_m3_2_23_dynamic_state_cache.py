from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_m3_2_17_cycle_closure as m317
import rebuild_m3_2_16_p4_p5_transitions as m316
from build_m3_2_19_active_b12_b23 import B12_EXTRAS
from build_m3_2_21_b12c_p2_entry import B12C_P2_ENTRY_EXTRAS
from build_m3_2_22_lifted_b12c_capture import B12C_LIFTED_EXTRAS
from optimize_m3_2_20_b12_two_stage import B12B_DYNAMIC_EXTRAS
from rebuild_m3_2_18_p1_p2_micro import build_micro_seeds
from evaluate_m3_2_15_short_primitives import contact_metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_route(seed_dir: Path) -> list[dict[str, Any]]:
    base = m316.load_chain(seed_dir)
    p1, p2, p3 = build_micro_seeds(base)
    b12a = m316.blend_seed(
        p1,
        p2,
        0.35,
        phase=1.35,
        name="bridge_p1_to_p2_b12a_middle_push",
        desired=["thumb", "index", "middle"],
        extras=B12_EXTRAS,
    )
    b12b = m316.blend_seed(
        p1,
        p2,
        0.55,
        phase=1.75,
        name="bridge_p1_to_p2_b12b_middle_ring_handoff",
        desired=["thumb", "middle", "ring"],
        extras=B12B_DYNAMIC_EXTRAS,
    )
    b12c = m316.blend_seed(
        b12b,
        p2,
        0.22,
        phase=2.08,
        name="bridge_b12b_to_p2_entry_lifted_support",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12C_LIFTED_EXTRAS,
    )
    b23 = m316.blend_seed(
        p2,
        p3,
        0.70,
        phase=2.5,
        name="bridge_p2_to_p3_index_ring_capture",
        desired=["thumb", "index", "ring"],
        extras={
            "index_pip": 0.70,
            "index_mid": 0.28,
            "middle_pip": 0.10,
            "middle_mid": 0.0,
            "ring_open": 0.04,
            "thumb_pitch": 0.08,
            "thumb_tip": 0.06,
        },
    )
    b12c_reference = m316.blend_seed(
        b12b,
        p2,
        0.18,
        phase=2.05,
        name="bridge_b12b_to_p2_entry_reference_m321",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12C_P2_ENTRY_EXTRAS,
    )
    b12c_reference["m3_2_23_cache_note"] = "Kept only for metadata comparison; route uses B12c lifted."
    return [p1, b12a, b12b, b12c, b23, p3]


def cache_record(env: Any, t: float, window: str) -> dict[str, Any]:
    state = env.get_state()
    metrics = contact_metrics(env)
    contacts = sorted(metrics["fingers"])
    return {
        "window": window,
        "t": float(t),
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "target_qpos": env.target_qpos.copy(),
        "hand_qpos": state.qpos.copy(),
        "hand_qvel": state.qvel.copy(),
        "pen_pos": env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3].copy(),
        "pen_quat": env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7].copy(),
        "pen_qvel": env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6].copy(),
        "pen_center": state.pen_center.copy(),
        "pen_axis": state.pen_axis.copy(),
        "rotation_phase": float(state.rotation_phase),
        "contacts": ",".join(contacts) or "none",
        "max_penetration_mm": float(metrics["max_penetration"] * 1000.0),
        "force_share": dict(metrics["force_share"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/dynamic_state_cache",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--segment-duration", type=float, default=0.18)
    parser.add_argument("--duration", type=float, default=0.90)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--action-delta-limit", type=float, default=0.026)
    parser.add_argument("--tracking-gain", type=float, default=0.62)
    parser.add_argument("--ramp-seconds", type=float, default=0.10)
    parser.add_argument("--initial-thumb-bias", type=float, default=0.035)
    parser.add_argument("--initial-ring-open", type=float, default=0.0)
    parser.add_argument("--initial-shelf-bias", type=float, default=0.018)
    parser.add_argument("--thumb-share-target", type=float, default=0.32)
    parser.add_argument("--ring-share-max", type=float, default=0.25)
    parser.add_argument("--ring-share-min", type=float, default=0.0)
    parser.add_argument("--axis-z-guard", type=float, default=0.17)
    parser.add_argument("--height-drop-guard", type=float, default=0.006)
    parser.add_argument("--thumb-bias-min", type=float, default=0.0)
    parser.add_argument("--thumb-bias-max", type=float, default=0.13)
    parser.add_argument("--ring-open-min", type=float, default=0.0)
    parser.add_argument("--ring-open-max", type=float, default=0.05)
    parser.add_argument("--ring-guard-max", type=float, default=0.10)
    parser.add_argument("--shelf-bias-min", type=float, default=0.0)
    parser.add_argument("--shelf-bias-max", type=float, default=0.05)
    parser.add_argument("--thumb-gain", type=float, default=0.065)
    parser.add_argument("--ring-open-gain", type=float, default=0.035)
    parser.add_argument("--ring-guard-gain", type=float, default=0.08)
    parser.add_argument("--ring-guard-decay", type=float, default=1.0)
    parser.add_argument("--shelf-gain", type=float, default=0.045)
    parser.add_argument("--height-gain", type=float, default=1.2)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_args = m317.make_controller_args(args)
    route = build_route(seed_dir)
    env = m316.make_env(route[0], controller_args, duration=args.duration, render=False)
    windows = {
        "earlier_pre_ring_for_B12b_to_B23": (0.16, 0.22),
        "earlier_pre_ring_016_019_for_B12b_to_B23": (0.16, 0.19),
        "earlier_pre_ring_019_022_for_B12b_to_B23": (0.19, 0.22),
        "early_pre_ring_for_B12b_to_B23": (0.22, 0.30),
        "pre_ring_collapse_for_B12b_to_B23": (0.30, 0.38),
        "b12c_lift_arrival_for_B12cLift_to_B23": (0.46, 0.56),
        "b23_arrival_for_B23_to_P3": (0.62, 0.74),
    }
    records_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    try:
        env.reset(seed=args.seed)
        m316.reset_to_seed(env, route[0])
        ctrl = m316.ControllerState(
            thumb_bias=args.initial_thumb_bias,
            ring_open=args.initial_ring_open,
            ring_guard=0.0,
            shelf_bias=args.initial_shelf_bias,
        )
        steps = int(round(args.duration * env.control_hz))
        steps_per_segment = int(round(args.segment_duration * env.control_hz))
        z0 = float(env.get_state().pen_center[2])
        dt = 1.0 / env.control_hz
        for step in range(steps):
            state = env.get_state()
            metrics = contact_metrics(env)
            m316.update_controller(ctrl, metrics, state, z0, controller_args, dt)
            ramp = min(1.0, (step + 1) / max(1.0, args.ramp_seconds * env.control_hz))
            base_q = m316.interpolate_route_q(route, step, steps_per_segment)
            target = m316.apply_controller(base_q, ctrl, env, ramp)
            action = np.clip(
                args.tracking_gain * (target - env.target_qpos.astype(np.float64)),
                -args.action_delta_limit,
                args.action_delta_limit,
            ).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            t = step / env.control_hz
            for window, (lo, hi) in windows.items():
                if lo <= t <= hi:
                    record = cache_record(env, t, window)
                    records_by_window[window].append(record)
                    rows.append(
                        {
                            "window": window,
                            "t": t,
                            "contacts": record["contacts"],
                            "pen_center_z": float(record["pen_center"][2]),
                            "max_penetration_mm": record["max_penetration_mm"],
                            "thumb_share": float(record["force_share"].get("thumb", 0.0)),
                            "index_share": float(record["force_share"].get("index", 0.0)),
                            "middle_share": float(record["force_share"].get("middle", 0.0)),
                            "ring_share": float(record["force_share"].get("ring", 0.0)),
                            "pinky_share": float(record["force_share"].get("pinky", 0.0)),
                        }
                    )
            if terminated or truncated:
                break
    finally:
        env.close()

    manifest = {
        "stage": "M3.2.23 dynamic reset state cache",
        "source_seed_dir": str(seed_dir),
        "route": [f"P{seed['phase']}:{seed['name']}" for seed in route],
        "windows": windows,
        "caches": {},
        "note": (
            "These states are dynamic rollout states. Use them for dynamic_state_cache primitives "
            "such as B12cLift_to_B23 and B23_to_P3 instead of resetting directly to a standalone JSON keypose."
        ),
    }
    for window, records in records_by_window.items():
        npz_path = output_dir / f"{window}.npz"
        np.savez_compressed(
            npz_path,
            t=np.asarray([r["t"] for r in records], dtype=np.float64),
            qpos=np.asarray([r["qpos"] for r in records], dtype=np.float64),
            qvel=np.asarray([r["qvel"] for r in records], dtype=np.float64),
            target_qpos=np.asarray([r["target_qpos"] for r in records], dtype=np.float64),
            hand_qpos=np.asarray([r["hand_qpos"] for r in records], dtype=np.float64),
            hand_qvel=np.asarray([r["hand_qvel"] for r in records], dtype=np.float64),
            pen_pos=np.asarray([r["pen_pos"] for r in records], dtype=np.float64),
            pen_quat=np.asarray([r["pen_quat"] for r in records], dtype=np.float64),
            pen_qvel=np.asarray([r["pen_qvel"] for r in records], dtype=np.float64),
            pen_center=np.asarray([r["pen_center"] for r in records], dtype=np.float64),
            pen_axis=np.asarray([r["pen_axis"] for r in records], dtype=np.float64),
            rotation_phase=np.asarray([r["rotation_phase"] for r in records], dtype=np.float64),
            contacts=np.asarray([r["contacts"] for r in records]),
            max_penetration_mm=np.asarray([r["max_penetration_mm"] for r in records], dtype=np.float64),
        )
        manifest["caches"][window] = {
            "path": str(npz_path),
            "num_states": len(records),
            "t_min": float(records[0]["t"]) if records else None,
            "t_max": float(records[-1]["t"]) if records else None,
        }
    write_csv(output_dir / "dynamic_state_cache_summary.csv", rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# M3.2.23 Dynamic State Cache",
        "",
        "These are rollout states from the selected lifted B12c scripted route. They are reset distributions for focused short-primitive RL, not successful target states.",
        "",
        "## Windows",
        "",
        "- `earlier_pre_ring_for_B12b_to_B23`: 0.16-0.22s curriculum window before the previous early-pre-ring cache; use it for index/thumb recapture transitions.",
        "- `earlier_pre_ring_016_019_for_B12b_to_B23`: first half of the 0.16-0.22s recapture curriculum.",
        "- `earlier_pre_ring_019_022_for_B12b_to_B23`: second half of the 0.16-0.22s recapture curriculum.",
        "- `early_pre_ring_for_B12b_to_B23`: earlier curriculum window before the old 0.30-0.38s cache. Use this first because the later cache is already ring/pinky dominated.",
        "- `pre_ring_collapse_for_B12b_to_B23`: before ring-only collapse; thumb/middle are still present, but pinky is also present. Use this to train pinky suppression and ring force limiting.",
        "- `b12c_lift_arrival_for_B12cLift_to_B23`: actual B12c arrival region; mostly ring-only. Use only for recovery curriculum after the pre-collapse policy improves.",
        "- `b23_arrival_for_B23_to_P3`: current B23/P3 arrival; middle/ring-only. Use to train final thumb/index/ring capture, but do not treat it as a solved B23 reset.",
        "",
        "Files:",
        "",
        "- `manifest.json`",
        "- `dynamic_state_cache_summary.csv`",
        "- `earlier_pre_ring_for_B12b_to_B23.npz`",
        "- `earlier_pre_ring_016_019_for_B12b_to_B23.npz`",
        "- `earlier_pre_ring_019_022_for_B12b_to_B23.npz`",
        "- `early_pre_ring_for_B12b_to_B23.npz`",
        "- `pre_ring_collapse_for_B12b_to_B23.npz`",
        "- `b12c_lift_arrival_for_B12cLift_to_B23.npz`",
        "- `b23_arrival_for_B23_to_P3.npz`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "caches": manifest["caches"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
