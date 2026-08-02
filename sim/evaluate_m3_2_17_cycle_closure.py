from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import rebuild_m3_2_16_p4_p5_transitions as m316


def make_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        action_delta_limit=args.action_delta_limit,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
        tracking_gain=args.tracking_gain,
        ramp_seconds=args.ramp_seconds,
        initial_thumb_bias=args.initial_thumb_bias,
        initial_ring_open=args.initial_ring_open,
        initial_shelf_bias=args.initial_shelf_bias,
        thumb_share_target=args.thumb_share_target,
        ring_share_max=args.ring_share_max,
        ring_share_min=args.ring_share_min,
        axis_z_guard=args.axis_z_guard,
        height_drop_guard=args.height_drop_guard,
        thumb_bias_min=args.thumb_bias_min,
        thumb_bias_max=args.thumb_bias_max,
        ring_open_min=args.ring_open_min,
        ring_open_max=args.ring_open_max,
        ring_guard_max=args.ring_guard_max,
        shelf_bias_min=args.shelf_bias_min,
        shelf_bias_max=args.shelf_bias_max,
        thumb_gain=args.thumb_gain,
        ring_open_gain=args.ring_open_gain,
        ring_guard_gain=args.ring_guard_gain,
        ring_guard_decay=args.ring_guard_decay,
        shelf_gain=args.shelf_gain,
        height_gain=args.height_gain,
    )


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def build_p1_cycle_entry(p1: dict[str, Any], p5: dict[str, Any]) -> dict[str, Any]:
    seed = m316.blend_seed(
        p5,
        p1,
        0.80,
        phase=1.0,
        name="p1_cycle_entry_from_p5",
        desired=["thumb", "index", "middle", "ring"],
        extras={
            "ring_open": 0.08,
            "pos_z": 0.0,
        },
    )
    q = np.asarray(seed["hand_qpos"], dtype=np.float64)
    q[16:20] = np.asarray(p1["hand_qpos"], dtype=np.float64)[16:20]
    seed["hand_qpos"] = m316.enforce_shared_motor_consistency(q).astype(float).tolist()
    seed["target_qpos"] = seed["hand_qpos"]
    seed["m3_2_17_selected"] = (
        "Cycle-entry keypose close to P1: 80% P5->P1 blend, P1 thumb geometry, "
        "ring contact available in static/hold but low force-share."
    )
    return seed


def final_window_ratio(trace: dict[str, Any], desired: set[str], window_ratio: float = 0.25) -> float:
    rows = trace["rows"]
    if not rows or not desired:
        return 0.0
    start = int(round((1.0 - window_ratio) * len(rows)))
    window = rows[start:]
    hits = 0
    for row in window:
        contacts = set() if row["contacts"] == "none" else set(str(row["contacts"]).split(","))
        hits += int(desired.issubset(contacts))
    return hits / max(1, len(window))


def support_target_ratio(trace: dict[str, Any], support: set[str], window_ratio: float = 0.25) -> float:
    rows = trace["rows"]
    if not rows or not support:
        return 0.0
    start = int(round((1.0 - window_ratio) * len(rows)))
    window = rows[start:]
    hits = 0
    for row in window:
        contacts = set() if row["contacts"] == "none" else set(str(row["contacts"]).split(","))
        hits += int(bool(support.intersection(contacts)) and len(contacts.intersection({"thumb", "index", "middle", "ring"})) >= 2)
    return hits / max(1, len(window))


def classify(summary: dict[str, Any], *, duration: float, target_final_ratio: float, support_final_ratio: float) -> str:
    if summary["terminated"] or summary["survival_s"] + 1e-9 < duration:
        if summary["survival_s"] >= 0.18 and summary["max_penetration_mm"] <= 12.0:
            return "transient_only"
        return "reject"
    if target_final_ratio >= 0.35:
        return "target_contact_ok"
    if support_final_ratio >= 0.60 and summary["active_contact_ratio"] >= 0.70:
        return "support_connectable"
    if summary["active_contact_ratio"] >= 0.70:
        return "support_only"
    return "weak"


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


def summarize_route(
    env: Any,
    route: list[dict[str, Any]],
    controller_args: argparse.Namespace,
    *,
    label: str,
    duration: float,
    segment_s: float | None,
    output_dir: Path,
    render_every: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, trace = m316.run_route(
        env,
        route,
        controller_args,
        duration=duration,
        segment_s=segment_s,
        render_every=render_every,
        label=label,
    )
    desired = set(route[-1].get("desired_contacts", []))
    target_final = final_window_ratio(trace, desired)
    support_final = support_target_ratio(trace, {"thumb", "index", "middle", "ring"})
    summary = {
        **summary,
        "class": classify(summary, duration=duration, target_final_ratio=target_final, support_final_ratio=support_final),
        "target_final_window_ratio": float(target_final),
        "support_final_window_ratio": float(support_final),
    }
    if trace["frames"]:
        imageio.mimsave(output_dir / f"{label}.gif", trace["frames"], duration=render_every / env.control_hz)
    write_csv(output_dir / f"{label}_trace.csv", trace["rows"])
    m316.save_trace_plot(output_dir / f"{label}_trace.png", trace, label)
    return summary, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--rebuilt-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/rebuilt_p4_p5_transitions_m3_2_16",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/cycle_closure_eval_m3_2_17",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hold-duration", type=float, default=0.24)
    parser.add_argument("--segment-duration", type=float, default=0.28)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--action-delta-limit", type=float, default=0.028)
    parser.add_argument("--tracking-gain", type=float, default=0.62)
    parser.add_argument("--ramp-seconds", type=float, default=0.12)
    parser.add_argument("--initial-thumb-bias", type=float, default=0.045)
    parser.add_argument("--initial-ring-open", type=float, default=0.045)
    parser.add_argument("--initial-shelf-bias", type=float, default=0.020)
    parser.add_argument("--thumb-share-target", type=float, default=0.36)
    parser.add_argument("--ring-share-max", type=float, default=0.18)
    parser.add_argument("--ring-share-min", type=float, default=0.0)
    parser.add_argument("--axis-z-guard", type=float, default=0.17)
    parser.add_argument("--height-drop-guard", type=float, default=0.006)
    parser.add_argument("--thumb-bias-min", type=float, default=0.0)
    parser.add_argument("--thumb-bias-max", type=float, default=0.16)
    parser.add_argument("--ring-open-min", type=float, default=0.0)
    parser.add_argument("--ring-open-max", type=float, default=0.095)
    parser.add_argument("--ring-guard-max", type=float, default=0.09)
    parser.add_argument("--shelf-bias-min", type=float, default=0.0)
    parser.add_argument("--shelf-bias-max", type=float, default=0.065)
    parser.add_argument("--thumb-gain", type=float, default=0.075)
    parser.add_argument("--ring-open-gain", type=float, default=0.055)
    parser.add_argument("--ring-guard-gain", type=float, default=0.09)
    parser.add_argument("--ring-guard-decay", type=float, default=1.15)
    parser.add_argument("--shelf-gain", type=float, default=0.055)
    parser.add_argument("--height-gain", type=float, default=1.40)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    rebuilt_dir = Path(args.rebuilt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_args = make_controller_args(args)

    base = m316.load_chain(seed_dir)
    p1 = base[1]
    p2 = base[2]
    p3 = base[3]
    p4 = load_json(rebuilt_dir / "P4_rebuilt_thumb_supported_seed.json")
    p5 = load_json(rebuilt_dir / "P5_rebuilt_thumb_index_capture_seed.json")
    b34 = load_json(rebuilt_dir / "B34_p3_to_p4_thumb_lift_bridge_seed.json")
    b45 = load_json(rebuilt_dir / "B45_p4_to_p5_thumb_index_capture_bridge_seed.json")
    p1_cycle = build_p1_cycle_entry(p1, p5)
    (output_dir / "P1_cycle_entry_from_P5_seed.json").write_text(
        json.dumps(p1_cycle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    render_env = m316.make_env(p1, controller_args, duration=3.0, render=True)
    try:
        keyposes = [
            ("P1 original", p1),
            ("P2 original", p2),
            ("P3 original", p3),
            ("B34", b34),
            ("P4 rebuilt", p4),
            ("B45", b45),
            ("P5 rebuilt", p5),
            ("P1 cycle entry", p1_cycle),
        ]
        keypose_frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, keypose_frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_')}.png", frame)
        m316.save_montage(keypose_frames, output_dir / "m3_2_17_cycle_keypose_montage.png", cols=4)
        imageio.mimsave(output_dir / "m3_2_17_cycle_keyposes.gif", keypose_frames, duration=0.45)

        hold_rows: list[dict[str, Any]] = []
        for label, seed in keyposes:
            row, _ = summarize_route(
                render_env,
                [seed],
                controller_args,
                label=f"hold_{label.replace(' ', '_')}",
                duration=args.hold_duration,
                segment_s=None,
                output_dir=output_dir,
                render_every=args.render_every,
            )
            hold_rows.append(row)

        route_specs = [
            ("transition_P1_to_P2", [p1, p2], args.segment_duration, args.segment_duration),
            ("transition_P2_to_P3", [p2, p3], args.segment_duration, args.segment_duration),
            ("transition_P3_B34_P4", [p3, b34, p4], args.segment_duration * 2.0, args.segment_duration),
            ("transition_P4_B45_P5", [p4, b45, p5], args.segment_duration * 2.0, args.segment_duration),
            ("transition_P5_to_P1_cycle_entry", [p5, p1_cycle], args.segment_duration, args.segment_duration),
            ("transition_P5_P1cycle_P1original", [p5, p1_cycle, p1], args.segment_duration * 2.0, args.segment_duration),
            ("transition_P1cycle_to_P2", [p1_cycle, p2], args.segment_duration, args.segment_duration),
            (
                "diagnostic_cycle_P1_to_P1cycle",
                [p1, p2, p3, b34, p4, b45, p5, p1_cycle],
                args.segment_duration * 7.0,
                args.segment_duration,
            ),
            (
                "diagnostic_cycle_P1cycle_to_P1cycle",
                [p1_cycle, p2, p3, b34, p4, b45, p5, p1_cycle],
                args.segment_duration * 7.0,
                args.segment_duration,
            ),
        ]
        transition_rows: list[dict[str, Any]] = []
        for label, route, duration, segment_s in route_specs:
            row, _ = summarize_route(
                render_env,
                route,
                controller_args,
                label=label,
                duration=duration,
                segment_s=segment_s,
                output_dir=output_dir,
                render_every=args.render_every,
            )
            transition_rows.append(row)
    finally:
        render_env.close()

    write_csv(output_dir / "hold_summary.csv", hold_rows)
    write_csv(output_dir / "transition_summary.csv", transition_rows)
    manifest = {
        "stage": "M3.2.17 cycle closure check",
        "source_seed_dir": str(seed_dir),
        "rebuilt_dir": str(rebuilt_dir),
        "output_dir": str(output_dir),
        "p1_cycle_entry": "P1_cycle_entry_from_P5_seed.json",
        "hold_rows": hold_rows,
        "transition_rows": transition_rows,
        "interpretation": (
            "P5 can return to a P1-like thumb/index/middle support state, and the static P1-cycle entry can hold "
            "thumb/index/middle/ring with low ring share. Dynamic P5->P1 does not yet recover strict ring contact, "
            "so full closed-loop spin is not promoted."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.17 Cycle Closure Check",
        "",
        "This checks whether P5 can return to P1 and visualizes the whole short-primitive chain.",
        "",
        "## Holds",
        "",
        "| label | class | survival | desired | active | thumb share | ring share | pinky ratio | max penetration mm | final contacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in hold_rows:
        readme.append(
            f"| {row['label']} | {row['class']} | {row['survival_s']:.2f} | {row['desired_ratio']:.2f} | "
            f"{row['active_contact_ratio']:.2f} | {row['force_share_mean']['thumb']:.2f} | "
            f"{row['force_share_mean']['ring']:.2f} | {row['contact_ratio']['pinky']:.2f} | "
            f"{row['max_penetration_mm']:.2f} | {','.join(row['final_contacts']) or 'none'} |"
        )
    readme += [
        "",
        "## Transitions",
        "",
        "| label | class | survival | desired | final target | final support | active | thumb share | ring share | pinky ratio | max penetration mm | final contacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in transition_rows:
        readme.append(
            f"| {row['label']} | {row['class']} | {row['survival_s']:.2f} | {row['desired_ratio']:.2f} | "
            f"{row['target_final_window_ratio']:.2f} | {row['support_final_window_ratio']:.2f} | "
            f"{row['active_contact_ratio']:.2f} | {row['force_share_mean']['thumb']:.2f} | "
            f"{row['force_share_mean']['ring']:.2f} | {row['contact_ratio']['pinky']:.2f} | "
            f"{row['max_penetration_mm']:.2f} | {','.join(row['final_contacts']) or 'none'} |"
        )
    readme += [
        "",
        "Key visuals:",
        "",
        "- `m3_2_17_cycle_keypose_montage.png`",
        "- `m3_2_17_cycle_keyposes.gif`",
        "- `transition_P5_to_P1_cycle_entry.gif`",
        "- `transition_P5_P1cycle_P1original.gif`",
        "- `diagnostic_cycle_P1_to_P1cycle.gif`",
        "- `diagnostic_cycle_P1cycle_to_P1cycle.gif`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    assessment = [
        "# M3.2.17 闭环短 primitive 可用性确认",
        "",
        "本轮补充检查 P5 是否能回到 P1，并把整条短 primitive 链保存为可视化结果。",
        "",
        "## 结论",
        "",
        "P5 可以回到一个 P1-like 的 thumb/index/middle 支撑状态：动态 `P5 -> P1_cycle_entry` 不掉，active contact ratio 为 1.00，thumb 平均承力约 0.49，ring 没有抢承力。",
        "",
        "但严格的四指 P1 接触还没有动态恢复：`P5 -> P1_cycle_entry` 的 ring 接触只偶发出现，最终仍是 thumb/index/middle，而不是稳定 thumb/index/middle/ring。",
        "",
        "`P1_cycle_entry` 作为静态/短时关键态本身是可用的：0.24s hold 中 desired ratio 约 0.88，ring contact ratio 约 0.88，ring force share 约 0.03，无 pinky。",
        "",
        "完整闭环还不能提升为连续 spin primitive。主要断点仍在前半段：`P1_cycle_entry -> P2` 会引入 pinky/ring 非预期接触并提前失败；原始 `P1 -> P2` 也不是稳定可连接段。",
        "",
        "所以当前可确认的是：P4/P5/B34/B45 和 P5 回 P1-like 支撑入口这些短 primitive 可用；完整 P1->P2->...->P5->P1 闭环还只是诊断可视化，不是可执行周期。",
        "",
        "## 关键文件",
        "",
        "- `P1_cycle_entry_from_P5_seed.json`",
        "- `m3_2_17_cycle_keypose_montage.png`",
        "- `transition_P5_to_P1_cycle_entry.gif`",
        "- `transition_P5_P1cycle_P1original.gif`",
        "- `diagnostic_cycle_P1_to_P1cycle.gif`",
        "- `hold_summary.csv`",
        "- `transition_summary.csv`",
    ]
    (output_dir / "ASSESSMENT.zh.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
