from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

import evaluate_m3_2_17_cycle_closure as m317
import rebuild_m3_2_16_p4_p5_transitions as m316
from build_m3_2_9_photo_five_phase_debug import save_montage
from build_m3_2_19_active_b12_b23 import B12_EXTRAS, B23_EXTRAS
from build_m3_2_21_b12c_p2_entry import B12C_P2_ENTRY_EXTRAS
from optimize_m3_2_20_b12_two_stage import B12B_DYNAMIC_EXTRAS
from rebuild_m3_2_18_p1_p2_micro import build_micro_seeds, save_json, static_row


B12C_LIFTED_EXTRAS = {
    "index_pip": 0.795,
    "index_mid": 0.282,
    "middle_pip": -0.077,
    "middle_mid": 0.117,
    "ring_open": 0.345,
    "thumb_pitch": 0.419,
    "thumb_tip": 0.219,
    "thumb_roll": 0.018,
}

B12B_INDEX_ASSIST_EXTRAS = {
    "index_pip": 1.20,
    "index_mid": 0.18,
    "middle_pip": 0.12,
    "middle_mid": -0.06,
    "ring_open": 0.08,
    "thumb_pitch": 0.14,
    "thumb_tip": 0.08,
    "thumb_roll": 0.05,
    "pos_x": 0.004,
    "pos_y": -0.004,
    "pos_z": 0.006,
}


def make_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return m317.make_controller_args(args)


def build_chain_seeds(
    p1: dict[str, Any],
    p2: dict[str, Any],
    p3: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    b12a = m316.blend_seed(
        p1,
        p2,
        0.35,
        phase=1.35,
        name="bridge_p1_to_p2_b12a_middle_push",
        desired=["thumb", "index", "middle"],
        extras=B12_EXTRAS,
    )
    b12a["m3_2_22_role"] = "Unchanged M3.2.20/M3.2.21 B12a: thumb lift plus middle push."

    b12b = m316.blend_seed(
        p1,
        p2,
        0.55,
        phase=1.75,
        name="bridge_p1_to_p2_b12b_middle_ring_handoff",
        desired=["thumb", "middle", "ring"],
        extras=B12B_DYNAMIC_EXTRAS,
    )
    b12b["m3_2_22_role"] = "Unchanged M3.2.20 B12b dynamic endpoint."

    b12c_reference = m316.blend_seed(
        b12b,
        p2,
        0.18,
        phase=2.05,
        name="bridge_b12b_to_p2_entry_reference_m321",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12C_P2_ENTRY_EXTRAS,
    )
    b12c_reference["m3_2_22_role"] = "M3.2.21 B12c reference."

    b12c_lifted = m316.blend_seed(
        b12b,
        p2,
        0.22,
        phase=2.08,
        name="bridge_b12b_to_p2_entry_lifted_support",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12C_LIFTED_EXTRAS,
    )
    b12c_lifted["m3_2_22_role"] = (
        "Selected M3.2.22 B12c: stronger thumb/index lift, reduced middle distal curl, "
        "and slightly more ring opening. It improves the B12c->B23 final support window "
        "without changing the preceding P1/B12a/B12b geometry."
    )

    b23 = m316.blend_seed(
        p2,
        p3,
        0.70,
        phase=2.5,
        name="bridge_p2_to_p3_index_ring_capture",
        desired=["thumb", "index", "ring"],
        extras=B23_EXTRAS,
    )
    b23["m3_2_22_role"] = "M3.2.19/M3.2.21 B23 reference, kept unchanged."

    b12b_index_assist = m316.blend_seed(
        p1,
        p2,
        0.62,
        phase=1.78,
        name="diagnostic_b12b_index_assist",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12B_INDEX_ASSIST_EXTRAS,
    )
    b12b_index_assist["m3_2_22_role"] = (
        "Diagnostic branch: keeps index active into the B12b endpoint. This lowers penetration "
        "and raises index contact ratio, but the downstream B23 support window is weaker."
    )

    b12c_from_index_assist = m316.blend_seed(
        b12b_index_assist,
        p2,
        0.22,
        phase=2.08,
        name="diagnostic_b12c_lifted_from_index_assist",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12C_LIFTED_EXTRAS,
    )
    b12c_from_index_assist["m3_2_22_role"] = (
        "Diagnostic B12c after the index-assist B12b. Kept for visual comparison; not the selected route."
    )

    return {
        "b12a": b12a,
        "b12b": b12b,
        "b12c_reference": b12c_reference,
        "b12c_lifted": b12c_lifted,
        "b23": b23,
        "b12b_index_assist": b12b_index_assist,
        "b12c_from_index_assist": b12c_from_index_assist,
    }


def final_finger_window_ratio(trace: dict[str, Any], finger: str, window_ratio: float = 0.25) -> float:
    rows = trace["rows"]
    if not rows:
        return 0.0
    start = int(round((1.0 - window_ratio) * len(rows)))
    window = rows[start:]
    hits = 0
    for row in window:
        contacts = set() if row["contacts"] == "none" else set(str(row["contacts"]).split(","))
        hits += int(finger in contacts)
    return hits / max(1, len(window))


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
) -> dict[str, Any]:
    row, trace = m317.summarize_route(
        env,
        route,
        controller_args,
        label=label,
        duration=duration,
        segment_s=segment_s,
        output_dir=output_dir,
        render_every=render_every,
    )
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        row[f"{finger}_final_window_ratio"] = final_finger_window_ratio(trace, finger)
    return row


def ratio(row: dict[str, Any], finger: str) -> float:
    return float(row.get("contact_ratio", {}).get(finger, 0.0))


def share(row: dict[str, Any], finger: str) -> float:
    return float(row.get("force_share_mean", {}).get(finger, 0.0))


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['label']} | {row['class']} | {row['survival_s']:.2f} | "
        f"{row['target_final_window_ratio']:.2f} | {row['support_final_window_ratio']:.2f} | "
        f"{row.get('thumb_final_window_ratio', 0.0):.2f} | {row.get('index_final_window_ratio', 0.0):.2f} | "
        f"{share(row, 'thumb'):.2f} | {share(row, 'ring'):.2f} | "
        f"{ratio(row, 'index'):.2f} | {ratio(row, 'middle'):.2f} | {ratio(row, 'ring'):.2f} | "
        f"{ratio(row, 'pinky'):.2f} | {row['max_penetration_mm']:.2f} | "
        f"{','.join(row['final_contacts']) or 'none'} |"
    )


def save_final_frame(gif_path: Path) -> str | None:
    if not gif_path.exists():
        return None
    frames = imageio.mimread(gif_path)
    if not frames:
        return None
    out_path = gif_path.with_name(f"{gif_path.stem}_final_frame.png")
    imageio.imwrite(out_path, frames[-1])
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/lifted_b12c_capture_m3_2_22",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hold-duration", type=float, default=0.24)
    parser.add_argument("--segment-duration", type=float, default=0.18)
    parser.add_argument("--render-every", type=int, default=2)
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
    controller_args = make_controller_args(args)

    base = m316.load_chain(seed_dir)
    p1_micro, p2_micro, p3_reference = build_micro_seeds(base)
    chain = build_chain_seeds(p1_micro, p2_micro, p3_reference)

    seeds = {
        "P1_four_finger_clamp_micro_thumb_supported_seed.json": p1_micro,
        "B12a_middle_push_seed.json": chain["b12a"],
        "B12b_middle_ring_handoff_seed.json": chain["b12b"],
        "B12c_reference_m3_2_21_seed.json": chain["b12c_reference"],
        "B12c_lifted_support_seed.json": chain["b12c_lifted"],
        "B23_reference_index_ring_capture_seed.json": chain["b23"],
        "B12b_index_assist_diagnostic_seed.json": chain["b12b_index_assist"],
        "B12c_lifted_from_index_assist_diagnostic_seed.json": chain["b12c_from_index_assist"],
        "P3_index_wrap_to_front_reference_seed.json": p3_reference,
    }
    seed_paths: dict[str, str] = {}
    for filename, seed in seeds.items():
        path = output_dir / filename
        save_json(path, seed)
        seed_paths[filename] = str(path)

    render_env = m316.make_env(p1_micro, controller_args, duration=args.segment_duration * 6.0 + 0.6, render=True)
    try:
        keyposes = [
            ("P1 micro", p1_micro),
            ("B12a", chain["b12a"]),
            ("B12b", chain["b12b"]),
            ("B12c M321", chain["b12c_reference"]),
            ("B12c lifted", chain["b12c_lifted"]),
            ("B23 reference", chain["b23"]),
            ("B12b index assist", chain["b12b_index_assist"]),
            ("B12c index assist", chain["b12c_from_index_assist"]),
            ("P3 reference", p3_reference),
        ]
        frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_')}.png", frame)
        save_montage(frames, output_dir / "m3_2_22_lifted_b12c_keypose_montage.png", cols=3)
        imageio.mimsave(output_dir / "m3_2_22_lifted_b12c_keyposes.gif", frames, duration=0.45)

        static_rows = [
            static_row(render_env, p1_micro, label="P1_micro_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, chain["b12a"], label="B12a_static", desired={"thumb", "index", "middle"}, allowed_extra={"pinky", "ring"}),
            static_row(render_env, chain["b12b"], label="B12b_static", desired={"thumb", "middle", "ring"}, allowed_extra={"pinky", "index"}),
            static_row(render_env, chain["b12c_reference"], label="B12c_reference_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, chain["b12c_lifted"], label="B12c_lifted_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, chain["b23"], label="B23_reference_static", desired={"thumb", "index", "ring"}, allowed_extra={"middle"}),
            static_row(render_env, chain["b12b_index_assist"], label="B12b_index_assist_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
        ]
        m317.write_csv(output_dir / "static_summary.csv", static_rows)

        hold_rows = [
            summarize_route(
                render_env,
                [chain["b12c_lifted"]],
                controller_args,
                label="hold_B12c_lifted_reset_diagnostic",
                duration=args.hold_duration,
                segment_s=None,
                output_dir=output_dir,
                render_every=args.render_every,
            ),
            summarize_route(
                render_env,
                [chain["b12b_index_assist"]],
                controller_args,
                label="hold_B12b_index_assist_reset_diagnostic",
                duration=args.hold_duration,
                segment_s=None,
                output_dir=output_dir,
                render_every=args.render_every,
            ),
        ]
        m317.write_csv(output_dir / "hold_summary.csv", hold_rows)

        transition_specs = [
            (
                "reference_m321_P1_B12a_B12b_B12c_B23",
                [p1_micro, chain["b12a"], chain["b12b"], chain["b12c_reference"], chain["b23"]],
                args.segment_duration * 4.0,
                args.segment_duration,
            ),
            (
                "selected_lifted_P1_B12a_B12b_B12cLift_B23",
                [p1_micro, chain["b12a"], chain["b12b"], chain["b12c_lifted"], chain["b23"]],
                args.segment_duration * 4.0,
                args.segment_duration,
            ),
            (
                "selected_lifted_P1_B12a_B12b_B12cLift_B23_P3",
                [p1_micro, chain["b12a"], chain["b12b"], chain["b12c_lifted"], chain["b23"], p3_reference],
                args.segment_duration * 5.0,
                args.segment_duration,
            ),
            (
                "diagnostic_indexassist_P1_B12a_B12bIdx_B12cLift_B23",
                [
                    p1_micro,
                    chain["b12a"],
                    chain["b12b_index_assist"],
                    chain["b12c_from_index_assist"],
                    chain["b23"],
                ],
                args.segment_duration * 4.0,
                args.segment_duration,
            ),
        ]
        transition_rows = [
            summarize_route(
                render_env,
                route,
                controller_args,
                label=label,
                duration=duration,
                segment_s=segment_s,
                output_dir=output_dir,
                render_every=args.render_every,
            )
            for label, route, duration, segment_s in transition_specs
        ]
        m317.write_csv(output_dir / "transition_summary.csv", transition_rows)
        final_frame_paths = {
            label: save_final_frame(output_dir / f"{label}.gif")
            for label, _, _, _ in transition_specs
        }
    finally:
        render_env.close()

    manifest = {
        "stage": "M3.2.22 lifted B12c capture optimization",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "selected_seed_files": seed_paths,
        "b12c_lifted_extras": B12C_LIFTED_EXTRAS,
        "b12b_index_assist_extras": B12B_INDEX_ASSIST_EXTRAS,
        "static_rows": static_rows,
        "hold_rows": hold_rows,
        "transition_rows": transition_rows,
        "final_frame_paths": final_frame_paths,
        "interpretation": (
            "The selected lifted B12c improves the B12c->B23 final support window versus M3.2.21 "
            "while preserving full 0.72s survival. It does not yet solve strict thumb/index/ring B23 "
            "target capture; final contact still tends to middle/ring. The index-assist branch raises "
            "index participation and reduces penetration, but weakens downstream support."
        ),
    }
    save_json(output_dir / "manifest.json", manifest)

    readme = [
        "# M3.2.22 Lifted B12c Capture",
        "",
        "This package keeps P1, B12a, B12b, and B23 mostly fixed, then retunes B12c as a dynamic P2/B23 entry state. The selected route is `selected_lifted_P1_B12a_B12b_B12cLift_B23`.",
        "",
        "## Hold",
        "",
        "| label | class | survival | final target | final support | thumb final | index final | thumb share | ring share | index ratio | middle ratio | ring ratio | pinky ratio | max penetration mm | final contacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in hold_rows:
        readme.append(row_line(row))
    readme += [
        "",
        "## Transitions",
        "",
        "| label | class | survival | final target | final support | thumb final | index final | thumb share | ring share | index ratio | middle ratio | ring ratio | pinky ratio | max penetration mm | final contacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in transition_rows:
        readme.append(row_line(row))
    readme += [
        "",
        "Key visuals:",
        "",
        "- `m3_2_22_lifted_b12c_keypose_montage.png`",
        "- `selected_lifted_P1_B12a_B12b_B12cLift_B23.gif`",
        "- `selected_lifted_P1_B12a_B12b_B12cLift_B23_final_frame.png`",
        "- `selected_lifted_P1_B12a_B12b_B12cLift_B23_P3.gif`",
        "- `diagnostic_indexassist_P1_B12a_B12bIdx_B12cLift_B23.gif`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    assessment = [
        "# M3.2.22 lifted B12c 优化评估",
        "",
        "本轮没有大改 P1/B12a/B12b/B23，而是把优化集中在 B12c 末端：增强 thumb/index 上托，略开 ring，减小 middle 远端卷曲，让 B12c 更像 B23 前的动态入口。",
        "",
        "## 结论",
        "",
        "`selected_lifted_P1_B12a_B12b_B12cLift_B23` 可以跑满 0.72s，并把最后 25% 窗口的 support ratio 从 M3.2.21 reference 的约 0.50 提到约 0.67。也就是说，它更像一个可连接的短 primitive。",
        "",
        "但严格的 B23 target capture 还没有解决：final target window 仍为 0，末端仍主要落在 `middle/ring`，thumb/index 没能在最终窗口稳定回接。",
        "",
        "`diagnostic_indexassist_P1_B12a_B12bIdx_B12cLift_B23` 说明另一条方向：让 B12b 保 index 可以降低最大穿透并提高全程 index 接触比例，但会削弱 B23 末端支撑，不作为主线。",
        "",
        "## 下一步",
        "",
        "如果继续优化，不建议只堆更强的 B23 目标姿态；需要加一个真正的 B23 捕获控制逻辑，按相位显式调 index/thumb，而不是只靠最后一个 keypose 插值。",
        "",
        "## 关键文件",
        "",
        "- `B12c_lifted_support_seed.json`",
        "- `B12b_index_assist_diagnostic_seed.json`",
        "- `m3_2_22_lifted_b12c_keypose_montage.png`",
        "- `selected_lifted_P1_B12a_B12b_B12cLift_B23.gif`",
        "- `selected_lifted_P1_B12a_B12b_B12cLift_B23_final_frame.png`",
        "- `selected_lifted_P1_B12a_B12b_B12cLift_B23_P3.gif`",
        "- `transition_summary.csv`",
    ]
    (output_dir / "ASSESSMENT.zh.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
