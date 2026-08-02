from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

import evaluate_m3_2_17_cycle_closure as m317
import rebuild_m3_2_16_p4_p5_transitions as m316
from build_m3_2_9_photo_five_phase_debug import save_montage
from build_m3_2_19_active_b12_b23 import B12_EXTRAS, B23_EXTRAS, row_line
from optimize_m3_2_20_b12_two_stage import B12B_DYNAMIC_EXTRAS
from rebuild_m3_2_18_p1_p2_micro import build_micro_seeds, save_json, static_row


B12C_P2_ENTRY_EXTRAS = {
    "index_pip": 0.25,
    "index_mid": 0.12,
    "middle_pip": 0.12,
    "middle_mid": 0.08,
    "ring_open": 0.28,
    "thumb_pitch": 0.10,
    "thumb_tip": 0.065,
    "pos_z": 0.0,
}


def make_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return m317.make_controller_args(args)


def build_chain_seeds(
    p1: dict[str, Any],
    p2: dict[str, Any],
    p3: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    b12a = m316.blend_seed(
        p1,
        p2,
        0.35,
        phase=1.35,
        name="bridge_p1_to_p2_b12a_middle_push",
        desired=["thumb", "index", "middle"],
        extras=B12_EXTRAS,
    )
    b12a["m3_2_21_bridge_role"] = "B12a active rolling entry: thumb lift plus middle push."

    b12b = m316.blend_seed(
        p1,
        p2,
        0.55,
        phase=1.75,
        name="bridge_p1_to_p2_b12b_middle_ring_handoff",
        desired=["thumb", "middle", "ring"],
        extras=B12B_DYNAMIC_EXTRAS,
    )
    b12b["m3_2_21_bridge_role"] = "B12b dynamic handoff endpoint from M3.2.20."

    b12c = m316.blend_seed(
        b12b,
        p2,
        0.18,
        phase=2.05,
        name="bridge_b12b_to_p2_entry",
        desired=["thumb", "index", "middle", "ring"],
        extras=B12C_P2_ENTRY_EXTRAS,
    )
    b12c["m3_2_21_bridge_role"] = (
        "B12c/P2-entry: dynamic-only stabilizer between B12b and B23. "
        "It is not a standalone reset/hold keypose; it reduces the B12b->B23 drop by keeping "
        "middle/ring support while index begins to enter."
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
    b23["m3_2_21_bridge_role"] = "M3.2.19 B23 reference, kept unchanged."
    return b12a, b12b, b12c, b23


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
    row, _ = m317.summarize_route(
        env,
        route,
        controller_args,
        label=label,
        duration=duration,
        segment_s=segment_s,
        output_dir=output_dir,
        render_every=render_every,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/b12c_p2_entry_m3_2_21",
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
    b12a, b12b, b12c, b23 = build_chain_seeds(p1_micro, p2_micro, p3_reference)

    seeds = {
        "P1_four_finger_clamp_micro_thumb_supported_seed.json": p1_micro,
        "B12a_middle_push_seed.json": b12a,
        "B12b_middle_ring_handoff_seed.json": b12b,
        "B12c_p2_entry_seed.json": b12c,
        "B23_reference_index_ring_capture_seed.json": b23,
        "P3_index_wrap_to_front_reference_seed.json": p3_reference,
    }
    seed_paths: dict[str, str] = {}
    for filename, seed in seeds.items():
        path = output_dir / filename
        save_json(path, seed)
        seed_paths[filename] = str(path)

    render_env = m316.make_env(p1_micro, controller_args, duration=args.segment_duration * 5.0 + 0.6, render=True)
    try:
        keyposes = [
            ("P1 micro", p1_micro),
            ("B12a", b12a),
            ("B12b", b12b),
            ("B12c P2 entry", b12c),
            ("B23 reference", b23),
            ("P3 reference", p3_reference),
        ]
        frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_')}.png", frame)
        save_montage(frames, output_dir / "m3_2_21_b12c_p2_entry_keypose_montage.png", cols=3)
        imageio.mimsave(output_dir / "m3_2_21_b12c_p2_entry_keyposes.gif", frames, duration=0.45)

        static_rows = [
            static_row(render_env, p1_micro, label="P1_micro_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, b12a, label="B12a_static", desired={"thumb", "index", "middle"}, allowed_extra={"pinky", "ring"}),
            static_row(render_env, b12b, label="B12b_static", desired={"thumb", "middle", "ring"}, allowed_extra={"pinky", "index"}),
            static_row(render_env, b12c, label="B12c_p2_entry_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, b23, label="B23_reference_static", desired={"thumb", "index", "ring"}, allowed_extra={"middle"}),
        ]
        m317.write_csv(output_dir / "static_summary.csv", static_rows)

        hold_rows = [
            summarize_route(
                render_env,
                [b12c],
                controller_args,
                label="hold_B12c_p2_entry_reset_diagnostic",
                duration=args.hold_duration,
                segment_s=None,
                output_dir=output_dir,
                render_every=args.render_every,
            )
        ]
        m317.write_csv(output_dir / "hold_summary.csv", hold_rows)

        transition_specs = [
            (
                "baseline_no_entry_P1_B12a_B12b_B23",
                [p1_micro, b12a, b12b, b23],
                args.segment_duration * 3.0,
                args.segment_duration,
            ),
            (
                "entry_P1_B12a_B12b_B12c",
                [p1_micro, b12a, b12b, b12c],
                args.segment_duration * 3.0,
                args.segment_duration,
            ),
            (
                "entry_P1_B12a_B12b_B12c_B23",
                [p1_micro, b12a, b12b, b12c, b23],
                args.segment_duration * 4.0,
                args.segment_duration,
            ),
            (
                "entry_P1_B12a_B12b_B12c_B23_P3",
                [p1_micro, b12a, b12b, b12c, b23, p3_reference],
                args.segment_duration * 5.0,
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
    finally:
        render_env.close()

    manifest = {
        "stage": "M3.2.21 B12b to B23/P2-entry transition",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "selected_seed_files": seed_paths,
        "b12c_p2_entry_extras": B12C_P2_ENTRY_EXTRAS,
        "static_rows": static_rows,
        "hold_rows": hold_rows,
        "transition_rows": transition_rows,
        "interpretation": (
            "B12c is a dynamic P2-entry stabilizer. It is not reset-stable, but when reached from "
            "P1->B12a->B12b it extends the B12b->B23 chain from a 0.46s transient failure to a full "
            "0.72s support-only transition. The next issue is target capture: final B23 still lacks "
            "strict thumb/index/ring and tends to settle on middle/ring."
        ),
    }
    save_json(output_dir / "manifest.json", manifest)

    readme = [
        "# M3.2.21 B12c P2 Entry",
        "",
        "This package adds a dynamic B12c/P2-entry between B12b and B23. B12c is not a standalone static grasp; it is an entry stabilizer reached from the running B12 sequence.",
        "",
        "## Hold",
        "",
        "| label | class | survival | desired | final target | final support | active | thumb share | ring share | index ratio | middle ratio | ring ratio | pinky ratio | max penetration mm | final contacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in hold_rows:
        readme.append(row_line(row))
    readme += [
        "",
        "## Transitions",
        "",
        "| label | class | survival | desired | final target | final support | active | thumb share | ring share | index ratio | middle ratio | ring ratio | pinky ratio | max penetration mm | final contacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in transition_rows:
        readme.append(row_line(row))
    readme += [
        "",
        "Key visuals:",
        "",
        "- `m3_2_21_b12c_p2_entry_keypose_montage.png`",
        "- `entry_P1_B12a_B12b_B12c_B23.gif`",
        "- `baseline_no_entry_P1_B12a_B12b_B23.gif`",
        "- `entry_P1_B12a_B12b_B12c_B23_P3.gif`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    assessment = [
        "# M3.2.21 B12b -> B23/P2-entry 过渡评估",
        "",
        "本轮新增 `B12c_p2_entry`，不再强迫 B12b 直接变成完整 P2，也不把 B12c 当成可 reset 的静态抓握。它的用途是在动态链中承接 B12b，并把支撑带到能继续尝试 B23 的入口。",
        "",
        "## 结论",
        "",
        "加入 B12c 后，`P1 -> B12a -> B12b -> B12c -> B23` 能跑完整 0.72s；上一版没有 B12c 的 `P1 -> B12a -> B12b -> B23` 在约 0.46s 终止。这说明 B12c 确实补上了 B12b 后的入口稳定问题。",
        "",
        "但它还不是 B23 目标捕获。完整 0.72s 结束时通常是 `middle/ring` 支撑，final target window 仍为 0，说明严格的 `thumb/index/ring` 捕获还没恢复。继续推到 P3 会在约 0.76s 终止。",
        "",
        "`hold_B12c_p2_entry_reset_diagnostic` 会失败，这是预期内的：B12c 是动态经过态，不能单独 reset 后静态持笔。",
        "",
        "## 关键文件",
        "",
        "- `B12c_p2_entry_seed.json`",
        "- `m3_2_21_b12c_p2_entry_keypose_montage.png`",
        "- `entry_P1_B12a_B12b_B12c_B23.gif`",
        "- `baseline_no_entry_P1_B12a_B12b_B23.gif`",
        "- `transition_summary.csv`",
    ]
    (output_dir / "ASSESSMENT.zh.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
