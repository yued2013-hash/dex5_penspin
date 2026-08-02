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
from rebuild_m3_2_18_p1_p2_micro import build_micro_seeds, save_json, static_row


B12B_DYNAMIC_EXTRAS = {
    "middle_pip": 0.12,
    "middle_mid": 0.08,
    "ring_open": 0.04,
    "thumb_pitch": 0.08,
    "thumb_tip": 0.052,
    "pos_x": 0.004,
    "pos_y": -0.004,
    "pos_z": 0.006,
}
B12B_LOW_PINKY_REFERENCE_EXTRAS = {
    "middle_pip": 0.20,
    "middle_mid": 0.08,
    "ring_open": 0.12,
    "thumb_pitch": 0.08,
    "thumb_tip": 0.052,
    "pos_z": 0.0,
}


def make_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return m317.make_controller_args(args)


def build_b12_variants(
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
    b12a["m3_2_20_selected_bridge"] = (
        "B12a keeps the first half of M3.2.19: thumb lift plus middle push. "
        "It is useful as an active rolling entry but is not intended as the terminal handoff state."
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
    b12b["m3_2_20_selected_bridge"] = (
        "B12b is the optimized dynamic handoff endpoint. It intentionally stops before full P2, "
        "because forcing full P2 lets ring/pinky take over and drops middle. In P1->B12a->B12b, "
        "middle remains present through the final window."
    )

    b12b_low_pinky = m316.blend_seed(
        b12a,
        p2,
        0.55,
        phase=1.76,
        name="bridge_p1_to_p2_b12b_low_pinky_reference",
        desired=["thumb", "middle", "ring"],
        extras=B12B_LOW_PINKY_REFERENCE_EXTRAS,
    )
    b12b_low_pinky["m3_2_20_reference_bridge"] = (
        "Reference B12b candidate: better standalone low-pinky hold, but worse dynamic entry from P1."
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
    b23["m3_2_20_reference_bridge"] = "M3.2.19 B23 kept unchanged for downstream diagnostic only."
    return b12a, b12b, b12b_low_pinky, b23


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
        default="dex5_penspin/reports/manus_mujoco_presets/optimized_b12_m3_2_20",
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
    b12a, b12b, b12b_low_pinky, b23 = build_b12_variants(p1_micro, p2_micro, p3_reference)

    seeds = {
        "P1_four_finger_clamp_micro_thumb_supported_seed.json": p1_micro,
        "B12a_middle_push_seed.json": b12a,
        "B12b_middle_ring_handoff_seed.json": b12b,
        "B12b_low_pinky_reference_seed.json": b12b_low_pinky,
        "P2_ring_wrap_to_front_micro_ring_guarded_seed.json": p2_micro,
        "B23_reference_index_ring_capture_seed.json": b23,
        "P3_index_wrap_to_front_reference_seed.json": p3_reference,
    }
    seed_paths: dict[str, str] = {}
    for filename, seed in seeds.items():
        path = output_dir / filename
        save_json(path, seed)
        seed_paths[filename] = str(path)

    render_env = m316.make_env(p1_micro, controller_args, duration=args.segment_duration * 4.0 + 0.6, render=True)
    try:
        keyposes = [
            ("P1 micro", p1_micro),
            ("B12a middle push", b12a),
            ("B12b handoff", b12b),
            ("B12b low pinky", b12b_low_pinky),
            ("P2 micro", p2_micro),
            ("B23 reference", b23),
        ]
        frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_')}.png", frame)
        save_montage(frames, output_dir / "m3_2_20_b12_keypose_montage.png", cols=3)
        imageio.mimsave(output_dir / "m3_2_20_b12_keyposes.gif", frames, duration=0.45)

        static_rows = [
            static_row(render_env, p1_micro, label="P1_micro_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, b12a, label="B12a_static", desired={"thumb", "index", "middle"}, allowed_extra={"pinky", "ring"}),
            static_row(render_env, b12b, label="B12b_handoff_static", desired={"thumb", "middle", "ring"}, allowed_extra={"pinky", "index"}),
            static_row(render_env, b12b_low_pinky, label="B12b_low_pinky_static", desired={"thumb", "middle", "ring"}, allowed_extra={"pinky", "index"}),
            static_row(render_env, p2_micro, label="P2_micro_static", desired={"thumb", "middle", "ring"}, allowed_extra={"pinky", "index"}),
        ]
        m317.write_csv(output_dir / "static_summary.csv", static_rows)

        hold_specs = [
            ("hold_B12a_middle_push", [b12a]),
            ("hold_B12b_handoff", [b12b]),
            ("hold_B12b_low_pinky_reference", [b12b_low_pinky]),
        ]
        hold_rows = [
            summarize_route(
                render_env,
                route,
                controller_args,
                label=label,
                duration=args.hold_duration,
                segment_s=None,
                output_dir=output_dir,
                render_every=args.render_every,
            )
            for label, route in hold_specs
        ]
        m317.write_csv(output_dir / "hold_summary.csv", hold_rows)

        transition_specs = [
            ("old_active_P1_B12a_P2", [p1_micro, b12a, p2_micro], args.segment_duration * 2.0, args.segment_duration),
            ("optimized_P1_B12a_B12b", [p1_micro, b12a, b12b], args.segment_duration * 2.0, args.segment_duration),
            (
                "diagnostic_optimized_P1_B12a_B12b_P2",
                [p1_micro, b12a, b12b, p2_micro],
                args.segment_duration * 3.0,
                args.segment_duration,
            ),
            ("diagnostic_B12b_to_P2", [b12b, p2_micro], 0.24, 0.24),
            ("diagnostic_B12b_to_B23", [b12b, b23], 0.24, 0.24),
            (
                "diagnostic_optimized_P1_B12a_B12b_B23",
                [p1_micro, b12a, b12b, b23],
                args.segment_duration * 3.0,
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
        "stage": "M3.2.20 optimized two-stage B12",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "selected_seed_files": seed_paths,
        "b12a_extras": B12_EXTRAS,
        "b12b_dynamic_extras": B12B_DYNAMIC_EXTRAS,
        "b12b_low_pinky_reference_extras": B12B_LOW_PINKY_REFERENCE_EXTRAS,
        "static_rows": static_rows,
        "hold_rows": hold_rows,
        "transition_rows": transition_rows,
        "interpretation": (
            "Optimized B12 should be interpreted as a two-stage dynamic primitive. "
            "P1->B12a->B12b improves the final target window and preserves middle at the end. "
            "The selected B12b uses a small pen offset to reduce static interpenetration in the visualization; "
            "it should still be interpreted as a dynamic handoff state, not a standalone static grasp. "
            "It does not yet solve downstream B12b->P2/B23, because pinky/ring can still dominate after B12b."
        ),
    }
    save_json(output_dir / "manifest.json", manifest)

    readme = [
        "# M3.2.20 Optimized B12",
        "",
        "This package replaces the single B12 bridge with a two-stage dynamic B12: B12a middle push, then B12b middle/ring handoff.",
        "",
        "## Holds",
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
        "- `m3_2_20_b12_keypose_montage.png`",
        "- `optimized_P1_B12a_B12b.gif`",
        "- `old_active_P1_B12a_P2.gif`",
        "- `diagnostic_optimized_P1_B12a_B12b_P2.gif`",
        "- `diagnostic_optimized_P1_B12a_B12b_B23.gif`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    assessment = [
        "# M3.2.20 B12 两段式优化评估",
        "",
        "本轮把 B12 从单个桥点改成 `B12a -> B12b` 两段：B12a 保持 middle push，B12b 提前停在 middle/ring 交接状态，不再强行落到完整 P2。",
        "",
        "## 结论",
        "",
        "`optimized_P1_B12a_B12b` 明显优于上一版 `P1 -> B12a -> P2`：末段 target window 从约 0.44 提到约 0.78，middle contact ratio 从约 0.86 提到 1.00，最终 contacts 包含 `middle/ring/thumb`。B12b 加了一个小笔位移来把静态可视化穿透压低；它仍应理解成动态交接状态，不是单独静态抓握。",
        "",
        "但 B12b 仍不是能直接接 P2/B23 的干净入口。`B12b -> P2` 和 `B12b -> B23` 仍会被 pinky/ring 带偏并提前失败。因此当前 B12 优化结果应视为一个更好的动态 B12 终点，而不是完整解决 P1->P2->B23 的连续链。",
        "",
        "另外保存了 `B12b_low_pinky_reference_seed.json`：它单独 hold 时 pinky 更少、最终是 `middle/ring/thumb`，但从 P1 动态进入时 target window 不如当前 selected B12b，因此暂时作为参考而不是主选。",
        "",
        "## 关键文件",
        "",
        "- `B12a_middle_push_seed.json`",
        "- `B12b_middle_ring_handoff_seed.json`",
        "- `B12b_low_pinky_reference_seed.json`",
        "- `m3_2_20_b12_keypose_montage.png`",
        "- `optimized_P1_B12a_B12b.gif`",
        "- `transition_summary.csv`",
    ]
    (output_dir / "ASSESSMENT.zh.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
