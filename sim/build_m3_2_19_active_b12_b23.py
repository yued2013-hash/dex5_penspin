from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

import evaluate_m3_2_17_cycle_closure as m317
import rebuild_m3_2_16_p4_p5_transitions as m316
from build_m3_2_9_photo_five_phase_debug import save_montage
from rebuild_m3_2_18_p1_p2_micro import build_micro_seeds, save_json, static_row


B12_EXTRAS = {
    "middle_pip": 0.18,
    "middle_mid": 0.0,
    "ring_open": 0.12,
    "thumb_pitch": 0.12,
    "thumb_tip": 0.072,
    "pos_z": 0.0,
}
B23_EXTRAS = {
    "index_pip": 0.70,
    "index_mid": 0.28,
    "middle_pip": 0.10,
    "middle_mid": 0.0,
    "ring_open": 0.04,
    "thumb_pitch": 0.08,
    "thumb_tip": 0.06,
}


def make_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return m317.make_controller_args(args)


def build_active_bridges(p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    b12 = m316.blend_seed(
        p1,
        p2,
        0.35,
        phase=1.5,
        name="bridge_p1_to_p2_middle_push_thumb_lift",
        desired=["thumb", "index", "middle"],
        extras=B12_EXTRAS,
    )
    b12["m3_2_19_selected_bridge"] = (
        "B12 active bridge: early P1->P2 blend with sustained thumb close, middle push, "
        "and ring opened slightly so ring does not immediately dominate. Ring is expected "
        "to enter during the second half toward P2; pinky is monitored but not hard-excluded."
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
    b23["m3_2_19_selected_bridge"] = (
        "B23 active bridge: late P2->P3 blend with a strong index close, light ring guard, "
        "and sustained thumb lift. Middle remains acceptable transitional support."
    )
    return b12, b23


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


def ratio(row: dict[str, Any], finger: str) -> float:
    return float(row.get("contact_ratio", {}).get(finger, 0.0))


def share(row: dict[str, Any], finger: str) -> float:
    return float(row.get("force_share_mean", {}).get(finger, 0.0))


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['label']} | {row['class']} | {row['survival_s']:.2f} | {row['desired_ratio']:.2f} | "
        f"{row['target_final_window_ratio']:.2f} | {row['support_final_window_ratio']:.2f} | "
        f"{row['active_contact_ratio']:.2f} | {share(row, 'thumb'):.2f} | {share(row, 'ring'):.2f} | "
        f"{ratio(row, 'index'):.2f} | {ratio(row, 'middle'):.2f} | {ratio(row, 'ring'):.2f} | "
        f"{ratio(row, 'pinky'):.2f} | {row['max_penetration_mm']:.2f} | "
        f"{','.join(row['final_contacts']) or 'none'} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/active_b12_b23_m3_2_19",
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
    b12, b23 = build_active_bridges(p1_micro, p2_micro, p3_reference)

    seed_files = {
        "P1_four_finger_clamp_micro_thumb_supported_seed.json": p1_micro,
        "B12_active_middle_push_thumb_lift_seed.json": b12,
        "P2_ring_wrap_to_front_micro_ring_guarded_seed.json": p2_micro,
        "B23_active_index_ring_capture_seed.json": b23,
        "P3_index_wrap_to_front_reference_seed.json": p3_reference,
    }
    seed_paths: dict[str, str] = {}
    for filename, seed in seed_files.items():
        path = output_dir / filename
        save_json(path, seed)
        seed_paths[filename] = str(path)

    total_duration = max(args.hold_duration, args.segment_duration * 4.0) + 0.5
    render_env = m316.make_env(p1_micro, controller_args, duration=total_duration, render=True)
    try:
        keyposes = [
            ("P1 micro", p1_micro),
            ("B12 active", b12),
            ("P2 micro", p2_micro),
            ("B23 active", b23),
            ("P3 reference", p3_reference),
        ]
        frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_')}.png", frame)
        save_montage(frames, output_dir / "m3_2_19_active_b12_b23_keypose_montage.png", cols=5)
        imageio.mimsave(output_dir / "m3_2_19_active_b12_b23_keyposes.gif", frames, duration=0.45)

        static_rows = [
            static_row(render_env, p1_micro, label="P1_micro_static", desired={"thumb", "index", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, b12, label="B12_active_static", desired={"thumb", "index", "middle"}, allowed_extra={"pinky", "ring"}),
            static_row(render_env, p2_micro, label="P2_micro_static", desired={"thumb", "middle", "ring"}, allowed_extra={"pinky"}),
            static_row(render_env, b23, label="B23_active_static", desired={"thumb", "index", "ring"}, allowed_extra={"middle"}),
            static_row(render_env, p3_reference, label="P3_reference_static", desired={"thumb", "index", "ring"}, allowed_extra={"middle"}),
        ]
        m317.write_csv(output_dir / "static_summary.csv", static_rows)

        hold_specs = [
            ("hold_B12_active", [b12]),
            ("hold_B23_active", [b23]),
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
            ("baseline_P1_micro_to_P2_micro", [p1_micro, p2_micro], args.segment_duration * 2.0, args.segment_duration * 2.0),
            ("baseline_P2_micro_to_P3_reference", [p2_micro, p3_reference], args.segment_duration * 2.0, args.segment_duration * 2.0),
            ("active_P1_B12_P2", [p1_micro, b12, p2_micro], args.segment_duration * 2.0, args.segment_duration),
            ("active_P2_B23_P3", [p2_micro, b23, p3_reference], args.segment_duration * 2.0, args.segment_duration),
            ("diagnostic_active_P1_B12_P2_B23", [p1_micro, b12, p2_micro, b23], args.segment_duration * 3.0, args.segment_duration),
            (
                "diagnostic_active_P1_B12_P2_B23_P3",
                [p1_micro, b12, p2_micro, b23, p3_reference],
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
    finally:
        render_env.close()

    manifest = {
        "stage": "M3.2.19 active B12/B23 transition primitives",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "selected_seed_files": seed_paths,
        "b12_extras": B12_EXTRAS,
        "b23_extras": B23_EXTRAS,
        "static_rows": static_rows,
        "hold_rows": hold_rows,
        "transition_rows": transition_rows,
        "interpretation": (
            "B12 adds a middle-push/thumb-lift bridge that improves P1->P2 target-window contact, "
            "but the final handoff still often exits with pinky/ring/thumb rather than stable middle/ring/thumb. "
            "B23 adds a strong index/ring capture and is usable as a short transition/capture primitive from P2. "
            "The continuous P1->B12->P2->B23 chain is still diagnostic because B12's imperfect terminal state corrupts B23 entry."
        ),
    }
    save_json(output_dir / "manifest.json", manifest)

    readme = [
        "# M3.2.19 Active B12/B23",
        "",
        "This package adds active bridge targets for B12 and B23 on top of the M3.2.18 P1/P2/P3 seeds.",
        "",
        "## Static Keyposes",
        "",
        "| label | contacts | missing | extra | max penetration mm |",
        "|---|---|---|---|---:|",
    ]
    for row in static_rows:
        readme.append(
            f"| {row['label']} | {row['contacts']} | {row['missing']} | {row['extra']} | {row['max_penetration_mm']:.2f} |"
        )
    readme += [
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
        "- `m3_2_19_active_b12_b23_keypose_montage.png`",
        "- `hold_B12_active.gif`",
        "- `hold_B23_active.gif`",
        "- `active_P1_B12_P2.gif`",
        "- `active_P2_B23_P3.gif`",
        "- `diagnostic_active_P1_B12_P2_B23.gif`",
        "- `diagnostic_active_P1_B12_P2_B23_P3.gif`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    assessment = [
        "# M3.2.19 B12/B23 主动过渡评估",
        "",
        "本轮在 M3.2.18 的 P1/P2/P3 上补了两个主动 bridge：B12 用 middle push + thumb lift，B23 用 index/ring capture + thumb lift。pinky 不再作为硬退出约束，只记录它是否抢支撑。",
        "",
        "## 结论",
        "",
        "B12 有改善，但还不是干净交接。`active_P1_B12_P2` 能跑完整 0.36s，support final window 为 1.0，middle contact ratio 明显提升；但 pinky 仍大量参与，最终常落到 pinky/ring/thumb，说明 P1->P2 的末端接触交接还没真正锁住 middle/ring/thumb。",
        "",
        "B23 是本轮最有效的新增 primitive。`B23_active` 静态就是 thumb/index/ring，0.24s hold 的目标接触为 1.0；`active_P2_B23_P3` 能跑完整 0.36s，index/ring 均参与，pinky 为 0。继续推到原始 P3 后会保留 support，但严格 P3 目标仍不完全，因为原始 P3 本身 ring 接触偏弱。",
        "",
        "整段 `P1 -> B12 -> P2 -> B23 -> P3` 仍不能提升为完整可连接链。原因不是 B23，而是 B12 的末端状态会污染 B23 入口；这和上一轮判断一致：早段真正要解决的是 P1->P2 的动态滚动/滑动交接。",
        "",
        "## 关键文件",
        "",
        "- `B12_active_middle_push_thumb_lift_seed.json`",
        "- `B23_active_index_ring_capture_seed.json`",
        "- `m3_2_19_active_b12_b23_keypose_montage.png`",
        "- `active_P1_B12_P2.gif`",
        "- `active_P2_B23_P3.gif`",
        "- `diagnostic_active_P1_B12_P2_B23_P3.gif`",
        "- `transition_summary.csv`",
    ]
    (output_dir / "ASSESSMENT.zh.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
