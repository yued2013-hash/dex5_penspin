from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import evaluate_m3_2_17_cycle_closure as m317
import rebuild_m3_2_16_p4_p5_transitions as m316
from build_m3_2_9_photo_five_phase_debug import save_montage
from evaluate_m3_2_15_short_primitives import contact_metrics


P1_Q_DELTA = {
    1: 0.026985019070471527,
    2: 0.013492509535235763,
    9: 0.12200222731449273,
    10: 0.061001113657246364,
    13: -0.07271947808447689,
    14: -0.04363168685068613,
    18: 0.07338613098467814,
    19: 0.03669306549233907,
}
P1_POS_DELTA = np.array(
    [0.0011328985205840672, -0.001833815956171791, -0.002997552632815842],
    dtype=np.float64,
)

P2_Q_DELTA = {
    5: 0.040,
    6: 0.020,
    9: -0.040,
    10: -0.024,
    13: -0.120,
    14: -0.080,
}
P2_POS_DELTA = np.array([0.002, 0.0, -0.002], dtype=np.float64)


def q_delta_from_map(values: dict[int, float]) -> np.ndarray:
    q = np.zeros(20, dtype=np.float64)
    for idx, value in values.items():
        q[idx] = float(value)
    return m316.enforce_shared_motor_consistency(q)


def tag_seed(seed: dict[str, Any], *, label: str, note: str) -> dict[str, Any]:
    out = dict(seed)
    out["m3_2_18_rebuilt"] = True
    out["m3_2_18_label"] = label
    out["m3_2_18_note"] = note
    return out


def build_micro_seeds(base: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p1 = m316.mutate_seed(
        base[1],
        q_delta=q_delta_from_map(P1_Q_DELTA),
        pos_delta=P1_POS_DELTA,
        dtheta_deg=0.0,
        axis_z_delta=0.0,
        label="m3_2_18_p1_micro_thumb_supported",
    )
    p1["name"] = "four_finger_clamp_micro_thumb_supported"
    p1["desired_contacts"] = ["thumb", "index", "middle", "ring"]
    p1 = tag_seed(
        p1,
        label="P1 micro",
        note=(
            "Small q/pen shift from P1; keeps four-finger contact as a short hold, "
            "opens pinky slightly, and lets thumb remain a meaningful support."
        ),
    )

    p2 = m316.mutate_seed(
        base[2],
        q_delta=q_delta_from_map(P2_Q_DELTA),
        pos_delta=P2_POS_DELTA,
        dtheta_deg=0.0,
        axis_z_delta=0.0,
        label="m3_2_18_p2_micro_ring_guarded",
    )
    p2["name"] = "ring_wrap_to_front_micro_ring_guarded"
    p2["desired_contacts"] = ["thumb", "middle", "ring"]
    p2 = tag_seed(
        p2,
        label="P2 micro",
        note=(
            "Small P2 retune; middle closes a little, ring relaxes a little, pinky opens, "
            "and the pen shifts about 2mm in x/z to bring thumb into the static support set."
        ),
    )

    p3 = dict(base[3])
    p3["desired_contacts"] = ["thumb", "index", "ring"]
    p3["m3_2_18_reference"] = "Original P3 kept unchanged to test P1->P2->P3 micro-connectability."
    return p1, p2, p3


def static_row(env: Any, seed: dict[str, Any], *, label: str, desired: set[str], allowed_extra: set[str]) -> dict[str, Any]:
    m316.reset_to_seed(env, seed)
    score, debug = m316.static_score(env, seed, desired, allowed_extra)
    metrics = contact_metrics(env)
    return {
        "label": label,
        "static_score": float(score),
        "desired": ",".join(sorted(desired)),
        "allowed_extra": ",".join(sorted(allowed_extra)) or "none",
        "contacts": ",".join(sorted(metrics["fingers"])) or "none",
        "missing": ",".join(debug["missing"]) or "none",
        "extra": ",".join(debug["extra"]) or "none",
        "max_penetration_mm": float(debug["max_penetration_mm"]),
    }


def make_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return m317.make_controller_args(args)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/p1_p2_micro_rebuild_m3_2_18",
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

    seed_outputs = {
        "P1_four_finger_clamp_micro_thumb_supported_seed.json": p1_micro,
        "P2_ring_wrap_to_front_micro_ring_guarded_seed.json": p2_micro,
        "P3_index_wrap_to_front_reference_seed.json": p3_reference,
    }
    seed_output_paths: dict[str, str] = {}
    for filename, seed in seed_outputs.items():
        path = output_dir / filename
        save_json(path, seed)
        seed_output_paths[filename] = str(path)

    render_env = m316.make_env(
        p1_micro,
        controller_args,
        duration=max(args.hold_duration, args.segment_duration * 3.0) + 0.5,
        render=True,
    )
    try:
        keyposes = [
            ("P1 original", base[1]),
            ("P1 micro", p1_micro),
            ("P2 original", base[2]),
            ("P2 micro", p2_micro),
            ("P3 reference", p3_reference),
        ]
        keypose_frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, keypose_frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_')}.png", frame)
        save_montage(keypose_frames, output_dir / "m3_2_18_p1_p2_p3_keypose_montage.png", cols=5)
        imageio.mimsave(output_dir / "m3_2_18_p1_p2_p3_keyposes.gif", keypose_frames, duration=0.45)

        static_rows = [
            static_row(
                render_env,
                p1_micro,
                label="P1_micro_static",
                desired={"thumb", "index", "middle", "ring"},
                allowed_extra=set(),
            ),
            static_row(
                render_env,
                p2_micro,
                label="P2_micro_static",
                desired={"thumb", "middle", "ring"},
                allowed_extra=set(),
            ),
            static_row(
                render_env,
                p3_reference,
                label="P3_reference_static",
                desired={"thumb", "index", "ring"},
                allowed_extra={"middle"},
            ),
        ]
        m317.write_csv(output_dir / "static_summary.csv", static_rows)

        hold_specs = [
            ("hold_P1_micro", [p1_micro]),
            ("hold_P2_micro", [p2_micro]),
            ("hold_P3_reference", [p3_reference]),
        ]
        hold_rows: list[dict[str, Any]] = []
        for label, route in hold_specs:
            row, _ = m317.summarize_route(
                render_env,
                route,
                controller_args,
                label=label,
                duration=args.hold_duration,
                segment_s=None,
                output_dir=output_dir,
                render_every=args.render_every,
            )
            hold_rows.append(row)
        m317.write_csv(output_dir / "hold_summary.csv", hold_rows)

        transition_specs = [
            ("transition_P1_micro_to_P2_micro", [p1_micro, p2_micro], args.segment_duration),
            ("transition_P2_micro_to_P3_reference", [p2_micro, p3_reference], args.segment_duration),
            (
                "diagnostic_P1_micro_P2_micro_P3_reference",
                [p1_micro, p2_micro, p3_reference],
                args.segment_duration * 2.0,
            ),
            (
                "diagnostic_original_P1_P2_P3",
                [base[1], base[2], base[3]],
                args.segment_duration * 2.0,
            ),
        ]
        transition_rows: list[dict[str, Any]] = []
        for label, route, duration in transition_specs:
            row, _ = m317.summarize_route(
                render_env,
                route,
                controller_args,
                label=label,
                duration=duration,
                segment_s=args.segment_duration,
                output_dir=output_dir,
                render_every=args.render_every,
            )
            transition_rows.append(row)
        m317.write_csv(output_dir / "transition_summary.csv", transition_rows)
    finally:
        render_env.close()

    manifest = {
        "stage": "M3.2.18 P1/P2 micro rebuild and P1->P2->P3 diagnostic",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "selected_seed_files": seed_output_paths,
        "p1_micro_delta": {
            "q_delta": P1_Q_DELTA,
            "pos_delta": P1_POS_DELTA.astype(float).tolist(),
            "dtheta_deg": 0.0,
            "axis_z_delta": 0.0,
        },
        "p2_micro_delta": {
            "q_delta": P2_Q_DELTA,
            "pos_delta": P2_POS_DELTA.astype(float).tolist(),
            "dtheta_deg": 0.0,
            "axis_z_delta": 0.0,
        },
        "static_rows": static_rows,
        "hold_rows": hold_rows,
        "transition_rows": transition_rows,
        "interpretation": (
            "P1/P2 have been rebuilt as short-hold micro keyposes. Under the micro-change constraint, "
            "P1->P2->P3 is still a diagnostic chain rather than a confirmed connectable primitive: "
            "P1->P2 keeps support but attracts pinky/non-target contact, and P2->P3 does not reliably recover "
            "strict thumb/index/ring final contact."
        ),
    }
    save_json(output_dir / "manifest.json", manifest)

    readme = [
        "# M3.2.18 P1/P2 Micro Rebuild",
        "",
        "This package rebuilds P1/P2 with small pose changes only, keeps P3 as the reference target, and visualizes P1->P2->P3.",
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
        "- `m3_2_18_p1_p2_p3_keypose_montage.png`",
        "- `m3_2_18_p1_p2_p3_keyposes.gif`",
        "- `hold_P1_micro.gif`",
        "- `hold_P2_micro.gif`",
        "- `transition_P1_micro_to_P2_micro.gif`",
        "- `transition_P2_micro_to_P3_reference.gif`",
        "- `diagnostic_P1_micro_P2_micro_P3_reference.gif`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    assessment = [
        "# M3.2.18 P1/P2 微调重建评估",
        "",
        "本轮只重建 P1/P2，并保持 P3 原始目标不变，用来检查 `P1 -> P2 -> P3` 早段 primitive 是否能靠微调接上。",
        "",
        "## 结论",
        "",
        "P1/P2 可以作为短时 key pose：P1_micro 在 0.24s hold 中保持 thumb/index/middle/ring，P2_micro 在 0.24s hold 中保持 thumb/middle/ring，且 pinky 不作为静态主支撑。",
        "",
        "但 `P1_micro -> P2_micro -> P3_reference` 还不能提升为可用 primitive。P1 到 P2 的动态插值仍会把 pinky 或非目标接触拉进支撑；P2 到 P3 可以短时不掉，但末端没有稳定恢复 P3 需要的 thumb/index/ring 目标接触。",
        "",
        "这说明 P1/P2 的静态几何已经比原始版更适合作为短时接触态，但早段链路的主要问题不是单个姿态，而是 P1->P2 与 P2->P3 之间缺少真实的滚动/滑动过渡控制。",
        "",
        "## 下一步建议",
        "",
        "不要继续强迫 P1/P2 静态姿态大改。更值得做的是补一个 B12 和 B23 的主动 transition primitive：给 thumb 一个持续上托/闭合项，同时给 index/middle/ring 分工，显式推动笔滚到 P2/P3 接触点，pinky 保持退出。",
        "",
        "## 关键文件",
        "",
        "- `P1_four_finger_clamp_micro_thumb_supported_seed.json`",
        "- `P2_ring_wrap_to_front_micro_ring_guarded_seed.json`",
        "- `m3_2_18_p1_p2_p3_keypose_montage.png`",
        "- `hold_P1_micro.gif`",
        "- `hold_P2_micro.gif`",
        "- `transition_P1_micro_to_P2_micro.gif`",
        "- `transition_P2_micro_to_P3_reference.gif`",
        "- `diagnostic_P1_micro_P2_micro_P3_reference.gif`",
        "- `hold_summary.csv`",
        "- `transition_summary.csv`",
    ]
    (output_dir / "ASSESSMENT.zh.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
