from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the five-stage P5-to-P1 curriculum reset bundle."
    )
    parser.add_argument(
        "--source",
        default=(
            "reports/m3_2_30_index_near_surface/p5_p1_curriculum/"
            "stable_p5_hold200_success_cache_1200.npz"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "reports/m3_2_30_index_near_surface/"
            "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/m3_2_30_index_near_surface/p5_p1_curriculum/"
            "p5_p1_five_stage_bundle.npz"
        ),
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=1.0,
        help="Scale reset hand and pen velocities while preserving pose/contact state.",
    )
    parser.add_argument("--max_source_penetration_mm", type=float, default=float("inf"))
    parser.add_argument("--max_source_angular_speed", type=float, default=float("inf"))
    parser.add_argument("--max_source_force_n", type=float, default=float("inf"))
    args = parser.parse_args()
    if not 0.0 <= args.velocity_scale <= 1.0:
        raise ValueError("--velocity_scale must be in [0, 1]")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    p1 = next(row for row in manifest["phases"] if int(row["phase"]) == 1)
    p1_hand = np.asarray(p1["hand_qpos"], dtype=np.float32)
    with np.load(args.source) as source:
        base = {key: np.asarray(source[key]) for key in source.files}

    source_mask = (
        np.asarray(base["source_max_penetration_m"])
        <= args.max_source_penetration_mm / 1000.0
    )
    source_mask &= (
        np.asarray(base["source_angular_speed_rad_s"])
        <= args.max_source_angular_speed
    )
    source_mask &= (
        np.asarray(base["source_contact_force_n"]).max(axis=1)
        <= args.max_source_force_n
    )
    if not np.any(source_mask):
        raise RuntimeError("No source rows passed the reset margin gates")
    base = {
        key: value[source_mask]
        if value.ndim > 0 and len(value) == len(source_mask)
        else value
        for key, value in base.items()
    }

    source_qpos = np.asarray(base["source_qpos"], dtype=np.float32)
    source_qvel = (
        args.velocity_scale
        * np.asarray(base["source_qvel"], dtype=np.float32)
    )
    source_targets = np.asarray(
        base["source_target_qpos"], dtype=np.float32
    )
    count = len(source_qpos)
    stage_payloads: list[dict[str, np.ndarray]] = []
    for stage in range(5):
        target_qpos = source_qpos.copy()
        target_hand = source_targets.copy()
        if stage == 1:
            target_hand[:, 4:8] = (
                0.45 * source_targets[:, 4:8] + 0.55 * p1_hand[4:8]
            )
        elif stage == 2:
            target_hand[:, 4:8] = p1_hand[4:8]
        elif stage >= 3:
            target_hand[:, :12] = p1_hand[:12]
        target_qpos[:, :20] = target_hand
        stage_payloads.append(
            {
                "source_qpos": source_qpos,
                "source_qvel": source_qvel,
                "source_target_qpos": source_targets,
                "source_policy_action": np.asarray(
                    base["source_policy_action"], dtype=np.float32
                ),
                "target_qpos": target_qpos,
                "target_qvel": np.zeros((count, 26), dtype=np.float32),
                "source_contact_force_n": np.asarray(
                    base["source_contact_force_n"], dtype=np.float32
                ),
                "source_angular_speed_rad_s": np.asarray(
                    base["source_angular_speed_rad_s"], dtype=np.float32
                ),
                "source_max_penetration_m": np.asarray(
                    base["source_max_penetration_m"], dtype=np.float32
                ),
                "original_source_bank_ids": np.asarray(
                    base["original_source_bank_ids"], dtype=np.int64
                ),
                "phase_ids": np.full(count, stage, dtype=np.int64),
            }
        )

    keys = stage_payloads[0].keys()
    payload = {
        key: np.concatenate([row[key] for row in stage_payloads])
        for key in keys
    }
    payload["source_teacher_actions"] = np.zeros(
        (len(payload["source_qpos"]), 0, 20), dtype=np.float32
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report = {
        "source": args.source,
        "manifest": args.manifest,
        "output": str(output),
        "source_rows": count,
        "total_rows": len(payload["source_qpos"]),
        "rows_per_stage": count,
        "velocity_scale": args.velocity_scale,
        "source_margin_gates": {
            "max_penetration_mm": args.max_source_penetration_mm,
            "max_angular_speed_rad_s": args.max_source_angular_speed,
            "max_force_n": args.max_source_force_n,
        },
        "target_semantics": (
            "Preserve each real P5 pen pose; progressively introduce "
            "canonical P1 middle and thumb/index/middle joint targets."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
