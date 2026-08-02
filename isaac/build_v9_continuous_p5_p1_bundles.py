from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASES = (
    (1, "P5捕获到P1接近", "P1接近"),
    (2, "P1接近到接触", "P1接触"),
    (3, "P1接触到三指支撑", "P1三指支撑"),
    (4, "P1三指支撑到严格保持", "P1严格保持"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从同 episode 连续入口缓存构建 P5 到 P1 局部训练包。"
    )
    parser.add_argument("--cache-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "reports/m3_2_30_index_near_surface/"
            "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
        ),
    )
    parser.add_argument(
        "--max-penetration-mm", type=float, default=0.5
    )
    parser.add_argument("--max-force-n", type=float, default=6.0)
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="保留连续入口速度；仅用于必要的 reset 投影课程。",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    p1_hand = np.asarray(
        next(
            row
            for row in manifest["phases"]
            if int(row["phase"]) == 1
        )["hand_qpos"],
        dtype=np.float32,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "输入前缀": str(args.cache_prefix),
        "输出目录": str(args.output_dir),
        "最大穿透门槛_mm": args.max_penetration_mm,
        "最大接触力门槛_N": args.max_force_n,
        "速度缩放": args.velocity_scale,
        "阶段": {},
    }

    for stage, cache_label, output_label in PHASES:
        cache_path = args.cache_prefix.with_name(
            f"{args.cache_prefix.name}_{cache_label}.npz"
        )
        with np.load(cache_path) as data:
            source = {
                key: np.asarray(data[key]) for key in data.files
            }
        source_qpos = np.asarray(source["source_qpos"], dtype=np.float32)
        source_qvel = np.asarray(source["source_qvel"], dtype=np.float32)
        forces = np.asarray(
            source["source_contact_force_n"], dtype=np.float32
        )
        penetration = np.asarray(
            source["source_max_penetration_m"], dtype=np.float32
        )
        valid = (
            penetration <= args.max_penetration_mm / 1000.0
        ) & (forces.max(axis=1) <= args.max_force_n)
        if not np.any(valid):
            raise RuntimeError(f"{cache_path} 没有状态通过物理门槛")

        source_qpos = source_qpos[valid]
        source_qvel = args.velocity_scale * source_qvel[valid]
        source_target = np.asarray(
            source["source_target_qpos"], dtype=np.float32
        )[valid]
        source_action = np.asarray(
            source["source_policy_action"], dtype=np.float32
        )[valid]
        forces = forces[valid]
        penetration = penetration[valid]
        count = len(source_qpos)

        target_qpos = source_qpos.copy()
        target_hand = source_target.copy()
        if stage == 1:
            target_hand[:, 4:8] = (
                0.45 * source_target[:, 4:8]
                + 0.55 * p1_hand[4:8]
            )
        elif stage == 2:
            target_hand[:, 4:8] = p1_hand[4:8]
        else:
            # Index, middle and ring define the P1 closure pose. Thumb is
            # actively maintained through its contact gate.
            target_hand[:, :12] = p1_hand[:12]
            target_hand[:, 16:20] = p1_hand[16:20]
        target_qpos[:, :20] = target_hand

        payload = {
            "source_qpos": source_qpos,
            "source_qvel": source_qvel,
            "source_target_qpos": source_target,
            "source_policy_action": source_action,
            "target_qpos": target_qpos,
            "target_qvel": np.zeros((count, 26), dtype=np.float32),
            "source_contact_force_n": forces,
            "source_angular_speed_rad_s": np.linalg.norm(
                source_qvel[:, 23:26], axis=1
            ).astype(np.float32),
            "source_max_penetration_m": penetration,
            "original_source_bank_ids": np.flatnonzero(valid).astype(
                np.int64
            ),
            "phase_ids": np.full(count, stage, dtype=np.int64),
            "source_teacher_actions": np.zeros(
                (count, 0, 20), dtype=np.float32
            ),
        }
        output = args.output_dir / f"{output_label}_连续入口训练包.npz"
        np.savez_compressed(output, **payload)
        report["阶段"][output_label] = {
            "输入状态": int(len(valid)),
            "通过物理门槛": count,
            "输出": str(output),
            "目标阶段_id": stage,
        }

    report_path = args.output_dir / "连续入口训练包摘要.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
