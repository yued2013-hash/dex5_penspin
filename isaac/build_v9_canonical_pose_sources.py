from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/sources"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build small-noise reset sources around exact V9 canonical poses."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phases", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--joint-noise-std", type=float, default=0.004)
    parser.add_argument("--joint-noise-clip", type=float, default=0.010)
    parser.add_argument("--pen-position-noise-std", type=float, default=0.00015)
    parser.add_argument("--pen-position-noise-clip", type=float, default=0.0004)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_map = {int(row["phase"]): row for row in manifest["phases"]}
    reports = []
    for phase in args.phases:
        row = phase_map[phase]
        rng = np.random.default_rng(args.seed + phase)
        hand_center = np.asarray(row["hand_qpos"], dtype=np.float32)
        pen_center = np.concatenate(
            (
                np.asarray(row["pen_pos"], dtype=np.float32),
                np.asarray(row["pen_quat"], dtype=np.float32),
            )
        )
        hand_noise = np.clip(
            rng.normal(0.0, args.joint_noise_std, (args.count, 20)),
            -args.joint_noise_clip,
            args.joint_noise_clip,
        ).astype(np.float32)
        pen_position_noise = np.clip(
            rng.normal(0.0, args.pen_position_noise_std, (args.count, 3)),
            -args.pen_position_noise_clip,
            args.pen_position_noise_clip,
        ).astype(np.float32)
        hand = hand_center[None, :] + hand_noise
        pen = np.repeat(pen_center[None, :], args.count, axis=0)
        pen[:, :3] += pen_position_noise
        hand[0] = hand_center
        pen[0] = pen_center
        qpos = np.concatenate((hand, pen), axis=1).astype(np.float32)
        qvel = np.zeros((args.count, 26), dtype=np.float32)
        output = output_dir / f"P{phase}_v9_pose_source.npz"
        np.savez_compressed(
            output,
            source_qpos=qpos,
            source_qvel=qvel,
            source_target_qpos=hand.copy(),
            source_policy_action=np.zeros((args.count, 20), dtype=np.float32),
            source_teacher_actions=np.zeros((args.count, 1, 20), dtype=np.float32),
            target_qpos=qpos.copy(),
            target_qvel=qvel.copy(),
            phase_ids=np.full(args.count, phase, dtype=np.int64),
            source_trajectory_ids=np.arange(args.count, dtype=np.int64),
            canonical_hand_rms_rad=np.sqrt(np.mean(hand_noise**2, axis=1)).astype(
                np.float32
            ),
        )
        reports.append(
            {
                "phase": phase,
                "name": row["name"],
                "count": args.count,
                "expected_contacts": row["contact_set"],
                "max_hand_rms_rad": float(
                    np.sqrt(np.mean(hand_noise**2, axis=1)).max()
                ),
                "max_pen_position_offset_mm": float(
                    np.linalg.norm(pen_position_noise, axis=1).max() * 1000.0
                ),
                "output": str(output),
            }
        )
    report = {
        "status": "candidate_sources_only_not_physx_validated",
        "manifest": str(manifest_path),
        "sources": reports,
    }
    (output_dir / "source_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
