from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import (
    concatenate_payloads,
    convert_legacy_source,
    grouped_split,
    save_npz_atomic,
    sha256_file,
    validate_r0_arrays,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/canonical_reset_sources_o0.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/candidate"
)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _derive_trajectory_source(
    source_path: Path,
    definition: dict[str, object],
    output_dir: Path,
) -> Path:
    canonical_id = str(definition["canonical_id"])
    stage_id = int(definition["stage_id"])
    selection = str(definition.get("selection", "first"))
    horizon = int(definition.get("continuation_control_steps", 8))
    if horizon < 1 or horizon > 8:
        raise ValueError("continuation_control_steps must be within [1, 8]")
    original_sha256 = sha256_file(source_path)
    with np.load(source_path, allow_pickle=False) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        stages = np.asarray(data["stage_id"], dtype=np.int64)
        qpos_all = np.asarray(data["qpos"], dtype=np.float32)
        qvel_all = np.asarray(data["qvel"], dtype=np.float32)
        target_all = np.asarray(data["processed_target"], dtype=np.float32)
        action_all = np.asarray(data["action"], dtype=np.float32)
        contact_all = np.asarray(data["contact_force_n"], dtype=np.float32)
        penetration_all = np.asarray(data["max_penetration_m"], dtype=np.float32)
        episode_ids: list[int] = []
        frame_ids: list[int] = []
        for episode, length in enumerate(lengths):
            matches = np.flatnonzero(stages[episode, :length] == stage_id)
            if matches.size == 0:
                continue
            frame = int(matches[0] if selection == "first" else matches[-1])
            episode_ids.append(episode)
            frame_ids.append(frame)
        episodes = np.asarray(episode_ids, dtype=np.int64)
        frames = np.asarray(frame_ids, dtype=np.int64)
        if episodes.size == 0:
            raise ValueError(f"No stage {stage_id} frames found in {source_path}")
        qpos = qpos_all[episodes, frames]
        qvel = qvel_all[episodes, frames]
        target = target_all[episodes, frames]
        action = action_all[episodes, frames]
        contacts = contact_all[episodes, frames]
        penetration = penetration_all[episodes, frames]
        continuation_target = np.empty(
            (episodes.size, 8, 20), dtype=np.float32
        )
        continuation_action = np.empty_like(continuation_target)
        continuation_length = np.zeros(episodes.size, dtype=np.int64)
        for row, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
            remaining = int(lengths[episode] - frame)
            valid = min(horizon, remaining)
            continuation_length[row] = valid
            target_rows = target_all[episode, frame : frame + valid]
            action_rows = action_all[episode, frame : frame + valid]
            continuation_target[row, :valid] = target_rows
            continuation_action[row, :valid] = action_rows
            continuation_target[row, valid:] = target_rows[-1]
            continuation_action[row, valid:] = action_rows[-1]

    derived_path = output_dir / "derived_sources" / f"{canonical_id}_trajectory_source.npz"
    save_npz_atomic(
        derived_path,
        {
            "source_qpos": qpos,
            "source_qvel": qvel,
            "source_target_qpos": target,
            "source_policy_action": action,
            "source_contact_force_n": contacts,
            "source_max_penetration_m": penetration,
            "source_trajectory_ids": episodes,
            "source_episode_step": frames,
            "phase_ids": np.full(episodes.size, stage_id, dtype=np.int64),
            "robot_continuation_q_target": continuation_target,
            "robot_continuation_action": continuation_action,
            "robot_continuation_valid_length": continuation_length,
            "source_original_uri": np.asarray(str(source_path)),
            "source_original_sha256": np.asarray(original_sha256),
        },
    )
    return derived_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build O0 canonical reset candidate banks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config_path = _resolve(args.config).resolve()
    output_dir = _resolve(args.output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    asset_path = _resolve(config["asset_path"]).resolve()
    asset_hash = "sha256:" + sha256_file(asset_path)
    payloads: list[dict[str, np.ndarray]] = []
    report_rows: list[dict[str, object]] = []

    for definition in config["sources"]:
        original_source_path = _resolve(definition["path"]).resolve()
        source_path = original_source_path
        if definition.get("format", "reset_bank") == "trajectory":
            source_path = _derive_trajectory_source(
                source_path, definition, output_dir
            )
        payload = convert_legacy_source(
            source_path,
            definition["canonical_id"],
            definition["state_type"],
            source_label=definition["source_label"],
            asset_hash=asset_hash,
            environment_version=config["environment_version"],
            physics_version=config["physics_version"],
            phase_id=definition.get("phase_id"),
            mass=float(config["physics"]["mass"]),
            static_friction=float(config["physics"]["static_friction"]),
            dynamic_friction=float(config["physics"]["dynamic_friction"]),
            com_offset=np.asarray(config["physics"]["com_offset"], dtype=np.float32),
            physics_dt=float(config["physics"]["physics_dt"]),
            control_dt=float(config["physics"]["control_dt"]),
            solver_position_iterations=int(config["physics"]["solver_position_iterations"]),
            solver_velocity_iterations=int(config["physics"]["solver_velocity_iterations"]),
            contact_offset=float(config["physics"]["contact_offset"]),
            rest_offset=float(config["physics"]["rest_offset"]),
        )
        split = grouped_split(payload["metadata_source_trajectory_id"])
        payload["metadata_split"] = split
        r0_precheck, checks = validate_r0_arrays(payload)
        payload["labels_r0_precheck_pass"] = r0_precheck
        payload["labels_r0_precheck_reasons"] = np.asarray(
            [
                "pass"
                if r0_precheck[row]
                else ",".join(key for key, value in checks.items() if not value[row])
                for row in range(r0_precheck.size)
            ],
            dtype="U160",
        )
        canonical_id = definition["canonical_id"]
        output_path = output_dir / f"{canonical_id}_candidate.npz"
        save_npz_atomic(output_path, payload)
        payloads.append(payload)
        report_rows.append(
            {
                "canonical_id": canonical_id,
                "state_type": definition["state_type"],
                "source": str(source_path),
                "source_sha256": sha256_file(source_path),
                "original_source": str(original_source_path),
                "original_source_sha256": sha256_file(original_source_path),
                "rows": int(r0_precheck.size),
                "unique_source_trajectories": int(
                    np.unique(payload["metadata_source_trajectory_id"]).size
                ),
                "r0_precheck_pass": int(r0_precheck.sum()),
                "split_counts": {
                    name: int(np.sum(split == name)) for name in ("train", "validation", "test")
                },
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            }
        )

    combined = concatenate_payloads(payloads)
    canonical_ids = [str(item["canonical_id"]) for item in config["sources"]]
    if canonical_ids == ["P1", "P2", "P3", "P4", "P5"]:
        combined_name = "P1_P5_candidate.npz"
    elif canonical_ids == ["C1", "C2", "C3", "C4", "C5", "C6"]:
        combined_name = "C1_C6_candidate.npz"
    else:
        combined_name = "canonical_candidate_combined.npz"
    combined_path = output_dir / combined_name
    save_npz_atomic(combined_path, combined)
    report = {
        "status": "candidate_only_not_physx_validated",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "asset": str(asset_path),
        "asset_hash": asset_hash,
        "canonical_banks": report_rows,
        "combined": str(combined_path),
        "combined_rows": int(combined["metadata_state_id"].shape[0]),
        "combined_sha256": sha256_file(combined_path),
    }
    write_json_atomic(output_dir / "candidate_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
