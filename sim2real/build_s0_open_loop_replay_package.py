from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from dex5_hardware_adapter import (
    COUPLED_PAIRS,
    Dex5HardwareCompatibilityAdapter,
    JOINT_LOWER,
    JOINT_UPPER,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resample_60_to_100(commands: np.ndarray, speed_scale: float) -> np.ndarray:
    source_t = np.arange(commands.shape[0], dtype=np.float64) / 60.0
    target_count = int(np.floor(source_t[-1] / speed_scale * 100.0)) + 1
    target_t = np.arange(target_count, dtype=np.float64) / 100.0
    return np.stack(
        [
            np.interp(target_t * speed_scale, source_t, commands[:, joint])
            for joint in range(20)
        ],
        axis=-1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--speed-scale", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.speed_scale <= 1.0:
        raise ValueError("speed-scale must be in (0, 1]")

    with np.load(args.dataset, allow_pickle=False) as data:
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        sources = np.asarray(data["episode_source_trajectory_id"], dtype=np.int64)
        splits = np.asarray(data["episode_split"])
        sdk = np.asarray(data["sdk_absolute_target"], dtype=np.float32)
        virtual = np.asarray(data["adapter_projected_target"], dtype=np.float32)
        bank_ids = np.asarray(data["episode_bank_id"], dtype=np.int64)

        selected: list[int] = []
        used_sources: set[int] = set()
        for episode, (source, split) in enumerate(zip(sources, splits, strict=True)):
            if split != "train" or int(source) in used_sources:
                continue
            selected.append(episode)
            used_sources.add(int(source))
            if len(selected) == args.trajectories:
                break
        if len(selected) < args.trajectories:
            raise RuntimeError("not enough unique train sources for replay package")

    commands: list[np.ndarray] = []
    virtual_targets: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    for episode in selected:
        start, end = offsets[episode : episode + 2]
        command = resample_60_to_100(sdk[start:end], args.speed_scale)
        virtual_target = resample_60_to_100(virtual[start:end], args.speed_scale)
        commands.append(command)
        virtual_targets.append(virtual_target)
        timestamps.append(np.arange(command.shape[0], dtype=np.float32) / 100.0)

    command_offsets = np.concatenate(
        ([0], np.cumsum([command.shape[0] for command in commands]))
    ).astype(np.int64)
    command_rows = np.concatenate(commands, axis=0)
    virtual_rows = np.concatenate(virtual_targets, axis=0)
    logical_rows = Dex5HardwareCompatibilityAdapter.virtual_to_logical_actuator(
        virtual_rows
    ).astype(np.float32)
    timestamp_rows = np.concatenate(timestamps, axis=0)
    max_coupling_error = max(
        float(np.max(np.abs(command_rows[:, first] - command_rows[:, second])))
        for first, second in COUPLED_PAIRS
    )
    within_limits = bool(
        np.all(command_rows >= JOINT_LOWER - 1.0e-6)
        and np.all(command_rows <= JOINT_UPPER + 1.0e-6)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_path = args.output_dir / "open_loop_commands_100hz.npz"
    np.savez_compressed(
        command_path,
        sdk_absolute_target_100hz=command_rows,
        virtual_joint_target_100hz=virtual_rows,
        logical_actuator_target_100hz=logical_rows,
        timestamp_s=timestamp_rows,
        trajectory_offsets=command_offsets,
        source_episode_id=np.asarray(selected, dtype=np.int64),
        source_trajectory_id=sources[selected],
        source_bank_id=bank_ids[selected],
        command_hz=np.asarray(100.0, dtype=np.float32),
    )
    max_command_delta = max(
        float(np.max(np.abs(np.diff(command, axis=0)))) for command in commands
    )
    max_command_velocity = 100.0 * max_command_delta
    project_root = Path(__file__).resolve().parents[1]
    adapter_config_source = (
        project_root / "Dex5_Sim2Real_S0/contracts/hardware_adapter_config.json"
    )
    shutil.copy2(adapter_config_source, args.output_dir / "hardware_adapter_config.json")
    safety_limits = {
        "status": "PROVISIONAL_OFFLINE_LIMITS_ONLY",
        "command_hz": 100.0,
        "replay_speed_scale": args.speed_scale,
        "maximum_observed_command_delta_rad": max_command_delta,
        "maximum_observed_command_velocity_rad_s": max_command_velocity,
        "required_initial_replay_delta_cap_rad": 0.01,
        "absolute_joint_lower_rad": JOINT_LOWER.tolist(),
        "absolute_joint_upper_rad": JOINT_UPPER.tolist(),
        "workspace_limit": "PENDING_HARDWARE_MEASUREMENT",
        "watchdog_timeout_s": 0.1,
        "emergency_stop": "REQUIRED_NOT_YET_ACCEPTED",
    }
    (args.output_dir / "safety_limits.json").write_text(
        json.dumps(safety_limits, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "OFFLINE_ONLY_NOT_HARDWARE_APPROVED",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "command_file": command_path.name,
        "command_file_sha256": sha256(command_path),
        "trajectory_count": len(selected),
        "command_rows": int(command_rows.shape[0]),
        "command_hz": 100.0,
        "replay_speed_scale": args.speed_scale,
        "command_semantics": "20D absolute URDF-joint targets for the public Dex5 SDK",
        "virtual_target_semantics": "20D coupled virtual-joint destination",
        "logical_actuator_semantics": (
            "16D coupled diagnostic coordinates; vendor SDK performs the actual "
            "nonlinear URDF-to-motor conversion"
        ),
        "max_coupling_error_rad": max_coupling_error,
        "within_offline_urdf_limits": within_limits,
        "maximum_observed_100hz_delta_rad": max_command_delta,
        "maximum_observed_velocity_rad_s": max_command_velocity,
        "adapter_config": "hardware_adapter_config.json",
        "safety_limits": "safety_limits.json",
        "hardware_blockers": [
            "joint direction test not run",
            "PIP/DIP coupling identification not run",
            "latency and PD response not measured",
            "emergency stop behavior not accepted",
            "Dex5 currently has no power connection",
        ],
        "required_first_execution": "human-supervised open-loop replay at reduced speed and reduced range",
        "privileged_oracle_included": False,
        "autonomous_policy_execution": False,
    }
    (args.output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if not within_limits or max_coupling_error > 1.0e-7:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
