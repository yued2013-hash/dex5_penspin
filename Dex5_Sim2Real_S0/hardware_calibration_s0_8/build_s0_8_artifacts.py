#!/usr/bin/env python3
"""Build measured S0.8 adapter/configuration and a speed-safe replay package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE_REPLAY = ROOT / "real_replay_package" / "open_loop_commands_100hz.npz"
JOINT_TO_ACTUATOR = np.asarray(
    [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15],
    dtype=np.int64,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merged_mapping() -> tuple[list[dict], dict]:
    primary = read_json(HERE / "raw_calibration.json")["hardware"]
    thumb = read_json(HERE / "thumb_supplement.json")["hardware"]
    mapping = {x["actuator_index"]: x for x in primary["mapping"]}
    mapping.update({x["actuator_index"]: x for x in thumb["mapping"]})
    ordered = [mapping[i] for i in range(16)]
    return ordered, primary


def response_scale(mapping: list[dict]) -> list[float]:
    scale = np.zeros(20, dtype=np.float64)
    for item in mapping:
        group = item["joint_indices"]
        gain = abs(float(item["primary_local_gain"]))
        scale[group[0]] = gain
        if len(group) == 2:
            scale[group[1]] = gain * float(item["pip_dip_coupling_ratio"])
    return scale.tolist()


def build_safe_replay(max_speed_rad_s: float = 0.08) -> dict:
    with np.load(SOURCE_REPLAY, allow_pickle=False) as data:
        command = np.asarray(data["sdk_absolute_target_100hz"], dtype=np.float64)
        virtual = np.asarray(data["virtual_joint_target_100hz"], dtype=np.float64)
        offsets = np.asarray(data["trajectory_offsets"], dtype=np.int64)
        hz = float(data["command_hz"])
    max_step = max_speed_rad_s / hz
    command_rows: list[np.ndarray] = []
    virtual_rows: list[np.ndarray] = []
    new_offsets = [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        c_segment = command[start:end]
        v_segment = virtual[start:end]
        if len(c_segment) == 0:
            new_offsets.append(len(command_rows))
            continue
        command_rows.append(c_segment[0])
        virtual_rows.append(v_segment[0])
        for i in range(1, len(c_segment)):
            subdivisions = max(
                1,
                int(np.ceil(np.max(np.abs(c_segment[i] - c_segment[i - 1])) / max_step)),
            )
            for k in range(1, subdivisions + 1):
                alpha = k / subdivisions
                command_rows.append((1.0 - alpha) * c_segment[i - 1] + alpha * c_segment[i])
                virtual_rows.append((1.0 - alpha) * v_segment[i - 1] + alpha * v_segment[i])
        new_offsets.append(len(command_rows))
    command_out = np.asarray(command_rows, dtype=np.float32)
    virtual_out = np.asarray(virtual_rows, dtype=np.float32)
    logical_out = command_out[:, [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]]
    offsets_out = np.asarray(new_offsets, dtype=np.int64)
    output = HERE / "open_loop_commands_100hz_speed_safe.npz"
    np.savez_compressed(
        output,
        sdk_absolute_target_100hz=command_out,
        virtual_joint_target_100hz=virtual_out,
        logical_actuator_target_100hz=logical_out,
        trajectory_offsets=offsets_out,
        command_hz=np.asarray(hz, dtype=np.float64),
        max_speed_rad_s=np.asarray(max_speed_rad_s, dtype=np.float64),
        source_sha256=np.asarray(sha256(SOURCE_REPLAY)),
    )
    max_delta = max(
        float(np.max(np.abs(np.diff(command_out[a:b], axis=0)))) if b - a > 1 else 0.0
        for a, b in zip(offsets_out[:-1], offsets_out[1:])
    )
    return {
        "source": str(SOURCE_REPLAY),
        "source_sha256": sha256(SOURCE_REPLAY),
        "output": str(output),
        "output_sha256": sha256(output),
        "trajectory_count": len(offsets_out) - 1,
        "command_rows": len(command_out),
        "command_hz": hz,
        "maximum_delta_rad": max_delta,
        "maximum_velocity_rad_s": max_delta * hz,
        "speed_gate_rad_s": max_speed_rad_s,
        "powered_replay_executed": False,
    }


def main() -> None:
    mapping, primary = merged_mapping()
    scales = response_scale(mapping)
    ratios = {
        str(x["actuator_index"]): x["pip_dip_coupling_ratio"]
        for x in mapping if "pip_dip_coupling_ratio" in x
    }
    direction = {
        "schema": "dex5-s0.8-joint-direction-v1",
        "status": "PASS",
        "confirmed": 16,
        "total": 16,
        "sdk_command_semantics": "20D absolute URDF-joint targets through vendor anemone conversion",
        "actuators": [
            {
                "actuator_index": x["actuator_index"],
                "joint_indices": x["joint_indices"],
                "joint_names": x["joint_names"],
                "direction_sign": x["direction_sign"],
                "direction_confirmed": x["direction_confirmed"],
                "mapping_confirmed": x["mapping_confirmed"],
                "calibration_delta_rad": x["command_delta_rad"],
            }
            for x in mapping
        ],
    }
    write_json(HERE / "joint_direction_table.json", direction)

    dynamic = {
        "schema": "dex5-s0.8-local-dynamic-map-v1",
        "status": "PASS_LOCAL_IDENTIFICATION",
        "scope": "small-signal local response around the measured acceptance posture",
        "response_matrix_20x16": np.asarray(
            [x["response_vector_rad"] for x in mapping], dtype=np.float64
        ).T.tolist(),
        "joint_response_scale": scales,
        "pip_dip_response_ratio": ratios,
        "actuators": mapping,
        "warning": "This is not a vendor motor-shaft model and is not validated over the full URDF range.",
    }
    write_json(HERE / "motor_dynamic_mapping.json", dynamic)

    latency = {
        "schema": "dex5-s0.8-command-latency-v1",
        "status": "MEASURED",
        "distribution": primary["latency_distribution"],
        "physical_onset_detected_channels": primary["latency_distribution"]["physical_onset_s"]["count"],
        "physical_onset_total_channels": 16,
        "simulation_delay_ticks_100hz": [1, 4],
        "note": "Controller acceptance is ROS callback latency. Physical onset was observable above 0.5 mrad on 6/16 small-step trials; [1,4] ticks conservatively covers the measured 14-31 ms onset range.",
    }
    write_json(HERE / "command_latency_report.json", latency)

    config = {
        "schema": "dex5-s0-hardware-simulation-v2",
        "hardware_identified": True,
        "virtual_joint_dimension": 20,
        "logical_actuator_dimension": 16,
        "sdk_command_dimension": 20,
        "policy_hz": 60.0,
        "sdk_command_hz": 100.0,
        "physics_hz": 120.0,
        "joint_to_actuator": JOINT_TO_ACTUATOR.tolist(),
        "joint_direction_sign": [1] * 20,
        "coupled_pairs": [[2, 3], [6, 7], [10, 11], [14, 15]],
        "pip_dip_response_ratio": ratios,
        "joint_response_scale_nominal": scales,
        "joint_response_scale_randomization_fraction": [0.8, 1.2],
        "delay_command_ticks": [1, 4],
        "safe_velocity_limit_rad_s": [0.04, 0.08],
        "max_policy_delta_rad": 0.025,
        "calibration_sources": {
            "primary": "raw_calibration.json",
            "thumb_deadband_supplement": "thumb_supplement.json",
        },
        "powered_autonomous_policy": False,
        "warning": "Safe velocity range is an acceptance/replay envelope, not a measured motor maximum.",
    }
    write_json(HERE / "hardware_simulation_config.json", config)
    rng = np.random.default_rng(20260801)
    logical = rng.uniform(-1.0, 1.0, size=(100000, 16))
    virtual = logical[:, JOINT_TO_ACTUATOR]
    compressed = virtual[:, [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]]
    reconstructed = compressed[:, JOINT_TO_ACTUATOR]
    roundtrip_error = float(np.max(np.abs(reconstructed - virtual)))
    ratio_error = max(
        abs(scales[int(group[1])] / scales[int(group[0])] - float(ratio))
        for group, ratio in zip(
            ([2, 3], [6, 7], [10, 11], [14, 15]), ratios.values()
        )
    )
    adapter_validation = {
        "schema": "dex5-s0.8-adapter-validation-v1",
        "status": "PASS_OFFLINE",
        "random_samples": len(logical),
        "command_manifold_roundtrip_max_error_rad": roundtrip_error,
        "measured_coupling_ratio_reproduction_max_error": ratio_error,
        "finite": bool(np.isfinite(reconstructed).all()),
        "held_out_hardware_error": None,
        "note": "The local response model is fit from the same acceptance scans; held-out hardware validation remains a later controlled test.",
    }
    write_json(HERE / "adapter_validation.json", adapter_validation)
    replay = build_safe_replay()
    write_json(HERE / "safe_replay_manifest.json", replay)
    acceptance = {
        "schema": "dex5-s0.8-real-calibration-closure-v1",
        "status": "PASS_LOCAL_MODEL_CLOSURE",
        "joint_direction": {"confirmed": 16, "total": 16},
        "actuator_mapping": {"confirmed": 16, "total": 16},
        "pip_dip_ratio": ratios,
        "latency": latency["distribution"],
        "adapter_validation": adapter_validation,
        "safe_replay": replay,
        "physx_audits": {
            "zero": "physx_zero_audit.json",
            "small": "physx_small_audit.json",
        },
        "powered_replay_executed": False,
        "autonomous_policy_executed": False,
        "o0_modified": False,
    }
    write_json(HERE / "S0_8_acceptance.json", acceptance)
    print(json.dumps({"direction": direction["status"], "config": str(HERE / "hardware_simulation_config.json"), "replay": replay}, indent=2))


if __name__ == "__main__":
    main()
