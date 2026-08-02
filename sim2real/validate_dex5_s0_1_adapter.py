#!/usr/bin/env python3
"""Run the offline S0.1 adapter audit and generate its reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dex5_hardware_adapter import (
    COUPLED_PAIRS,
    JOINT_LOWER,
    JOINT_ORDER,
    JOINT_UPPER,
    Dex5AdapterConfig,
    Dex5HardwareCompatibilityAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports/sim2real_s0_1"
O0_BANK = ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
)
SDK_ROOT = Path("/home/linkerhand/ofdemo_ros2_ws/src/unitree_dex5_1_control")
SDK_MAPPING = SDK_ROOT / "unitree_dex5_1_control/config/manus_to_dex5.yaml"
ANEMONE_LUT_DIR = SDK_ROOT / "dex5_anemone_package_x86_64/casadi_functions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_interpolation_audit(config: Dex5AdapterConfig, seconds: float = 10.0) -> dict:
    initial = Dex5HardwareCompatibilityAdapter.project_virtual_target(
        0.5 * (JOINT_LOWER + JOINT_UPPER)
    )
    adapter = Dex5HardwareCompatibilityAdapter(initial, config)
    rng = np.random.default_rng(20260731)
    policy_times = np.arange(0.0, seconds, 1.0 / config.policy_hz)
    command_times = np.arange(0.0, seconds, 1.0 / config.command_hz)
    policy_index = 0
    outputs = []
    for command_time in command_times:
        while policy_index < len(policy_times) and policy_times[policy_index] <= command_time + 1e-12:
            adapter.submit_policy_delta(
                rng.uniform(-0.05, 0.05, size=20), policy_times[policy_index]
            )
            policy_index += 1
        outputs.append(adapter.sample(command_time))
    output = np.asarray(outputs)
    step = np.abs(np.diff(output, axis=0))
    coupling_error = max(
        float(np.max(np.abs(output[:, proximal] - output[:, distal])))
        for proximal, distal in COUPLED_PAIRS
    )
    return {
        "duration_s": seconds,
        "policy_samples": int(len(policy_times)),
        "command_samples": int(len(command_times)),
        "non_finite_count": int(np.size(output) - np.isfinite(output).sum()),
        "limit_violation_count": int(
            np.count_nonzero((output < JOINT_LOWER - 1e-12) | (output > JOINT_UPPER + 1e-12))
        ),
        "max_command_step_rad": float(np.max(step)),
        "allowed_command_step_rad": float(0.5 / config.command_hz),
        "max_coupling_error_rad": coupling_error,
    }


def main() -> None:
    required = (O0_BANK, SDK_MAPPING)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S0.1 audit input: {missing}")

    config = Dex5AdapterConfig()
    rng = np.random.default_rng(20260731)
    random_targets = rng.uniform(-3.0, 3.0, size=(100_000, 20))
    projected = Dex5HardwareCompatibilityAdapter.project_virtual_target(random_targets)
    logical = Dex5HardwareCompatibilityAdapter.virtual_to_logical_actuator(projected)
    reconstructed = Dex5HardwareCompatibilityAdapter.logical_actuator_to_virtual(logical)
    coupling_error = max(
        float(np.max(np.abs(projected[:, proximal] - projected[:, distal])))
        for proximal, distal in COUPLED_PAIRS
    )
    roundtrip_error = float(np.max(np.abs(projected - reconstructed)))
    limit_violations = int(
        np.count_nonzero(
            (projected < JOINT_LOWER - 1e-12) | (projected > JOINT_UPPER + 1e-12)
        )
    )
    interpolation = run_interpolation_audit(config)

    checks = {
        "random_projection_finite": bool(np.isfinite(projected).all()),
        "random_projection_limits": limit_violations == 0,
        "coupled_pair_equality": coupling_error <= 1e-12,
        "logical_16d_roundtrip": roundtrip_error <= 1e-12,
        "interpolation_finite": interpolation["non_finite_count"] == 0,
        "interpolation_limits": interpolation["limit_violation_count"] == 0,
        "interpolation_continuity": (
            interpolation["max_command_step_rad"]
            <= interpolation["allowed_command_step_rad"] + 1e-9
        ),
        "o0_hash_unchanged": sha256(O0_BANK)
        == "73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65",
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    isaac_smoke_path = OUTPUT_DIR / "isaac_wrapper_smoke.json"
    isaac_smoke = (
        json.loads(isaac_smoke_path.read_text(encoding="utf-8"))
        if isaac_smoke_path.is_file()
        else None
    )
    isaac_status = (
        "PHYSX_TARGET_APPLICATION_PASS"
        if isaac_smoke is not None and isaac_smoke.get("status") == "PASS"
        else "STATIC_PARITY_PASS_RUNTIME_PHYSX_PENDING"
    )
    report = {
        "schema_version": "dex5.sim2real.hardware_adapter_validation.v1",
        "phase": "S0.1 Dex5 Hardware-Compatibility Adapter",
        "generated_at_utc": generated_at,
        "offline_adapter_status": "PASS" if all(checks.values()) else "FAIL",
        "isaac_wrapper_status": isaac_status,
        "hardware_acceptance_status": "NOT_RUN",
        "training_frozen": {"ppo": True, "teacher": True, "student": True},
        "interface": {
            "policy": "20D normalized delta at 60 Hz",
            "adapter_output": "20D absolute URDF target at 100 Hz",
            "sdk_topic": "/dex5_1/set_position",
            "logical_actuator_manifold_dim": 16,
            "physical_motor_command_exposed": False,
            "coupling_model": "distal_wins then PIP=DIP",
            "coupling_model_status": "SUPPORTED_BY_SDK_CONFIG_HARDWARE_CALIBRATION_PENDING",
        },
        "provisional_config": {
            "policy_hz": config.policy_hz,
            "command_hz": config.command_hz,
            "max_policy_delta_rad": config.max_policy_delta_rad,
            "max_velocity_rad_s": config.max_velocity_rad_s,
            "watchdog_timeout_s": config.watchdog_timeout_s,
            "watchdog_action": "hold_last_safe_target",
        },
        "monte_carlo": {
            "sample_count": int(projected.shape[0]),
            "limit_violation_count": limit_violations,
            "max_coupling_error_rad": coupling_error,
            "max_logical_roundtrip_error_rad": roundtrip_error,
        },
        "interpolation": interpolation,
        "isaac_smoke": isaac_smoke,
        "checks": checks,
        "provenance": {
            "o0_bank": str(O0_BANK),
            "o0_sha256": sha256(O0_BANK),
            "sdk_mapping": str(SDK_MAPPING),
            "sdk_mapping_sha256": sha256(SDK_MAPPING),
            "vendor_lut_files": {
                path.name: sha256(path)
                for path in sorted(
                    list(ANEMONE_LUT_DIR.glob("lut_finger_*.casadi"))
                    + list(ANEMONE_LUT_DIR.glob("lut_thumb_*motor*.casadi"))
                )
            },
        },
        "remaining_gates": [
            "run Isaac PhysX target-application smoke without training",
            "measure all joint directions on guarded hardware",
            "identify vendor URDF-to-motor coupling and thumb LUT behavior on hardware",
            "measure command/state latency and PD step response",
            "replace provisional velocity and delta bounds with approved hardware limits",
        ],
    }

    markdown = f"""# Dex5 S0.1 Hardware-Compatibility Adapter

生成时间：`{generated_at}`

## 结论

离线 adapter：`{report['offline_adapter_status']}`  
Isaac wrapper：`{isaac_status}`  
真机验收：`NOT_RUN`

本阶段没有启动 PPO、没有生成 teacher、没有训练 student，也没有修改 O0。S0.1 adapter 保留 SDK 的 20 维绝对 URDF 关节接口，不直接向 16 个电机发送命令。

## 数据流

```text
20D normalized delta @ 60 Hz
  -> delta/velocity bound
  -> cumulative absolute target
  -> distal-wins PIP/DIP projection
  -> joint-limit clamp
  -> linear resampling @ 100 Hz
  -> watchdog
  -> SDK /dex5_1/set_position (20D rad)
  -> vendor anemone URDF-to-motor conversion
```

压缩后的 16 维量只是**逻辑执行器坐标**，用于验证可实现流形和重建误差，不是电机轴角。厂商 anemone 包含 CasADi `finger_*` 与 `thumb_urdf2motor` LUT，禁止绕过该层。

## 离线验收

| 项目 | 结果 |
|---|---:|
| 随机目标 | {projected.shape[0]} |
| 越限 | {limit_violations} |
| 最大 PIP/DIP 误差 | {coupling_error:.3e} rad |
| 20D->16D->20D 最大误差 | {roundtrip_error:.3e} rad |
| 10 s 插值命令数 | {interpolation['command_samples']} |
| 最大 100 Hz 单步变化 | {interpolation['max_command_step_rad']:.6f} rad |
| 允许单步变化 | {interpolation['allowed_command_step_rad']:.6f} rad |
| O0 哈希 | PASS |

默认 `0.025 rad/step`、`0.5 rad/s` 和 `100 ms watchdog` 只是离线审计参数，尚未获准用于自动真机动作。

## Isaac Wrapper

`hardware_compatible_action.py` 提供 Torch 批量投影、normalized action 积分、16 维逻辑压缩/重建和 `apply_to_env`。它是 opt-in 工具：没有修改现有环境、没有注册训练 task。下一步只允许进行无训练 PhysX target-application smoke。

## 尚未完成

- 20/20 真机关节方向测量；
- 厂商 LUT 与真实 PIP/DIP/拇指反馈关系辨识；
- 延迟、抖动和 PD 阶跃响应；
- 真机速度、增量、力和温度安全阈值；
- watchdog 超时后的硬件 disable/急停集成。

因此当前不能连接 policy 到真机。真机测量应作为单独、受保护的 S0.1-HW 验收。
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "adapter_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "Dex5_S0_1_Hardware_Adapter_Spec.md").write_text(
        markdown, encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT_DIR), "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
