#!/usr/bin/env python3
"""Generate the read-only Dex5 Sim2Real-S0.0 interface audit artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = Path("/home/linkerhand/ofdemo_ros2_ws/src/unitree_dex5_1_control")
SDK_PACKAGE = SDK_ROOT / "unitree_dex5_1_control"
ENV_SOURCE = ROOT / "isaac/dex5_penspin_isaaclab/tasks/index_near_surface/index_near_surface_physics_env.py"
ORACLE_SOURCE = ROOT / "isaac/dex5_penspin_isaaclab/tasks/index_near_surface/privileged_oracle_physics_env.py"
SIM_URDF = ROOT / "sim/assets/dex5_urdf_import/Dex5-URDF-R.urdf"
REAL_URDF = SDK_ROOT / "Dex5-URDF-R/Dex5-URDF-R.urdf"
CONTROLLER_SOURCE = SDK_PACKAGE / "src/dex5_1_controller.cpp"
OUTPUT_DIR = ROOT / "reports/sim2real_s0_0"

SDK_ORDER = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
MOTOR_INDEX = (0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15)
FINGER_BY_BLOCK = ("index", "middle", "ring", "pinky", "thumb")
FINGERTIP_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_tuple(source: Path, variable: str) -> tuple[str, ...]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return tuple(value)
    raise RuntimeError(f"{variable} not found in {source}")


def urdf_joints(path: Path) -> tuple[list[str], dict[str, dict[str, float | str]]]:
    root = ET.parse(path).getroot()
    order: list[str] = []
    joints: dict[str, dict[str, float | str]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        name = str(joint.get("name"))
        limit = joint.find("limit")
        axis = joint.find("axis")
        order.append(name)
        joints[name] = {
            "axis": axis.get("xyz", "") if axis is not None else "",
            "lower_rad": float(limit.get("lower")) if limit is not None else float("nan"),
            "upper_rad": float(limit.get("upper")) if limit is not None else float("nan"),
            "urdf_effort": float(limit.get("effort")) if limit is not None else float("nan"),
            "urdf_velocity_rad_s": float(limit.get("velocity")) if limit is not None else float("nan"),
        }
    return order, joints


def require_source_contracts(controller_text: str) -> None:
    required = (
        "current_mode_(POSITION_CONTROL), publish_rate_(100.0)",
        "std::chrono::duration<double>(1.0 / publish_rate_)",
        "defaultPositionKp",
        "defaultPositionKd",
        "positions.size() != kDex5JointCount",
        "hand_cmd_publisher_->Write(hand_cmd)",
    )
    missing = [text for text in required if text not in controller_text]
    if missing:
        raise RuntimeError(f"SDK source contract changed; missing: {missing}")


def main() -> None:
    required = (ENV_SOURCE, ORACLE_SOURCE, SIM_URDF, REAL_URDF, CONTROLLER_SOURCE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")

    isaac_order = python_tuple(ENV_SOURCE, "DEX5_JOINT_ORDER")
    sim_doc_order, sim_joints = urdf_joints(SIM_URDF)
    real_doc_order, real_joints = urdf_joints(REAL_URDF)
    controller_text = CONTROLLER_SOURCE.read_text(encoding="utf-8")
    require_source_contracts(controller_text)

    if isaac_order != SDK_ORDER:
        raise RuntimeError("Isaac policy order does not match the audited SDK order")
    if set(sim_joints) != set(SDK_ORDER) or set(real_joints) != set(SDK_ORDER):
        raise RuntimeError("URDF joint set does not match the 20-joint interface")

    joint_rows = []
    limit_mismatches = []
    for index, name in enumerate(SDK_ORDER):
        sim = sim_joints[name]
        real = real_joints[name]
        equal_limits = (
            abs(float(sim["lower_rad"]) - float(real["lower_rad"])) <= 1e-9
            and abs(float(sim["upper_rad"]) - float(real["upper_rad"])) <= 1e-9
        )
        if not equal_limits:
            limit_mismatches.append(name)
        block = index // 4
        coupled_to = None
        independent = True
        if index in (3, 7, 11, 15):
            coupled_to = SDK_ORDER[index - 1]
            independent = False
        joint_rows.append({
            "policy_index": index,
            "sdk_joint_index": index,
            "name": name,
            "finger": FINGER_BY_BLOCK[block],
            "urdf_document_index_sim": sim_doc_order.index(name),
            "urdf_document_index_real": real_doc_order.index(name),
            "physical_motor_index": MOTOR_INDEX[index],
            "independent_physical_actuator": independent,
            "coupled_to": coupled_to,
            "sim_to_sdk_position": {"scale": 1.0, "offset_rad": 0.0},
            "direction_sign": 1,
            "direction_evidence": "same joint name, axis, limits, and URDF coordinate convention; hardware motion-sign test still required",
            "axis_xyz": sim["axis"],
            "lower_rad": sim["lower_rad"],
            "upper_rad": sim["upper_rad"],
            "urdf_velocity_rad_s": sim["urdf_velocity_rad_s"],
            "isaac_velocity_override_rad_s": 4.18,
            "isaac_effort_override": 0.93,
            "limit_match_sim_real_urdf": equal_limits,
        })

    generated_at = datetime.now(timezone.utc).isoformat()
    mapping = {
        "schema_version": "dex5.sim2real.joint_mapping.v1",
        "audit_phase": "Sim2Real-S0.0",
        "generated_at_utc": generated_at,
        "status": "STATIC_MAPPING_PASS_HARDWARE_DIRECTION_PENDING",
        "joint_count": 20,
        "independent_physical_motor_count": 16,
        "policy_and_sdk_order_equal": isaac_order == SDK_ORDER,
        "policy_order": list(isaac_order),
        "raw_urdf_document_order": sim_doc_order,
        "fingertip_force_order": list(FINGERTIP_ORDER),
        "warning": "Fingertip force arrays are thumb-first, while joint arrays are index-first.",
        "direct_command": {
            "topic": "/dex5_1/set_position",
            "message": "std_msgs/msg/Float64MultiArray",
            "representation": "20 absolute URDF joint positions in rad",
            "sdk_clamps_limits": False,
            "deployment_adapter_must_clamp": True,
        },
        "sources": {
            "isaac_env": str(ENV_SOURCE),
            "sim_urdf": {"path": str(SIM_URDF), "sha256": sha256(SIM_URDF)},
            "real_sdk_urdf": {"path": str(REAL_URDF), "sha256": sha256(REAL_URDF)},
            "sdk_controller": {"path": str(CONTROLLER_SOURCE), "sha256": sha256(CONTROLLER_SOURCE)},
        },
        "checks": {
            "joint_name_set": "PASS",
            "policy_sdk_order": "PASS",
            "sim_real_urdf_limits": "PASS" if not limit_mismatches else "FAIL",
            "direction_static": "PASS_WITH_HARDWARE_CONFIRMATION_REQUIRED",
            "hardware_direction_sweep": "NOT_MEASURED",
        },
        "joints": joint_rows,
    }

    latency = {
        "schema_version": "dex5.sim2real.control_latency.v1",
        "audit_phase": "Sim2Real-S0.0",
        "generated_at_utc": generated_at,
        "measurement_mode": "STATIC_SOURCE_AUDIT_ONLY",
        "hardware_connected": False,
        "status": "NOT_MEASURED",
        "known_timing": {
            "isaac_physics_hz": 120.0,
            "isaac_decimation": 2,
            "isaac_policy_hz": 60.0,
            "isaac_policy_period_ms": 16.6666666667,
            "real_sdk_command_republish_hz": 100.0,
            "real_sdk_command_republish_period_ms": 10.0,
            "real_state_publish_mode": "event-driven on DDS HandState reception",
            "real_state_rate_hz": None,
            "ros_command_to_first_dds_write": "immediate in callback; not measured",
        },
        "pd": {
            "isaac": {"kp": 8.0, "kd": 0.2, "effort_limit": 0.93, "velocity_limit_rad_s": 4.18},
            "real_position_command": {
                "index_middle_ring_pinky": {"kp": 0.1, "kd": 0.001},
                "thumb": {"kp": 1.0, "kd": 0.02},
            },
            "compatibility": "NOT_CALIBRATED",
            "note": "Backend gain units and actuator dynamics are not established as equivalent; compare closed-loop step responses, not numeric gains alone.",
        },
        "measurements": {
            "command_topic_hz": None,
            "joint_state_topic_hz": None,
            "command_to_applied_echo_ms": None,
            "command_to_joint_motion_onset_ms": None,
            "round_trip_p50_ms": None,
            "round_trip_p95_ms": None,
            "round_trip_p99_ms": None,
            "jitter_std_ms": None,
            "clock_alignment_verified": False,
        },
        "required_hardware_test": {
            "command": "bounded one-joint position step inside a guarded posture",
            "sample_count_min": 500,
            "report": ["state rate", "motion onset p50/p95/p99", "jitter", "packet gaps", "steady-state error"],
        },
    }

    table_lines = []
    for row in joint_rows:
        coupling = row["coupled_to"] or "-"
        table_lines.append(
            f"| {row['policy_index']} | `{row['name']}` | {row['finger']} | {row['physical_motor_index']} | "
            f"+1* | [{row['lower_rad']:.6f}, {row['upper_rad']:.6f}] | {coupling} |"
        )

    report = f"""# Dex5 Sim2Real-S0.0 Interface Spec

生成时间：`{generated_at}`  
审计方式：只读静态源码/资产审计；未启动 PPO、未生成 teacher、未修改 O0、未向真机发送命令。

## 结论

**总体状态：`PARTIAL_NOT_READY_FOR_HARDWARE_POLICY`。**

- `PASS`：Isaac 策略 20 维关节顺序与真实 SDK 20 维 ROS/DDS 关节顺序一致。
- `PASS`：仿真 URDF 与 SDK URDF 的 20 个关节名称、坐标轴和位置限位一致。
- `PASS_WITH_PENDING_TEST`：静态坐标映射为 `q_real = +1 * q_sim + 0`；仍需低速逐关节真机方向试验确认编码器/固件链路。
- `BLOCKED`：Isaac 以 20 个独立 DOF 建模，但真机四个 DIP 与对应 PIP 共用电机，实际只有 16 个物理执行器。
- `BLOCKED`：Isaac oracle 为 60 Hz 累积相对目标，SDK 接受 100 Hz 重发的绝对目标；部署适配器尚未实现。
- `BLOCKED`：PD 动态、真实状态频率、命令到运动延迟和抖动尚未在线测量。
- `BLOCKED`：当前 152/172 维 privileged oracle observation 含真机不可直接获得的信息，不能直接部署。

## 1. 关节顺序与方向

策略和 SDK 的统一顺序是 `index -> middle -> ring -> pinky -> thumb`，每指四维。原始 URDF 文档是 thumb-first，Isaac 通过关节名称查询后重排，因此任何代码都不得依赖 USD/URDF 内部下标。

| idx | joint | finger | motor | direction | limit rad | coupled to |
|---:|---|---|---:|---:|---|---|
{chr(10).join(table_lines)}

`*` `+1` 来自同名关节、相同轴和相同 URDF 坐标约定；不是已完成的真机运动方向测量。

四个共享关系必须在 policy action adapter 中强制处理：`Pitch_24R=Pitch_23R`、`Pitch_34R=Pitch_33R`、`Pitch_44R=Pitch_43R`、`Pitch_54R=Pitch_53R`。SDK 的直接 `/set_position` 路径没有限位 clamp，也没有在该回调中归一化共享关节，因此部署端必须先耦合再限位。

另一个容易出错的顺序：指尖接触/力数组为 `thumb,index,middle,ring,pinky`，与关节数组不同，必须按名字重排。

## 2. Limit 审计

位置上下限逐关节一致，结果为 `PASS`。但 Isaac actuator 将全部关节的速度上限覆盖为 `4.18 rad/s`、effort 上限覆盖为 `0.93`；原始 URDF 对非拇指关节给出了更低且不一致的速度值，部分 effort 为 0。S0 部署应采用更保守的真机实测速度/增量门，而不是沿用 Isaac override。

最低安全适配要求：检查长度和有限值，处理 PIP/DIP 耦合，按本文件限位 clamp，再施加每步增量和速度限制。当前 SDK 只验证数组长度，不负责这些 S0 policy 安全条件。

## 3. PD 与控制频率

| 项目 | Isaac | 真实 SDK | 判定 |
|---|---:|---:|---|
| physics/control | 120 / 60 Hz | 命令缓存重发 100 Hz | 需要 rate adapter |
| 非拇指 Kp/Kd | 8.0 / 0.2 | 0.1 / 0.001 | 未校准 |
| 拇指 Kp/Kd | 8.0 / 0.2 | 1.0 / 0.02 | 未校准 |
| action | 累积 `delta q`, scale 0.025 rad | 绝对 `q` rad | 需要 action adapter |

不能仅根据数值断言 PD 差 80 倍，因为 PhysX 与固件增益单位/执行器模型未证明等价。正确验收是同一关节小阶跃的上升时间、超调、稳态误差和延迟对齐。源码注释写“50 Hz”但实际成员固定初始化为 100 Hz，审计以可执行代码为准。

## 4. Observation 接口

当前 oracle observation 不可直接真机部署：它包含笔的真值位姿/线速度/角速度、指尖真值位置、三维接触力、滑移、质量、摩擦、COM 偏置和 canonical ID。

S0 student 的可用基础接口建议冻结为：

1. `q[20]`：`/dex5_1/joint_state.finger_positions`，rad；需验证反馈确实为 20 维。
2. `dq[20]`：同 topic 的 `finger_velocities`，rad/s；需做噪声和延迟审计。
3. `previous_target[20]`：本地保存的已 clamp 绝对目标，不能依赖异步 echo。
4. `fingertip_normal_force[5]`：优先 `/paxini/fingertip_forces`，按 `finger_names` 重排；需要同步和标定。
5. 可选视觉笔状态：当前没有已验证的相机标定/追踪接口，状态为 `UNAVAILABLE`。

`finger_forces` 当前 controller 填 0，不能作为力观测。`/dex5_1/applied_position` 只是命令回显，不是执行器已达到目标的证明。student schema、归一化、时间堆叠和缺帧策略仍需在 S0.1 冻结。

## 5. 延迟审计

静态代码只能确认 60 Hz 仿真决策周期和 100 Hz SDK 命令重发周期。真实 DDS 状态频率是事件驱动，命令回调到首个 DDS write 虽然在代码中立即发生，但 OS、ROS、DDS、USB/固件和机械响应延迟均未测量。因此 `control_latency_report.json` 的在线字段均为 `null/NOT_MEASURED`，不能用理论 10 ms 代替实测延迟。

## 6. 兼容性门

进入真机 policy dry-run 前必须完成：

- 逐关节小幅方向检查，确认全部 20 个逻辑关节符号；
- 16 电机耦合 action adapter，并证明不会发送冲突 PIP/DIP 目标；
- 绝对目标 clamp、增量限制、速度限制、watchdog 和急停；
- 真实 q/dq/force 的维数、名称、频率、时间戳和缺帧测试；
- 至少 500 次受限阶跃的 p50/p95/p99 延迟与 PD 响应报告；
- 定义可部署 student observation，禁止 privileged oracle 直接连接真机。

## 7. 审计输入

- Isaac env：`{ENV_SOURCE}`
- Oracle env：`{ORACLE_SOURCE}`
- 仿真 URDF SHA-256：`{sha256(SIM_URDF)}`
- SDK URDF SHA-256：`{sha256(REAL_URDF)}`
- SDK controller commit：`831f0a262a68b12bac47d8d410b018e8a95c4ae3`（工作区存在用户改动，本审计未修改）
- 冻结 O0 SHA-256：`73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`（未修改）
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "joint_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "control_latency_report.json").write_text(
        json.dumps(latency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "Dex5_Sim2Real_Interface_Spec.md").write_text(report, encoding="utf-8")
    print(f"generated {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
