# Dex5 S0.8 Real Calibration Closure

## 结论

```text
状态: PASS_LOCAL_MODEL_CLOSURE

允许:
- 使用实测 S0.8 配置进行 hardware-compatible PhysX 审计
- 使用限速 replay 包做离线检查

仍禁止:
- PPO 或 student 训练
- 修改 O0
- powered autonomous policy
- 将本报告解释为已完成带电 replay 验收
```

S0.8 已完成 Dex5 公共 20D URDF 指令接口到 16 个物理执行通道的局部小信号辨识。该模型适用于当前验收姿态附近，不是电机轴模型，也没有覆盖完整关节范围。

## 受控测试

- 主扫描：16 通道，慢速对称 `0.02 rad` 映射测试和 `0.008 rad` 动态小阶跃。
- 拇指补充扫描：仅执行通道 12-15，`0.05 rad`、`0.05 rad/s`，用于跨过 vendor 转换的小命令死区。
- 每个通道单独执行并回到局部基线。
- 控制器 watchdog：`0.15 s`。
- 两轮结束均发送软件急停，未执行 policy、teacher、student 或 replay。

## 方向与映射

| 项目 | 结果 | 状态 |
|---|---:|---|
| joint direction | 16/16 | PASS |
| actuator mapping | 16/16 | PASS |
| 方向符号 | 全部 `+1` | PASS |
| SDK 指令维度 | 20D absolute URDF target | CONFIRMED |
| 物理执行通道 | 16 | CONFIRMED |

四个共享执行通道的 PIP/DIP 实测局部响应比：

| 通道 | PIP/DIP | 响应比 `DIP/PIP` |
|---:|---|---:|
| 2 | Pitch_23R / Pitch_24R | 0.2946 |
| 5 | Pitch_33R / Pitch_34R | 0.3064 |
| 8 | Pitch_43R / Pitch_44R | 0.2875 |
| 11 | Pitch_53R / Pitch_54R | 0.2908 |

平均响应比为 `0.2948`，范围为 `0.2875-0.3064`。真实反馈不是旧仿真假设中的 PIP/DIP 等角运动；新适配器在保持公共命令对耦合的同时，对 PhysX 目标应用实测响应尺度。

## 局部动态模型

16 个执行通道的主反馈局部增益依次为：

```text
0.6811, 0.8465, 0.1587, 0.7199,
0.9183, 0.1525, 0.8370, 0.9024,
0.1431, 0.6450, 0.8150, 0.1850,
0.1780, 0.1702, 0.0957, 0.0441
```

完整 `20 x 16` 响应矩阵、逐通道原始响应和耦合比保存在 `hardware_calibration_s0_8/motor_dynamic_mapping.json`。

## 延迟

| 指标 | p05 | p50 | p95 | max |
|---|---:|---:|---:|---:|
| ROS controller 接收 | 0.228 ms | 0.312 ms | 0.601 ms | 0.777 ms |
| 可检测物理起动 | 14.44 ms | 19.90 ms | 29.19 ms | 31.14 ms |
| 达到本次响应 63% | 7.36 ms | 38.61 ms | 239.04 ms | 321.30 ms |

`0.008 rad` 小阶跃中，只有 6/16 通道超过 `0.5 mrad` 物理起动门，因此物理起动分布是可检测通道统计，不是完整 16 通道延迟上界。S0.8 PhysX 配置采用保守 `1-4` 个 100 Hz command tick，并保留这一测量边界。

## PhysX Adapter

新增 opt-in 实测模型：

- 旧 S0/O0/O1 行为默认不变；
- 公共 20D 命令先投影到 16 通道命令流形；
- 延迟后应用逐关节实测局部响应尺度；
- PIP/DIP 反馈按实测约 `0.29` 比例响应；
- 安全速度随机化范围为 `0.04-0.08 rad/s`，它不是电机最大速度。

离线适配器验收：

| 指标 | 结果 |
|---|---:|
| 随机样本 | 100000 |
| 20D→16D→20D 命令流形最大误差 | 0 rad |
| 实测耦合比复现最大误差 | 0 |
| held-out 真机模型误差 | 尚未测量 |

16 环境、30 控制步的实测 PhysX wrapper 回归：

| 指标 | 零动作 | 小动作 |
|---|---:|---:|
| finite | PASS | PASS |
| done | 0 | 0 |
| 至少 2 个 TIM 接触 | 100% | 100% |
| 三个 TIM 全接触 | 93.13% | 93.96% |
| drop/contact-loss/ballistic | 0/0/0 | 0/0/0 |
| 最大高度下降 | 0.538 mm | 0.538 mm |
| 最大穿透 | 0.350 mm | 0.350 mm |
| target limit violation | 0 | 0 |

两种模式最大笔位置漂移约 `4.7 mm`，因此该结果只证明 wrapper 没有破坏短时接触和数值稳定性，不等于实测模型已经完成全状态 sim→real 对齐。

## Replay 限速

原 20 条 replay 没有被覆盖。新生成的离线包：

```text
command frequency: 100 Hz
command rows:       7891
speed gate:         0.08 rad/s
observed max speed: 0.07998 rad/s
powered replay:     NOT RUN
```

该包只通过插值延长时间，不改变每条轨迹端点。任何带电 replay 仍需单独、人工监督的后续验收。

## 温度记录

测试期间控制器记录过 `61-63 C` 温度警告。按当前项目决定，没有增加 `60 C` 自动失能锁存；两轮测试结束均由软件急停关闭电机。后续带电测试前应留出冷却时间并继续记录温度。

## 产物

- `hardware_calibration_s0_8/raw_calibration.json`
- `hardware_calibration_s0_8/thumb_supplement.json`
- `hardware_calibration_s0_8/joint_direction_table.json`
- `hardware_calibration_s0_8/motor_dynamic_mapping.json`
- `hardware_calibration_s0_8/command_latency_report.json`
- `hardware_calibration_s0_8/hardware_simulation_config.json`
- `hardware_calibration_s0_8/adapter_validation.json`
- `hardware_calibration_s0_8/open_loop_commands_100hz_speed_safe.npz`
- `hardware_calibration_s0_8/safe_replay_manifest.json`
- `hardware_calibration_s0_8/S0_8_acceptance.json`
- `hardware_calibration_s0_8/physx_zero_audit.json`
- `hardware_calibration_s0_8/physx_small_audit.json`

## 验收判定

| 验收项 | 判定 |
|---|---|
| sim→real joint direction 一致 | PASS, 16/16 |
| 16 actuator mapping 一致 | PASS, 16/16 |
| PIP/DIP coupling 更新 | PASS, local model |
| command latency 记录 | PASS, 带可检测性边界 |
| hardware-compatible adapter 更新 | PASS, opt-in |
| adapter 离线误差 | PASS |
| replay 速度安全缩放 | PASS, offline only |

因此 S0.8 在“真实执行器局部模型闭环”口径下通过，但不授予 powered replay 或 autonomous policy 权限。
