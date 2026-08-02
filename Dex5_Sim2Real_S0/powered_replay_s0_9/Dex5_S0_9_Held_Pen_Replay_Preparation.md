# Dex5 S0.9 持笔轨迹回放准备

## 当前状态

```text
状态: INTERFACE_PASS_TASK_FAIL
真实持笔回放: 1/1 COMPLETED, PEN NOT DROPPED, NO VISIBLE ROTATION
PPO / student / teacher / online learning: NOT RUN
```

本阶段只准备确定性的映射轨迹。首次验收仅执行轨迹 0，一次执行一条，
每条结束后自动急停并由现场人员记录是否掉笔。

轨迹 0 已完成首次持笔验收：308 个控制点、3.178 s、命令速度峰值
`0.07962 rad/s`，现场确认笔未掉落。该结果只证明一条确定性轨迹在当前
摆放和当前硬件条件下可执行，但笔没有转动。该结果不是转笔任务成功，
不代表 20 条轨迹、重复性或策略部署通过。

前两次尝试都在轨迹记录前被单窗口反馈速度门中止。第二次触发通道为
耦合派生 DIP 关节 15，现场未观察到明显位移。根据现场验收决定，默认路径
改为直接发送轨迹起点；反馈速度门只审计 16 个独立执行器对应通道，并要求
连续三个新的 50 ms 窗口超限。命令速度、增量、watchdog 和急停保持不变。

## 固定输入

- 映射后轨迹：`mapped_replay_100hz.npz`
- 轨迹 SHA-256：`dfcc88f356fe63815b527bde4b38db0a948e3a7a8444fbcc2942989565725108`
- 拇指映射 SHA-256：`273bafb3631a50907a7738d57212e1b83df1184c91953fac9056a1f2a43c415b`
- P3 静态门 SHA-256：`46882c610578a3c40918af2a65404d026031f960af59ead0b9f3728f74adf95b`
- 轨迹数：20
- 控制频率：100 Hz
- 最大离线速度：0.0799954 rad/s
- 最大离线单步增量：0.000799954 rad

## 首次回放顺序

1. 启动 anemone 与 ROS 控制器，确认关节反馈正常。
2. 空手运行 S0.8.1 mapped-P3 静态定位工具，并通过已验收姿态门。
3. 工具急停后，在尽量不移动手指的情况下放入笔，软垫和物理急停就绪。
4. 启动 S0.9，使用默认 `low_gain_preload` 启动模式；交互提示出现后输入 `ARM`。
5. 轨迹结束自动急停，根据现场结果输入笔是否掉落。

## 操作命令

以下两条命令都要求 anemone 和 ROS 控制器已经在另外两个终端运行。
控制器必须使用显式的低增益 hybrid 参数启动：

```bash
ros2 launch unitree_dex5_1_control dex5_1_control.launch.py \
  enable_manus_glove:=false \
  enable_manus_force_hybrid:=false \
  command_timeout_sec:=0.15 \
  hybrid_kp_scale:=0.15 \
  hybrid_kd_scale:=0.30 \
  hybrid_thumb_kp_scale:=0.15 \
  hybrid_thumb_kd_scale:=0.30
```

`low_gain_preload` 会在 ARM 前通过 controller parameter service 读回并校验这四个值；
kp 任一通道高于 0.20 或 kd 高于 0.40 都会拒绝使能。

新启动路径必须先做空手 startup-only 验收，提示出现后输入 `STARTUP`：

```bash
/usr/bin/python3 \
  Dex5_Sim2Real_S0/powered_replay_s0_9/dex5_s0_9_powered_replay.py \
  --execute \
  --startup-only \
  --startup-mode low_gain_preload \
  --static-gate Dex5_Sim2Real_S0/powered_replay_s0_9/mapped_p3_runtime_gate_retry.json \
  --trajectory-start 1 \
  --count 1 \
  --confirm-soft-pad YES \
  --confirm-empty-hand YES \
  --arm-token DEX5-S0.9-STARTUP \
  --output-json Dex5_Sim2Real_S0/powered_replay_s0_9/trajectory1_startup_gate.json \
  --output-report Dex5_Sim2Real_S0/powered_replay_s0_9/trajectory1_startup_gate.md
```

只有输出 `STARTUP_GATE_PASS` 后，才能放笔并执行下方的 powered replay。

先空手复现 mapped P3；出现提示后输入 `P3`：

```bash
cd "/home/linkerhand/桌面/deep learning/dex5_penspin"
source /opt/ros/jazzy/setup.bash
source /home/linkerhand/ofdemo_ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

/usr/bin/python3 \
  Dex5_Sim2Real_S0/hardware_calibration_s0_8_1/dex5_thumb_global_calibrate.py \
  --execute \
  --stage-mapped-p3 \
  Dex5_Sim2Real_S0/hardware_calibration_s0_8_1/thumb_global_mapping_all.json \
  --thumb-command-override 0.1980349710,0.7970652983,0.9899153359,0.6662725608 \
  --confirm-empty-hand YES \
  --arm-token DEX5-S0.8.1-P3 \
  --output Dex5_Sim2Real_S0/powered_replay_s0_9/mapped_p3_runtime_gate.json \
  --report Dex5_Sim2Real_S0/powered_replay_s0_9/mapped_p3_runtime_gate.md
```

确认输出为 `PASS_THUMB_P3_STATIC_GATE` 后放入笔，再执行轨迹 0；出现提示后输入
`ARM`：

```bash
/usr/bin/python3 \
  Dex5_Sim2Real_S0/powered_replay_s0_9/dex5_s0_9_powered_replay.py \
  --execute \
  --startup-mode low_gain_preload \
  --static-gate Dex5_Sim2Real_S0/powered_replay_s0_9/mapped_p3_runtime_gate.json \
  --trajectory-start 0 \
  --count 1 \
  --confirm-soft-pad YES \
  --confirm-pen-placed YES \
  --arm-token DEX5-S0.9-ARM
```

不要再使用旧的 `--stage-only` 路径；它混用了命令与原始反馈坐标，现已禁用。

## 执行门

- 当前反馈必须与已验收 P3 原始反馈逐关节一致，最大偏差不超过 0.12 rad。
- 重新使能时先通过 `/dex5_1/set_hybrid` 使用 controller 的低增益接住当前姿态，以 0.01 rad/s 对齐 P3，连续 3 个 50 ms 窗口低于 0.04 rad/s 后才切入正常位置控制。
- 低增益预加载和正常增益切换均使用单个新鲜 50 ms 窗口的 0.12 rad/s 硬中止门。
- 反馈 watchdog 为 0.1 s。
- 命令速度不超过 0.08 rad/s，单步增量不超过 0.01 rad。
- 16 个独立执行器对应反馈连续三个 50 ms 窗口超过 0.12 rad/s 时中止。
- 派生 DIP `3/7/11/15` 不参与反馈速度硬门，但仍完整记录。
- 75 C 为温度硬中止门；不设置 60 C 自动失能锁存。
- 首次回放只使用拇指映射的 `up` 分支。负载下 reverse branch 尚未验收。
- `legacy_direct` 和 `legacy_soft_resume` 仅保留用于历史结果复现，不得用于新的 powered replay。

## 指标边界

原始 SDK 反馈与映射后的命令不处于完全相同的标定坐标中，尤其是拇指和
PIP/DIP 耦合关节。因此报告中的 `raw cmd-feedback` 仅作诊断，不能直接称为
物理关节跟踪误差。本次正式门只依赖姿态复现、反馈连续性、速度、温度和掉笔记录。
