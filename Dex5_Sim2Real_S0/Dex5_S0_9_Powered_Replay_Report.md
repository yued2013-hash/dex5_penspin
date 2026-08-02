# Dex5 S0.9 Powered Replay Acceptance

## 结论

```text
状态: BLOCKED_GLOBAL_THUMB_MAPPING
执行轨迹: 0/1
autonomous policy: NOT RUN
PPO/student/online learning: NOT RUN
```

## 固定安全门

- S0.8 replay SHA-256: `60a97c8d7a1aa017ecf80230991b954d251c8d80c524ede6c76917a7bd26dbc8`
- command frequency: `100 Hz`
- command speed cap: `0.08 rad/s`
- command delta cap: `0.01 rad`
- feedback watchdog: `0.1 s`
- emergency stop: 每条轨迹结束及异常路径均发送
- thermal monitor: `/rosout` controller temperature event，`75.0 C` abort

## 逐轨迹结果

| ID | 状态 | mean error rad | p95 error rad | raw SDK dq max | 50ms position-diff max | max temp C | dropped |
|---:|---|---:|---:|---:|---:|---:|---|
| - | NOT RUN | - | - | - | - | - | - |

## P3 入口真机门失败

2026-08-01 的受控测试先后完成了通信恢复、100 Hz 反馈检查和两次安全摆位中止。
随后现场人员明确选择直接发送轨迹 0 的 P3 起点绝对位置。控制器正确接收了目标，
但拇指全局反馈与目标严重不一致：

| joint | P3 target rad | hardware feedback rad | URDF range status |
|---:|---:|---:|---|
| 16 | +0.4891 | -0.5868 | lower boundary |
| 17 | +0.6560 | -5.3494 | invalid |
| 18 | +0.9171 | +5.4998 | invalid |
| 19 | +0.4503 | -4.7830 | invalid |

最大 P3 起点误差为 `6.0054 rad`。异常连续出现在多帧反馈及 controller
`FB(act/des)` 日志中，不是单帧 ROS 读取噪声。检测后已发送
`/dex5_1/emergency_stop=true`，controller 确认 `Motors disabled`，未执行任何 replay。

S0.8 只在初始姿态附近用 `0.05 rad` 小扰动辨识拇指通道，报告也明确声明其为
局部模型、未覆盖完整关节范围。因此这次结果判定为 **全局拇指映射/角度环绕缺口**，
不能解释为轨迹策略失败或掉笔失败。

进入 powered replay 前必须补齐：

1. 空手、逐拇指通道的分段全范围 command/feedback 扫描；
2. 角度环绕、符号、偏置和通道顺序辨识；
3. P3 目标经过新映射后的静态到位门，20 维反馈全部位于 URDF 范围；
4. 最大起点误差重新达到 `<=0.12 rad`。

## 边界

安全超速判据使用 S0.7 已验收的 50 ms 位置差分速度；原始 SDK `dq` 仅作诊断。
温度只有在 vendor controller 产生温度日志时可观测；`None` 不代表低温。
掉笔由现场人员逐轨迹人工标注。motor response 使用公共 20D joint feedback
和 `/dex5_1/applied_position` 记录，不解释为电机轴测量。
