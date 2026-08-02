# Dex5 S0.9 Powered Replay Acceptance

## 结论

```text
状态: OFFLINE_PREFLIGHT_PASS
执行轨迹: 0/1
autonomous policy: NOT RUN
PPO/student/online learning: NOT RUN
```

## 固定安全门

- mapped S0.9 replay SHA-256: `dfcc88f356fe63815b527bde4b38db0a948e3a7a8444fbcc2942989565725108`
- thumb mapping SHA-256: `273bafb3631a50907a7738d57212e1b83df1184c91953fac9056a1f2a43c415b`
- accepted P3 static gate SHA-256: `0109ecac16e22a9fd2c0395eb18fed49ae30e7d11493cb6867ddf7c3efaaf46c`
- command frequency: `100 Hz`
- command speed cap: `0.08 rad/s`
- command delta cap: `0.01 rad`
- startup mode: `low_gain_preload`
- feedback watchdog: `0.1 s`
- emergency stop: 每条轨迹结束及异常路径均发送
- thermal monitor: `/rosout` controller temperature event，`75.0 C` abort

## 逐轨迹结果

| ID | 状态 | raw cmd-feedback mean | raw cmd-feedback p95 | raw SDK dq max | 50ms position-diff max | max temp C | dropped | rotated |
|---:|---|---:|---:|---:|---:|---:|---|---|
| - | NOT RUN | - | - | - | - | - | - | - |

## 边界

安全超速判据使用 S0.7 已验收的 50 ms 位置差分速度；原始 SDK `dq` 仅作诊断。
温度只有在 vendor controller 产生温度日志时可观测；`None` 不代表低温。
掉笔由现场人员逐轨迹人工标注。`raw cmd-feedback` 只记录命令坐标与原始
SDK 反馈的数值差；由于拇指映射和 PIP/DIP 耦合，它不解释为物理关节跟踪误差。
motor response 使用公共 20D joint feedback 和 `/dex5_1/applied_position` 记录，
不解释为电机轴测量。
