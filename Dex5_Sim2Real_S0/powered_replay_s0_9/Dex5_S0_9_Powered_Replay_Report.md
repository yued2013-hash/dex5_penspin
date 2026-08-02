# Dex5 S0.9 Powered Replay Report

## 结论

```text
状态: INTERFACE_PASS_TASK_FAIL
真实持笔回放: 1/1 completed
掉笔: NO
可见转动: NO
autonomous policy / PPO / student / online learning: NOT RUN
```

轨迹 0 在 mapped P3 持笔入口完成 308 个 100 Hz 控制点，执行约 3.178 s。
命令速度峰值为 0.07962 rad/s，低于本次 0.08 rad/s 命令门；单步最大命令
增量为 0.0007962 rad。轨迹结束后发送软件急停，现场确认笔未掉落，
但笔没有发生可见转动。因此只能判定命令接口回放通过，转笔任务失败。

## 失败定位

- 原仿真 episode 为 46 个 60 Hz 步，约 0.77 s；真机速度安全包扩展为 308 个 100 Hz 点，约 3.08 s。动态动作被放慢约 4 倍，不能假设仍保留转动惯性。
- 仿真成功判据积分 `pen_angvel[:, 2]` 得到约 38.4 度，刚好越过 45+/-10 度的下界。
- 同一仿真轨迹的笔姿态四元数净变化只有约 7.7 度，逐帧姿态变化累计约 20.7 度，与积分成功指标不一致。
- 真机没有物体姿态观测，旧回放器只检查命令完成和掉笔，因此错误地产生了 `PASS`。

在解决仿真旋转指标一致性和动态时间缩放之前，冻结其余 19 条持笔回放。

## 固定资产

- replay SHA-256: `dfcc88f356fe63815b527bde4b38db0a948e3a7a8444fbcc2942989565725108`
- thumb mapping SHA-256: `273bafb3631a50907a7738d57212e1b83df1184c91953fac9056a1f2a43c415b`
- runtime P3 gate SHA-256: `210804f0b442c05a65caa6e99c80dc5eabae2a85bb72cfeee47c3277a6e5c12a`
- result SHA-256: `6f568695d1cb8bde37ec6d819b99cc07cd51b2f2def4047135304470f6d9eccb`

## 实测结果

| 指标 | 结果 |
|---|---:|
| command rows | 308 |
| duration | 3.1775 s |
| command max speed | 0.07962 rad/s |
| command max delta | 0.0007962 rad |
| 50 ms position-difference peak | 0.42734 rad/s |
| touch samples | 308 |
| touch active ratio | 28.88% |
| force samples | 0 |
| temperature events | 0 |
| dropped object | false |

## 解释边界

- `reported_velocity_max_rad_s=242.98` 是异常的原始 SDK `dq`，与位置差分不一致，不能解释为真实关节速度。
- `raw command-feedback difference` 混合了拇指非线性标定和 PIP/DIP 耦合，不是物理跟踪误差。
- 没有收到法向力样本，不能声称完成夹持力验收。
- 没有温度事件只表示 controller 未报告温度，不等于已测得低温。
- 当前仅为接口完成，不是任务 PASS；不覆盖重复性、其他 19 条轨迹或自主策略。
