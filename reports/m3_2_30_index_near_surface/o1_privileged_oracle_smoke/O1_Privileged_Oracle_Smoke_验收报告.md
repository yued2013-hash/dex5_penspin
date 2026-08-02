# O1 Privileged Oracle PPO Smoke 验收报告

**结论：`PASS_REACHABILITY_ONLY`**

O1 训练管线已在本机 RTX 5060 Laptop 8GB 上打通，策略已产生目标轴旋转和 debounced contact switching；但失联率和错误轴旋转仍高，不能宣称已学会可靠 finger gait。

## 环境合同

- Task: `Dex5-PenSpin-PrivilegedOracle-Direct-v0`
- Frozen reset bank: `P1_P5_v9_physx_projected_R3_validated.npz`
- Bank SHA-256: `73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`
- Reset split: `train`
- Reset mixture P1/P2/P3/P4/P5: `40/20/20/15/5`
- Observation: 152 维 privileged state
- Action: 20 维增量 joint-position target
- Gait rule: 单指 release 后再 re-contact，两次变化均需连续稳定至少 3 个 control steps
- Reward 不包含 P1-P5 pose tracking 或指定接触序列奖励

## 管线验证

| 检查 | 结果 |
|---|---:|
| 16 env 物理 smoke | PASS |
| Observation / action | `152 / 20` |
| NaN/Inf | 0 |
| 关节目标实际最大移动 | `0.174 rad` |
| 64 env PPO，2 updates | PASS |
| 256 env PPO，2 updates | PASS |
| 256 env PPO，30 updates | PASS |
| 30-update transitions | `184,320` |
| 30-update 训练时间 | `23.17 s` |
| 实测显存占用 | 约 `3.3 GB` |
| GPU 利用率抽样 | 约 `70-77%` |

30-update TensorBoard 中，total reward mean 从约 `-7.42` 上升到 `13.99`，目标轴角速度从约 `1.60` 上升到 `2.24 rad/s`。这只证明可达性，不是技能验收。

## 确定性评估

Checkpoint: `best_agent.pt`

- SHA-256: `10998862bdd0aa5a85f7491300be3056b09f546cb0ff63401d0035401129420b`
- 256 env，240 control steps
- 完整 episodes: `559`

| 指标 | 结果 |
|---|---:|
| 目标轴平均角速度 | `2.422 rad/s` |
| 错误轴平均角速度 | `5.551 rad/s` |
| 笔平均线速度 | `0.0650 m/s` |
| 平均接触指数 | `1.918` |
| 平均动作绝对值 | `0.0952` |
| 转过 180 deg | `408/559 = 73.0%` |
| debounced gait event | `163/559 = 29.2%` |
| 失败 | `339/559 = 60.6%` |
| 接触失联 | `204/559 = 36.5%` |
| 穿透终止 | `33/559 = 5.9%` |
| 抛笔终止 | `12/559 = 2.1%` |
| 最大运行穿透 | `0.999 mm` |

实际 reset 累计为 P1/P2/P3/P4/P5 = `318/180/147/128/42`，与预设混合比例基本一致。

## 判定

已经证明：

1. 本机可稳定执行 256 env privileged PPO smoke。
2. O0 reset bank、privileged observation、动作、reward、SKRL 和 checkpoint 管线已连通。
3. 短训可产生旋转和持续 contact transition 信号。

尚未证明：

1. contact transition 是可靠 finger gait，而非失稳前的接触切换。
2. 策略能在低失联率下连续旋转。
3. 当前 reward 已抑制错误轴旋转、滚滑和局部抛笔。

因此下一门应是 `O1.1 Stability and Gait Disambiguation`，先降低失联和错误轴旋转，再决定是否扩大训练。当前不需要 5090/H200，也不应直接延长 PPO 训练。
