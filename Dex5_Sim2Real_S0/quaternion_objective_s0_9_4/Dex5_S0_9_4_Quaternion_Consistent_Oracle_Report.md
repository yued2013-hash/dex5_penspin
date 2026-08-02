# Dex5 S0.9.4 Quaternion-Consistent Oracle Objective Report

## 结论

状态：`PASS_FIRST_QUATERNION_TRAJECTORY_GATE`

S0.9.4 已取得实际目标轴四元数旋转 `>=30 deg`、轴对齐
`>=0.70`、并满足完整物理门连续 `150 ms` 的轨迹。可以停止小预算
smoke；在进入大规模采集前，仍需单独关闭 rollout 运动学一致性数据门。

## 目标修正

- rotation phase、progress、target error、overshoot、braking 和 success
  均改为相对 `root_quat_w` 的有限步四元数运动。
- `root_ang_vel_w` 只保留为接触/冲击诊断，不作为任务进度或成功证据。
- 35 度审计任务的实际成功窗口为 `30-40 deg`。
- strict hold 要求角度、轴对齐、速度、高度、TIM 接触、ring/pinky clear
  和无当前物理失败连续满足 9 个 60 Hz 控制步，即 150 ms。
- 历史接触变化继续记录，但不永久否决后来已恢复的严格 hold 窗口。
- 增加低权重 `strict_hold_progress`，仅对现有严格门的连续计数提供时间
  信用；没有增加手指顺序或 contact-pattern reward。

## 训练预算

```text
task: Dex5-S0-ShortRotation-35Deg-Long-Direct-v0
envs: 256
seed: 20260804
initial: 30 iterations
continuation: 30 iterations, same seed and checkpoint
total: 60 iterations
```

没有增加第二个 seed，没有扩大环境数，没有使用 teacher/student，也没有
修改 O0。

## 正式评估

使用固定评估 seed `20260814`，共 512 个 episode：

| 指标 | 结果 |
|---|---:|
| 环境内部 corrected success | 369/512 = 72.07% |
| 离线同步状态严格重建成功 | 362/512 = 70.70% |
| 实际最大进度 >=30 deg | 433/512 = 84.57% |
| 任意严格门 hold | 429/512 = 83.79% |
| 最长严格 hold | 9 steps = 150 ms |
| 中位最大目标轴旋转 | 33.60 deg |
| P95 最大目标轴旋转 | 37.80 deg |
| 最大观测旋转 | 48.86 deg |

正式结论保守采用可从同步数据独立重建的 `362/512`。环境与离线审计相差
7 条，是 collector 在 success termination 边界没有保存环境内部最后一次
hold 计数状态；这不会影响“至少一条”的阶段门，但后续采集器应补齐 terminal
state 再作为训练数据管线使用。

一个明确通过的审计样本是 episode 22：最大实际目标轴旋转 `39.99 deg`，
最大进度处轴对齐 `0.847`，严格门连续保持 9 步。

## 数据边界

当前 512-episode rollout 的 `dataset_authorized=false`。原因不是任务未成功，
而是原始 PhysX 瞬时角速度与四元数有限差分仍未达到数据一致性门：

```text
60 Hz omega-z correlation: 0.749
60 Hz omega RMSE: 1.885 rad/s
quaternion/reported angular path ratio: 0.305
```

因此：

- checkpoint 可作为 S0.9.4 技能证据；
- `pen_pose` 与 `pen_angular_velocity_pose_delta` 可用于本阶段离线审计；
- 当前整包不得直接进入 student、powered replay 或大规模 teacher 数据集；
- 下一步应先修复 terminal-state 记录并冻结“pose-delta velocity 为运动通道”的
  数据合同，再进行受控扩采。

## 产物与哈希

- checkpoint: `agent_720.pt`
  - SHA256: `f3528f0c997c6e55f61269f1f8139288dc5c81161bc8163af1af2437f14f720d`
- evaluation: `continued_agent_720_eval_512.npz`
  - SHA256: `460b16432282569c9227873083ea858697c651ff0220243f86109398d6bee922`
- collector report: `continued_agent_720_eval_512.report.json`
  - SHA256: `5d64f2709e5175d8ccc8d8317863f1a2761192f9544e2a09f38719dff4c43817`
- independent gate audit: `continued_agent_720_gate_audit.json`
  - SHA256: `6babb58b22cbd7c3a2a19ec2a5991a6c25cc811a3348bd89291936a3be15904d`

