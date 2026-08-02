# O1.1 Stability and Gait Disambiguation 验收报告

## 结论

**O1.1 诊断与 exploit guard：PASS。当前 oracle 策略稳定性：NOT PASS。**

当前 best checkpoint 能产生转动和原始接触切换，但尚未产生合格 finger gait。旧的 raw gait 和 180 deg 指标被大量 ballistic、接触丢失和错轴旋转污染，因此当前 checkpoint 只能归类为 `REACHABILITY_ONLY`。

## 确定性对照

同一 checkpoint、seed `20260730`、256 env、240 control steps：

| 指标 | Baseline | Ballistic guard | 解读 |
|---|---:|---:|---|
| 完成 episode | 559 | 569 | guard 使失败 episode 更早 reset |
| raw gait | 29.16% | 28.47% | 不能当作稳定 gait |
| qualified gait | 0.00% | 0.00% | 策略未学会合格 finger gait |
| raw >180 deg | 72.99% | 73.11% | 旋转量本身不足以证明成功 |
| qualified >180 deg | 0.00% | 0.00% | 无稳定、正确轴向的 180 deg episode |
| ballistic episode | 46.51% | 45.17% | 策略倾向未被改变 |
| 平均 ballistic 暴露 | 1.236 steps | 0.452 steps | **下降 63.46%** |
| contact-loss termination | 36.49% | 1.05% | **下降 97.11%** |
| 总失败率 | 60.64% | 61.51% | guard 是拒绝失败，不是美化成功率 |
| 目标轴角速度 | 2.422 rad/s | 2.426 rad/s | 基本不变 |
| 错轴角速度 | 5.551 rad/s | 5.546 rad/s | 仍为目标轴的 2.29 倍 |

Ballistic guard 在连续 2 个无接触控制步，且线速度 `>0.20 m/s` 或角速度 `>12 rad/s` 时立即终止。它显著缩短了仿真中的投机飞行，但没有重训策略，所以不能解读为策略质量提升。

## Canonical 失败画像

| 入口 | Baseline 主要问题 | 判定 |
|---|---|---|
| P1 | ballistic 50.99%，contact loss 48.02%，axis alignment 0.367 | 转动伴随接触崩溃 |
| P2 | ballistic 100%，contact loss 69.67%，failure 99.18% | 最明显的 exploit 入口 |
| P3 | failure 8%，无 ballistic，但 axis alignment 仅 0.437 | 接触较稳，转轴不合格 |
| P4 | failure 100%，ballistic 31.78%，penetration 25.23% | 制动入口未被当前统一策略维持 |
| P5 | failure 3.57%，但 axis alignment -0.070 | 能保持，几乎没有正向转动 |

P1/P2 中的 raw gait 主要来自 thumb/index/middle 断开后的偶发重接，而不是保持笔稳定的换指。P3 保持相对稳定，但没有任何合格 gait；P4 则是当前统一策略的另一个严重缺口。

## 完成内容

- episode-level 三步 debounce 的逐指 break/re-contact/gait 指标。
- 目标轴 directional alignment 与 absolute alignment。
- contact loss、ballistic、penetration、throw、drop、drift、invalid 的独立与互斥 primary failure 分类。
- 分 P1-P5 的 deterministic JSON/Markdown 评估。
- 可选 ballistic termination guard 与暴露时长指标。

## 边界与下一步

本阶段没有修改 O0 bank、没有生成 teacher、没有进行 PPO 训练。O0 bank SHA-256 仍为：

`73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`

下一训练迭代应把 ballistic guard 作为安全终止，同时针对错轴旋转和 P1/P2 接触崩溃调整学习目标。在新策略的 qualified gait `>0` 之前，不应扩大到多 seed 或 5090/H200。

## 证据

- `o1_1_best_agent_baseline_256env_240steps.json`
- `o1_1_best_agent_baseline_256env_240steps.md`
- `o1_1_best_agent_ballistic_guard_256env_240steps.json`
- `o1_1_best_agent_ballistic_guard_256env_240steps.md`
- `O1_1_Stability_and_Gait_Disambiguation_验收结果.json`
