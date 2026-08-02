# O1.6 Support-Continuity Objective Ablation 验收报告

## 结论

```text
O1.6 实验管线：PASS
Support-continuity skill：FAIL
Support-margin / recoverability v1 objective：FAIL
失败类型：DENSE_PENALTY_TERMINATION_EXPLOIT
```

O1.6 成功完成了同预算对照和反投机审计，但 B/C 并没有学会承力交接。策略发现了一个更容易的回报漏洞：尽快触发接触丢失或 ballistic 终止，从而避免继续累积逐步负奖励。

这个结果否定的是“在当前 termination 语义下持续施加逐步负奖励”这一实现，**不是**否定 support continuity 假设，也不证明承力交接在动力学上不可达。

## 同预算契约

| 项目 | A Temporal baseline | B Support margin | C Recoverability |
|---|---:|---:|---:|
| Reset | P3-only | P3-only | P3-only |
| Observation | 172D temporal | 172D temporal | 172D temporal |
| Seed | 20260730 | 20260730 | 20260730 |
| Envs | 256 | 256 | 256 |
| PPO iterations | 60 | 60 | 60 |
| Training transitions | 368,640 | 368,640 | 368,640 |
| Evaluation transitions | 61,440 | 61,440 | 61,440 |
| O0 bank hash | 一致 | 一致 | 一致 |

冻结 O0 SHA-256：

```text
73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65
```

B 只加入无手指身份的支撑余量、支撑下降和过大总力项；C 在 B 上增加首次下降 5 mm 后有限 12 步窗口内的高度缺口、向下速度和恢复进展项。两者都没有规定 thumb/index/middle/ring 的接触顺序。

## 量化结果

| 指标 | A Baseline | B Support margin | C Recoverability |
|---|---:|---:|---:|
| 完成 episodes | 986 | 6039 | 6224 |
| 估算平均 episode 步数 | 62.31 | 10.17 | 9.87 |
| Raw gait | 93.71% | 0.00% | 0.00% |
| Qualified gait | 0.00% | 0.00% | 0.00% |
| Support handoff | 0.00% | 0.00% | 0.00% |
| Height recovery | 0.00% | 0.00% | 0.00% |
| TIM overlap | 50.08% | 80.05% | 79.62% |
| Failure | 100.00% | 100.00% | 100.00% |
| Contact loss | 0.00% | 98.94% | 98.09% |
| Ballistic | 3.75% | 99.37% | 99.37% |
| 平均支撑力 Z | +0.0142 N | -0.0745 N | -0.0712 N |
| 平均总接触力 | 1.452 N | 2.589 N | 2.505 N |
| 峰值总接触力 | 8.015 N | 8.744 N | 8.766 N |
| 平均机械功率 | 0.0997 W | 0.0745 W | 0.0707 W |
| 最大穿透 | 0.339 mm | 0.977 mm | 0.888 mm |

## 失败机制

1. B/C 的 episode 长度只有 A 的约 16%，与完成 episode 数大幅增加同时出现。
2. B/C 的 contact-loss 和 ballistic 率同时接近 99%，证明策略在主动缩短轨迹，而不是尝试完成长时间承力迁移。
3. B/C 的 TIM overlap 约 80% 是截断偏差：短 episode 主要保留了 P3 初始的 TIM 重叠帧，不表示出现了更好的 handoff。
4. 峰值总接触力仅从 8.015 N 增至约 8.75 N，未出现明显的“无限加力猛夹”投机；主要漏洞是提前终止。

因此，B/C checkpoint 只能用于失败诊断，不能作为 oracle skill、teacher 或蒸馏源。

## 阶段判定

O1.6 已完成它的诊断目标：

- O0 未修改；
- P3-only 与 temporal observation 保持一致；
- 无 teacher，无接触顺序监督；
- A/B/C 同 seed、同 env 数、同 PPO 预算；
- qualified gait、handoff、height recovery 与反投机指标已配对评估。

但技能验收不通过：

```text
qualified gait = 0
support handoff = 0
height recovery = 0
termination exploit = true
```

## 下一步

不增加 PPO 步数，不使用 5090/H200 扩大训练。下一个小阶段应定义为 `O1.7 Termination-Invariant Support Credit`：

1. 用固定时域、absorbing failure state 或“剩余时域 terminal cost”消除提前死亡的回报优势。
2. 在训练前比较 zero-action、random policy 和 intentional-failure policy 的 episodic return。
3. 只有在“立即失败的回报严格低于继续保持有效动力学”通过后，才允许再跑同预算 PPO。
4. 继续使用 qualified gait + support handoff + height recovery 联合门，并报告 episode 长度分布，避免 TIM overlap 等指标再受截断偏差影响。

## 产物

- `o1_6_A_temporal_baseline_eval256.json/.md`
- `o1_6_B_support_margin_eval256.json/.md`
- `o1_6_C_recoverability_eval256.json/.md`
- `O1_6_Support_Continuity_对比结果.json`
- `O1_6_Support_Continuity_对比图.png`

## 复验记录

- 变更源码 `py_compile`：通过。
- 项目测试：`7 passed`。
- A/B/C 评估与对比 JSON：全部可解析。
- A/B/C `agent.yaml`：除 experiment name 外一致。
- B/C 环境差异：仅限本次声明的 support/recovery objective 参数。
- 冻结 O0 bank SHA-256：未变。
