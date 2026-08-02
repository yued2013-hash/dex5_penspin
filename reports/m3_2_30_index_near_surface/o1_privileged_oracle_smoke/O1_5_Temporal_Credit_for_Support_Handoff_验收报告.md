# O1.5 Temporal Credit for Support Handoff 验收报告

## 结论

```text
O1.5 对照实验：PASS
Temporal observation 管线：PASS
Support handoff skill：FAIL
分类：TRANSITION_ACTIVITY_UP_HANDOFF_NOT_LEARNED
```

Temporal privileged observation 明显增加了指尖 break/recontact，但没有把它们变成有效承力接力。策略更会“换接触”，却更容易破坏 TIM 重叠支撑。

## 单变量实验

| 设定 | Baseline | Temporal |
|---|---:|---:|
| Task | P3-only | P3-only |
| Observation | 152 维 | 172 维 |
| Reward | O1.2 penalty60 | 完全相同 |
| Seed | 20260730 | 20260730 |
| Environments | 256 | 256 |
| PPO iterations | 60 | 60 |
| Rollout control steps | 1440 | 1440 |
| Simulation transitions | 368,640 | 368,640 |
| O0 bank SHA-256 | `73ab...c65` | `73ab...c65` |

Temporal 组仅增加以下通用历史信息：

- 五指接触占空比 EMA；
- 五指力幅 EMA 和差分；
- reset-relative height 当前值和 EMA；
- 竖直净支撑力的当前值、EMA 和差分。

所有手指使用对称表示，没有提供“先放拇指/后放中指”之类接触顺序标签。

## 评估结果

| 指标 | Baseline | Temporal | 变化 |
|---|---:|---:|---:|
| Raw gait | 13.45% | 93.71% | +80.26 pp |
| Qualified gait | 0% | 0% | 0 |
| Support handoff success | 0% | 0% | 0 |
| Thumb gait | 0.36% | 20.18% | +19.82 pp |
| Index gait | 0% | 75.66% | +75.66 pp |
| Middle gait | 13.09% | 63.49% | +50.40 pp |
| Height recovery | 0% | 0% | 0 |
| Height drop | 99.45% | 100% | +0.55 pp |
| TIM overlap ratio | 97.56% | 50.08% | -47.48 pp |
| Failure rate | 85.27% | 100% | +14.73 pp |
| Ballistic rate | 0% | 3.75% | +3.75 pp |

Baseline 评估完成 `550` episodes，temporal 完成 `986` episodes；两者使用相同的 `256 x 240` 评估 transition 预算。Temporal 因更早终止而完成了更多 episodes，表中全部为 episode-normalized rate。

## Support Handoff 判据

Support handoff 不要求指定手指顺序，但必须同时满足：

1. 至少一根手指完成经 `3` 步 debounce 的 break/recontact；
2. TIM 至少两指重叠覆盖 episode 的 `>=80%`；
3. 全无 TIM 支撑不超过连续 `2` 步；
4. 无 ballistic/失败；
5. 高度不下降 5 mm，或下降后恢复至 reset 高度 2 mm 范围内。

Qualified gait 仍独立要求目标轴对齐 `>=0.70`、无 ballistic、无 contact-loss 和无失败。

## 解读

Temporal observation 解决了部分“策略看不见历史”的问题：三指重接触率都显著增加。但由于当前冻结 objective 不区分“频繁切换”和“可恢复的承力接力”，策略将新信息用来制造更多接触变化，而不是维持支撑。

因此：

- **不能**将 raw gait `93.71%` 宣称为 finger gait 进展；
- temporal observation 单独不足以解决 support handoff credit；
- 当前不应扩大训练或上 5090/H200；
- 下一步应保留通用 temporal state，但用“支撑重叠 + 高度可恢复 + 非 ballistic”作为联合 objective/evaluation contract，仍不指定手指顺序。

## 产物

- `o1_5_baseline_P3Only_budget60_eval256.json/.md`
- `o1_5_temporal_P3Only_budget60_eval256.json/.md`
- `O1_5_Temporal_Credit_对比结果.json`
- `O1_5_Temporal_Credit_对比图.png`

本阶段未修改 O0，未新建 teacher，未引入接触顺序 reward。
