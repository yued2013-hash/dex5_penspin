# O1.3 Privileged State Augmentation 验收报告

## 结论

**O1.3 观测与能力诊断：PASS。Oracle skill：NOT PASS。**

状态增强证明了 thumb 失败不是“策略从未尝试恢复”。新策略在 `98.70%` episode 中尝试 thumb 恢复，并在 `64.23%` episode 中完成三步 debounce 的 thumb re-contact。但 TIM 承力链几乎总是整体断开，contact loss 达 `98.70%`，高度恢复仍为 `0%`，qualified gait 仍为 `0%`。

## Observation 审计

原 152 维 observation 已包含：

- 笔的绝对位置、姿态、线速度、角速度和笔轴。
- 每个手指的即时接触二值、三维接触力、力模和切向滑移速度。
- 手指位置、关节状态、PD target、previous action 和物理参数。

因此没有重复添加 thumb force。O1.3 只增加 6 维缺失信息：

- `sin(theta_acc)`、`cos(theta_acc)` 和截断后的 unwrapped rotation。
- 相对 reset 高度。
- thumb 本 episode 是否已 release、是否已 re-contact。

总观测维度：`152 -> 158`。未增加人工 P1-P5 phase waypoint，未修改 O1.2 P3-only penalty60 reward。

## 确定性对照

同为 seed `20260730`、`256 env x 60 iterations`，评估为 `256 env x 240 steps`：

| 指标 | O1.2 penalty60 | O1.3 P3State60 |
|---|---:|---:|
| Raw gait | 35.16% | 97.18% |
| Qualified gait | 0.00% | 0.00% |
| Raw 180 deg | 99.80% | 86.04% |
| Failure | 54.88% | 100.00% |
| Ballistic | 0.20% | 17.39% |
| Contact loss | 0.59% | 98.70% |
| Axis-aligned episode | 0.00% | 0.31% |
| Mean axis alignment | 0.553 | 0.579 |
| Mean target / wrong-axis speed | 2.20 / 3.09 rad/s | 4.84 / 3.98 rad/s |
| Thumb attempt | 0.39% | 98.70% |
| Thumb re-contact | 0.00% | 64.23% |
| Height drop >=5 mm | 99.80% | 98.70% |
| Height recovery to <=2 mm error | 0.00% | 0.00% |
| Mean minimum height delta | -35.11 mm | -17.09 mm |

## Finger Capability

O1.3 最终 checkpoint，`1311` 个完成 episode：

| Finger | Break | Debounced re-contact |
|---|---:|---:|
| Thumb | 98.70% | 64.23% |
| Index | 98.70% | 93.44% |
| Middle | 98.70% | 13.04% |
| Ring | 0% | 0% |
| Pinky | 0% | 0% |

thumb 平均在第 `3.03` 个控制步断开，第 `9.43` 步出现恢复尝试，成功重接的 episode 平均在第 `13.19` 步通过 debounce。这证明策略有 thumb recovery action capability，但由于 index/middle/thumb 几乎同时 release，最长无接触窗口接近 contact-loss 终止门。

## 谨慎解读

O1.3 的 drop termination 为 `0`，但不代表高度稳定性已解决。`1294/1311` episode 更早被 contact loss 终止，属于竞争风险导致的 censoring。事实上 `98.70%` episode 下沉超过 5 mm，没有任何 episode 恢复到 reset 高度误差 2 mm 内。

## 下一瓶颈

当前问题已从“thumb 没有恢复意图”重新定位为：

```text
thumb 会尝试并能重接
        +
index 也能重接
        -
TIM 承力交接没有时间重叠
        -
笔高度无恢复
```

下一小实验应先测试 support handoff/overlap 的动力学可达性，而不是再加 reward、增加 PPO 步数或上 5090/H200。

## 证据

- `o1_3_baseline_penalty60_recovery_eval256.json`
- `o1_3_P3State60_final_eval256.json`
- `O1_3_Privileged_State_Augmentation_验收结果.json`
