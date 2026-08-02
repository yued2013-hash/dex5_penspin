# O1.9 Handoff Basin Analysis 验收报告

## 结论

```text
O1.9 特征提取与覆盖分析：PASS
Handoff feasibility classifier：NOT_IDENTIFIABLE_ONE_CLASS
A/B/C 归因：INCONCLUSIVE_NO_STRICT_FEASIBLE_LABELS
阶段状态：PARTIAL_PREREQUISITE_FAILED
```

O1.9 发现 O1.8 在 reset-return 帧读取了未刷新的 contact sensor。加入 `2` 个接触重建 hold step 后，严格 O1.8 从旧的 `6/12` 修正为 `0/12`。因此现在只有一个类别（strict infeasible），无法训练或验证可行性分类器。

## 读取的初始特征

对全部 `396` 个 P3-train R3 reset 执行只读 PhysX reset-replay，接触重建后提取 `66` 个特征：

- fingertip geometry：指尖轴向位置、笔表面间隙、TIM 指间距离；
- contact wrench：每指三维接触力、切向力、总力及关于笔质心的力矩；
- pen pose：位置与笔轴；
- velocity：线速度与角速度；
- support margin：重力残差、TIM/总接触力、承力分散度、等效力臂和 `6 N` 力上限余量。

所有原始特征组都已在 privileged oracle observation 中直接提供，或可由观测量与已知重力代数推导。

## Canonical Coverage Gap

| 项目 | 结果 |
|---|---:|
| P3-train reset | 396 |
| unique parent | 352 |
| unique source trajectory | 219 |
| 已进行修正 CEM 审计 | 12（3.03%） |
| 审计 parent/source | 12 / 12 |
| strict feasible | 0 |
| near-feasible（仅轴错位） | 1，bank `996` |
| 审计样本特征跨度覆盖中位数 | 65.2% |
| 全 P3 到最近审计点距离 p50/p95 | 0.645 / 1.104 RMS-z |

现在的 coverage gap 不是 O0 缺少 P3 reset，而是只有 `3.03%` 的 P3-train 状态获得了正确 CEM 标签，且没有正类。因此不能外推 feasible basin 的密度或位置。

## Feasibility Classifier

```text
status: NOT_IDENTIFIABLE_ONE_CLASS
positive labels: 0
negative labels: 12
model generated: false
```

在单类别下，ROC AUC、balanced accuracy、置换检验和全 P3 可行概率都没有定义。本阶段没有输出模型，也不得用 near-feasible bank `996` 当作正标签。

## A/B/C 判断

| 假设 | 当前状态 | 理由 |
|---|---|---|
| A. reset coverage 不足 | UNRESOLVED | 12/396 搜索覆盖不足，无法区分无 basin 与未采到的窄 basin |
| B. exploration 困难 | UNRESOLVED, PLAUSIBLE | 扩预算 CEM 仍为零，但已出现只差轴对齐的安全 TIM transition |
| C. representation 问题 | NOT TESTABLE | 特征在 oracle 中可观测，但无正标签无法检验可分性 |

所以本轮不应强行归因为 A、B 或 C。下一个有效门槛是先获得至少 `5-10` 个通过同一严格门槛的正类，或明确改写“物理可达”假设；在此之前不应启动 PPO 或修改 reward。

机器可读结果：[O1_9_Handoff_Basin_Analysis.json](./O1_9_Handoff_Basin_Analysis.json)

可视化：[O1_9_Handoff_Basin_Visualization.png](./O1_9_Handoff_Basin_Visualization.png)

只读特征快照：[O1_9_P3_initial_feature_snapshot.json](./O1_9_P3_initial_feature_snapshot.json)
