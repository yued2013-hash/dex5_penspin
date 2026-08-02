# O1.10 Handoff Feasibility Boundary Audit 验收报告

## 状态

```text
O1.10: PARTIAL_BLOCKED_BY_SIMULATOR_THROUGHPUT
strict feasible boundary: NOT CLOSED
```

本轮完成了边界审计代码、修正版 baseline 复用、搜索预算敏感性和 canonical-entry 分析。但本机新 Isaac 运行进入异常低吞吐路径，targeted `600 ms` 条件运行超过一小时仍未完成，已停止且未产生结果文件。

中止或超时不是 `0 feasible`。因此 horizon expansion、TIM-only 和 action-limit expansion 均标记为 `NOT_MEASURED`，没有伪造对照数字。

## 已实测 Baseline

| 设置 | 值 |
|---|---:|
| P3 canonical entries | 12 |
| horizon | 100 / 200 / 300 ms |
| action space | 20 维 residual joint target |
| action limit | `+-0.18 rad` |
| waypoints | 3 |
| CEM | population 64，10 iterations |
| strict feasible | **0/12** |

Failure taxonomy：

```text
no_TIM_transition: 11
axis_misaligned:    1
```

搜索预算增加到 population `128`、`30` iterations 后仍为 `0/12`：

```text
no_TIM_transition: 10
throw:              1
axis_misaligned:    1
```

这表明在 `<=300 ms / 20D / +-0.18 rad` 边界内，单纯扩大 CEM 样本预算没有打开 strict feasible basin。

## Feasibility Boundary

| 因素 | 已实测边界 | Strict feasible | 当前判断 |
|---|---|---:|---|
| temporal horizon | 100-300 ms | 0/12 | `<=300 ms` 未找到可行解 |
| action space | 全20关节 | 0/12 | 全动作空间未打开边界 |
| TIM cooperation | TIM-only `+-0.12 rad` | N/A | NOT MEASURED |
| action limit | `+-0.18 rad` | 0/12 | `+-0.30 rad` NOT MEASURED |
| canonical entry | 12个分散 P3 reset | 0/12 | bank `996` 最接近边界 |
| horizon expansion | 400-600 ms | N/A | NOT MEASURED |

## Canonical Entry

bank `996` 是当前唯一的安全 near-boundary entry：

- 初始主承力 TIM 完成 release-recontact；
- 高度、承力连续、接触力上限通过；
- 轴对齐 `0.385 < 0.5`；
- strict margin `-0.229`。

因此 canonical entry 很可能影响可达性，但尚没有 strict positive 能够定义其 basin 边界。

## 当前最严格的结论

```text
已排除：在 <=300 ms、+-0.18 rad 内，只需增加 CEM 样本量即可达。
未排除：更长时域、TIM 低维协同、更大动作边界、更合适的 P3 entry。
不能宣称：P3 handoff 绝对动力学不可达。
```

机器可读结果：[O1_10_Handoff_Feasibility_Boundary_Audit.json](./O1_10_Handoff_Feasibility_Boundary_Audit.json)

可视化：[O1_10_Handoff_Feasibility_Boundary_Visualization.png](./O1_10_Handoff_Feasibility_Boundary_Visualization.png)
