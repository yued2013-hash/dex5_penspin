# O1.15 Long-Horizon Handoff Basin Audit 验收报告

## 结论

O1.15 对 O1.14 的全部 `37` 条动态候选执行了匹配的 `300/400/500 ms` CEM。
这里的 basin size 是 **O1.14 候选层内的条件估计**，不是全部 P3 的体积或成功率。

- 至少一个 horizon 找到 strict handoff：`3/37 = 8.11%`
- strict 且末端高度恢复到 `-2 mm` 内：`3/37 = 8.11%`
- recovered-strict clone replay `>=50%`：`0/37 = 0.00%`
- 存在性成功来自 `3` 个 parent、`3` 个 source trajectory。
- 鲁棒子集来自 `0` 个 parent、`0` 个 source trajectory。

`996` 在本轮排名第 `5`，结局为 `contact_force_overload`。
这不撤回 O1.12 的可达性证据：O1.12 使用 `128x20` 两阶段 handoff+post-axis 搜索，
本轮为 `64x10` 单阶段搜索，差异反映搜索预算和参数化敏感性。

## Horizon 对比

| Horizon | TIM transition | Strict | Strict + height recovery | Robust clone |
|---:|---:|---:|---:|---:|
| 300 ms | 18/37 | 1/37 | 1/37 | 0/37 |
| 400 ms | 16/37 | 1/37 | 1/37 | 0/37 |
| 500 ms | 19/37 | 1/37 | 1/37 | 0/37 |

## 状态排名（前 20）

Level：`0=无转换，1=TIM转换，2=strict，3=strict+高度恢复，4=clone>=50%`。

| Rank | Bank | Horizon (ms) | Level | Clone recovered-strict | Axis | Min z (mm) | Final z (mm) | 结局 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `894` | 400 | 3 | 10.9% | 0.552 | 0.000 | 7.406 | `recovered_but_low_clone_reproducibility` |
| 2 | `1155` | 300 | 3 | 1.6% | 0.578 | -0.101 | 3.134 | `recovered_but_low_clone_reproducibility` |
| 3 | `831` | 500 | 3 | 1.6% | 0.520 | -0.095 | 5.612 | `recovered_but_low_clone_reproducibility` |
| 4 | `1239` | 400 | 1 | 0.0% | 0.470 | -0.019 | 9.466 | `contact_force_overload` |
| 5 | `996` | 300 | 1 | 0.0% | 0.453 | -0.233 | 1.421 | `contact_force_overload` |
| 6 | `1025` | 400 | 1 | 0.0% | 0.380 | -0.151 | 5.232 | `axis_misaligned_after_transition` |
| 7 | `1053` | 400 | 1 | 0.0% | 0.508 | -0.045 | 4.724 | `contact_force_overload` |
| 8 | `1225` | 300 | 1 | 0.0% | 0.312 | 0.000 | 3.668 | `axis_misaligned_after_transition` |
| 9 | `1224` | 500 | 1 | 0.0% | 0.306 | 0.000 | 3.379 | `axis_misaligned_after_transition` |
| 10 | `895` | 300 | 1 | 0.0% | 0.244 | -0.581 | 4.749 | `contact_force_overload` |
| 11 | `1036` | 300 | 1 | 0.0% | 0.227 | -0.083 | 2.305 | `axis_misaligned_after_transition` |
| 12 | `883` | 400 | 1 | 0.0% | 0.213 | -0.836 | 4.637 | `contact_force_overload` |
| 13 | `813` | 500 | 1 | 0.0% | 0.204 | 0.000 | 5.861 | `axis_misaligned_after_transition` |
| 14 | `1199` | 500 | 1 | 0.0% | 0.201 | 0.000 | 5.911 | `contact_force_overload` |
| 15 | `927` | 400 | 1 | 0.0% | 0.186 | 0.000 | 5.137 | `contact_force_overload` |
| 16 | `816` | 500 | 1 | 0.0% | 0.164 | 0.000 | 1.356 | `contact_force_overload` |
| 17 | `932` | 300 | 1 | 0.0% | 0.139 | 0.000 | 2.665 | `contact_force_overload` |
| 18 | `1115` | 300 | 1 | 0.0% | 0.123 | -0.569 | 0.559 | `axis_misaligned_after_transition` |
| 19 | `877` | 500 | 1 | 0.0% | 0.118 | -0.611 | 0.611 | `contact_force_overload` |
| 20 | `823` | 300 | 1 | 0.0% | -0.085 | -0.041 | 2.588 | `axis_misaligned_after_transition` |

## Failure taxonomy

| 类型 | 状态数 |
|---|---:|
| `recovered_but_low_clone_reproducibility` | 3 |
| `contact_force_overload` | 10 |
| `axis_misaligned_after_transition` | 7 |
| `no_TIM_transition` | 17 |

## 边界

- CEM 未序列化最优 action plan，也未生成 teacher。
- 未训练 PPO，未修改 reward 或 O0。
- 一次存在性 rollout 与 clone 可复现性分开报告，不把偶然成功直接当作 basin。

## 产物

- 机器可读结果：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_15_Long_Horizon_Handoff_Basin_Audit.json`
- 可视化：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_15_Long_Horizon_Handoff_Basin_Map.png`
