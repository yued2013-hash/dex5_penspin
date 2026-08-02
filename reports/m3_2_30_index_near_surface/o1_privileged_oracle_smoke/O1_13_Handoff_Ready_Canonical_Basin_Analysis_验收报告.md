# O1.13 Handoff-Ready Canonical Basin Analysis 验收报告

## 结论

`PASS (diagnostic) / STATIC CANONICAL REFINEMENT REJECTED`

已完成 geometry、wrench、contact/support margin、pen pose/velocity 和 20 维 joint configuration 的对比，并生成了透明的 prototype-margin readiness score。但独立 CEM 验证表明：**该分数没有扩大 handoff-ready P3 区域。**

| Distribution | Raw TIM transition | Admissible transition | Strict handoff |
|---|---:|---:|---:|
| readiness-selected | 0/12 | 0/12 | 0/12 |
| original P3 spread | 1/12 | 0/12 | 0/12 |

原始 spread 的唯一 raw transition 来自 bank `996`，但最大接触力 `8.223 N` 超过 `6 N` 门槛，且轴对齐仅 `0.220`，因此也不是 admissible/strict 成功。

## 原型对比

- feasible prototype：`996`
- axis failures：`1245 / 1171`
- transition failure：`1203`
- readiness 不是概率；它只是六组等权标准化原型距离 margin。

| Feature group | 996 vs failure mean RMS-z |
|---|---:|
| `joint_configuration` | 0.805 |
| `pen_pose` | 0.293 |
| `velocity` | 0.180 |
| `geometry` | 0.176 |
| `wrench` | 0.124 |
| `support_margin` | 0.107 |

最大差异来自 joint configuration，其次是 pen pose 和 geometry。但未见入口的负结果说明：**静态姿态/力特征接近 996，不等于局部动力学可控性接近 996。**

## Canonical refinement proposal

1. 不过滤、不替换 frozen P3，不修改 O0。
2. 当前 readiness score 只保留为 analysis-only hypothesis generator，不用于 reset sampler 或 curriculum。
3. 下一版 refinement 应基于更多 provenance-isolated 动态 CEM 标签，且分开 transition reachability 与 post-axis stabilization。
4. 增加 action-conditioned `50-100 ms` 局部响应特征：轴对齐响应、支撑力斜率、高度加速度，而不是继续叠加静态 pose 特征。
5. 任何 handoff-ready overlay 必须按 parent/source 留出验证，并在 clone replay 下显著超过 spread baseline 后才能晋级。

## 约束审计

- 未训练 PPO
- 未修改 reward
- 未生成 teacher
- 未修改 O0
- 未创建新 reset dataset
- readiness-selected 的 12 个入口与原型不共享 parent/source，选择内部也无 parent/source leakage

Basin map：`O1_13_Handoff_Ready_Canonical_Basin_Map.png`
