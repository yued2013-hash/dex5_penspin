# O1.19 Reduced-order Handoff MPC Feasibility Audit 验收报告

## 结论

Reduced-order MPC 未通过全部预注册 widening gates。

用于正式诊断比较的 MPC：`C_mpc_50ms`。Calibration safety eligibility：`FAIL`。

## Calibration

| Mode | Nominal basin | Joint-L1 basin | Joint-L1 unsafe | Score |
|---|---:|---:|---:|---:|
| `A_open_loop` | 7.03% | 16.41% | 1.56% | 0.1633 |
| `B_o1_18_task_dls_soft` | 3.12% | 2.34% | 7.03% | -0.0086 |
| `C_mpc_50ms` | 3.12% | 2.34% | 11.72% | -0.0320 |
| `C_mpc_100ms` | 3.12% | 2.34% | 12.50% | -0.0359 |

## Held-out A/B/C

| Mode | Scenario | Basin | Handoff | Height recovery | Min height p10 (mm) | Unsafe |
|---|---|---:|---:|---:|---:|---:|
| A open | `nominal` | 53.12% | 53.12% | 100.00% | -0.087 | 0.00% |
| A open | `joint_L1` | 10.94% | 10.94% | 100.00% | -0.111 | 3.91% |
| A open | `joint_L2` | 5.47% | 5.47% | 100.00% | -0.166 | 10.16% |
| A open | `joint_L3` | 0.78% | 0.78% | 100.00% | -0.219 | 21.09% |
| B task-space | `nominal` | 5.47% | 5.47% | 100.00% | -0.105 | 6.25% |
| B task-space | `joint_L1` | 3.12% | 3.12% | 100.00% | -0.093 | 7.03% |
| B task-space | `joint_L2` | 2.34% | 2.34% | 100.00% | -0.140 | 5.47% |
| B task-space | `joint_L3` | 0.00% | 0.00% | 100.00% | -0.247 | 21.09% |
| C MPC | `nominal` | 3.12% | 3.12% | 100.00% | -0.185 | 10.94% |
| C MPC | `joint_L1` | 1.56% | 1.56% | 100.00% | -0.179 | 14.06% |
| C MPC | `joint_L2` | 3.12% | 3.12% | 100.00% | -0.234 | 8.59% |
| C MPC | `joint_L3` | 0.00% | 0.00% | 100.00% | -0.402 | 21.88% |

## Widening gates

| Gate | 结果 |
|---|---|
| `mpc_joint_L1_at_least_30pct` | FAIL |
| `mpc_joint_L1_at_least_2x_open_loop` | FAIL |
| `mpc_joint_L1_better_than_o1_18_task_feedback` | FAIL |
| `mpc_joint_L2_at_least_10pct` | FAIL |
| `mpc_joint_L1_unsafe_not_worse_than_open_by_more_than_2pp` | FAIL |

## Failure taxonomy

| MPC scenario | Episode taxonomy |
|---|---|
| `nominal` | `no_TIM_transition`=87; `axis_misaligned`=19; `throw`=14; `contact_force_overload`=4; `recovered_strict`=4 |
| `joint_L1` | `no_TIM_transition`=69; `axis_misaligned`=31; `throw`=18; `contact_force_overload`=8; `recovered_strict`=2 |
| `joint_L2` | `no_TIM_transition`=80; `axis_misaligned`=29; `throw`=11; `contact_force_overload`=4; `recovered_strict`=4 |
| `joint_L3` | `no_TIM_transition`=82; `axis_misaligned`=17; `penetration`=16; `throw`=12; `joint_gate_miss`=1 |

完整 A/B/C taxonomy 同时保存在机器可读 JSON 的 `held_out_comparison[*].scenarios[*].failure_taxonomy`。

## 方法边界

- MPC horizon 为 50/100 ms，每步从 64 个候选序列中选择第一步动作并重新规划。
- 只优化 index/middle/thumb 的 12 个关节，残差上限为 `0.025 rad`。
- 预测约束覆盖支撑力、高度、轴向、聚合接触和动作边界。
- MPC 序列、CEM plan、Jacobian 和参考轨迹均未保存。
- 未训练 PPO、未修改 reward/O0、未生成 teacher/reset dataset。

## 产物

- 机器可读报告：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_19_Reduced_Order_MPC_Audit.json`
- 可视化：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_19_Reduced_Order_MPC_Map.png`
- 原始结果：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_19_Reduced_Order_MPC_Raw.json`
