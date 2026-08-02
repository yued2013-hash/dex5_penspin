# O1.18 Task-Space Closed-Loop Basin Audit 验收报告

## 结论

Task-space residual feedback 未通过全部预注册 widening gates。

用于诊断比较的 task-space 控制器：`C_task_dls_soft`。A/B/C 在 held-out 集合内使用完全匹配的扰动。

Calibration safety eligibility：`FAIL`。若为 FAIL，C 只是最优 fallback 的诊断结果，不是合格控制器。

## Calibration

| Mode | Nominal basin | Joint-L1 basin | Joint-L1 unsafe | Score |
|---|---:|---:|---:|---:|
| `A_open_loop` | 7.03% | 14.84% | 1.56% | 0.1477 |
| `B_o1_17_joint_tracking` | 31.25% | 4.69% | 4.69% | 0.0547 |
| `C_task_dls_soft` | 3.91% | 3.12% | 7.03% | 0.0000 |
| `C_task_dls_medium` | 1.56% | 2.34% | 10.94% | -0.0297 |
| `C_task_dls_strong` | 0.78% | 0.78% | 10.16% | -0.0422 |

## Held-out A/B/C

| Mode | Scenario | Basin | Handoff | Height recovery | Min height p10 (mm) | Unsafe |
|---|---|---:|---:|---:|---:|---:|
| A open | `nominal` | 40.62% | 40.62% | 100.00% | -0.086 | 0.00% |
| A open | `joint_L1` | 15.62% | 15.62% | 100.00% | -0.107 | 1.56% |
| A open | `joint_L2` | 6.25% | 6.25% | 100.00% | -0.155 | 9.38% |
| A open | `joint_L3` | 0.78% | 0.78% | 100.00% | -0.220 | 25.78% |
| B q-track | `nominal` | 30.47% | 30.47% | 100.00% | -0.109 | 3.91% |
| B q-track | `joint_L1` | 5.47% | 5.47% | 100.00% | -0.129 | 7.03% |
| B q-track | `joint_L2` | 1.56% | 1.56% | 100.00% | -0.165 | 10.16% |
| B q-track | `joint_L3` | 1.56% | 1.56% | 100.00% | -0.342 | 26.56% |
| C task-space | `nominal` | 1.56% | 1.56% | 100.00% | -0.108 | 7.03% |
| C task-space | `joint_L1` | 8.59% | 8.59% | 100.00% | -0.100 | 10.16% |
| C task-space | `joint_L2` | 4.69% | 4.69% | 100.00% | -0.131 | 7.03% |
| C task-space | `joint_L3` | 0.00% | 0.00% | 100.00% | -0.232 | 19.53% |

## Widening gates

| Gate | 结果 |
|---|---|
| `task_joint_L1_at_least_30pct` | FAIL |
| `task_joint_L1_at_least_2x_open_loop` | FAIL |
| `task_joint_L1_better_than_o1_17_tracking` | PASS |
| `task_joint_L2_at_least_10pct` | FAIL |
| `task_joint_L1_unsafe_not_worse_than_open_by_more_than_2pp` | FAIL |

## 方法边界

- Task input：笔高度、笔角速度、总支撑 force/torque、聚合接触数量。
- 输出为最大 `0.02-0.04 rad` 的小关节目标残差；没有指定某根手指必须接触。
- 三阶段 task Jacobian 通过内存中的正负关节探针辨识；Jacobian、CEM plan 和参考轨迹均未保存。
- 三阶段 Jacobian 的有效秩和条件数必须结合机器可读报告解释；病态局部映射会放大 task error 并触发 residual clipping。
- 未训练 PPO、未修改 reward/O0、未生成 teacher 或 reset dataset。
- 结论仅覆盖 bank `1155`、300 ms 和当前扰动尺度。

## 产物

- 机器可读报告：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_18_Task_Space_Basin_Audit.json`
- 可视化：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_18_Task_Space_Basin_Map.png`
- 原始结果：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_18_Task_Space_Basin_Raw.json`
