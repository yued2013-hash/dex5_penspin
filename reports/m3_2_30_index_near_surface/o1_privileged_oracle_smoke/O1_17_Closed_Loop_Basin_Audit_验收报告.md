# O1.17 闭环 Basin Widening Audit 验收报告

## 结论

闭环反馈未通过全部 widening gate；当前仍不能称为可靠技能吸引域。

选择的反馈器：`support_adaptive`。反馈器在 calibration seed 上选择，以下正式结果来自不同的 held-out perturbation seed。

## Calibration

| Mode | Nominal success | Joint-L1 success | Nominal unsafe | Joint-L1 unsafe | Selection score |
|---|---:|---:|---:|---:|---:|
| `open_loop` | 26.56% | 17.19% | 0.78% | 1.56% | 0.1906 |
| `q_track_weak` | 21.88% | 8.59% | 4.69% | 12.50% | 0.0453 |
| `q_track_medium` | 20.31% | 4.69% | 3.12% | 4.69% | 0.0437 |
| `support_adaptive` | 21.88% | 6.25% | 0.78% | 11.72% | 0.0258 |

| Scenario | Open loop | Closed loop | 增量 | Open unsafe | Closed unsafe |
|---|---:|---:|---:|---:|---:|
| `nominal` | 26.56% | 29.69% | +3.12% | 0.00% | 3.91% |
| `joint_L1` | 19.53% | 3.91% | -15.62% | 1.56% | 10.94% |
| `joint_L2` | 7.81% | 0.78% | -7.03% | 9.38% | 9.38% |
| `joint_L3` | 0.78% | 1.56% | +0.78% | 14.06% | 15.62% |

## 预注册门槛

| Gate | 结果 |
|---|---|
| `joint_L1_at_least_30pct` | FAIL |
| `joint_L1_at_least_2x_matched_baseline` | FAIL |
| `joint_L2_at_least_10pct` | FAIL |
| `joint_L1_unsafe_not_worse_by_more_than_2pp` | FAIL |

## 方法边界

- 反馈只跟踪内存中的 nominal `q/dq` 参考，并根据总支撑力和笔高度连续调节增益；没有规定手指接触顺序。
- calibration 与 held-out 使用不同扰动种子；open/closed 在各自集合内使用完全匹配的扰动。
- 这是 bank `1155`、300 ms、指定扰动尺度内的局部结论，不证明全局 attractor、完整 finger gait 或 sim2real 鲁棒性。
- 本轮只否定“单条 nominal `q/dq` 跟踪 + 支撑力/高度增益调节”这一反馈架构，不否定其他 object-state feedback、MPC 或学习型闭环控制。
- CEM plan 和 nominal reference 均未序列化；未训练 PPO、未修改 reward/O0、未生成 teacher/reset dataset。

## 产物

- 机器可读报告：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_17_Closed_Loop_Basin_Audit.json`
- 可视化：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_17_Closed_Loop_Basin_Map.png`
- 原始结果：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_17_Closed_Loop_Basin_Raw.json`
