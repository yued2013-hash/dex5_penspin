# O1.12 Post-Handoff Axis Stabilization Audit 验收报告

## 结论

`PASS (diagnostic) / POST-HANDOFF PHYSICALLY REACHABLE IN AUDITED BOUND`

- 400 ms handoff 搜索找到 transition entry：`3/4`
- 同一 PhysX episode 内连续复现 transition：`3/4`
- 200 ms post-handoff 严格稳定化存在性成功：`1/4`
- 失败 taxonomy：`{"success": 1, "axis_misaligned": 2, "handoff_transition_not_reproduced": 1}`

bank `996` 证明 transition 后的轴稳定化在当前 PhysX、20D residual action、`+/-0.12 rad`、200 ms 窗口内可达：

- 轴对齐：`0.852`，门槛 `0.5`
- 最大高度损失：`-0.000 mm`
- TIM overlap：`100.0%`
- 支撑力范围：`0.305 - 3.479 N`
- transition replay：`73.44%`
- strict post replay：`71.88%`
- 无 ballistic/contact loss/penetration/drop/throw

## 逐入口结果

| Bank | Transition | Post success | Axis | Min height delta | TIM overlap | Post replay | Taxonomy |
|---:|---:|---:|---:|---:|---:|---:|---|
| 996 | True | True | 0.852 | 0.000 | 1.000 | 71.88% | success |
| 1245 | True | False | 0.356 | -0.478 | 1.000 | 0.00% | axis_misaligned |
| 1203 | False | False | 0.000 | 0.000 | 0.000 | 0.00% | handoff_transition_not_reproduced |
| 1171 | True | False | 0.347 | 0.000 | 1.000 | 0.00% | axis_misaligned |

## 原因判断

1. **不是 post-handoff 动力学绝对不可达。** Bank 996 的严格成功排除了这一结论。
2. **高度和总支撑不是当前主失败门。** 可 transition 的失败入口仍保持高度、TIM overlap 和接触力。
3. **主要瓶颈是 entry-dependent axis basin。** `1245/1171` 在支撑安全时轴对齐仅约 `0.356/0.347`。
4. **transition 复现性仍是第二瓶颈。** `1203` 在 handoff search 中可达，但未在 post-stage 连续审计中稳定重现。

## 约束审计

- 未训练 PPO
- 未修改 reward
- 未修改 O0
- 未生成 teacher
- 未序列化任何 CEM action plan 或 post-transition state
- post 搜索从同一 PhysX episode 内的 transition 下一控制步开始

可视化：`O1_12_Post_Handoff_Axis_Stabilization_Map.png`
