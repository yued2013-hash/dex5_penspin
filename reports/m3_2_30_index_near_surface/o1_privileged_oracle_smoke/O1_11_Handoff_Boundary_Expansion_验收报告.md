# O1.11 Handoff Boundary Expansion 验收报告

## 结论

`PASS (diagnostic) / STRICT HANDOFF FAIL`

三个条件的严格 handoff 均为 `0/3`。当前不能宣称 P3 handoff 已物理可达，也不能宣称绝对不可达。

| 条件 | Strict | TIM transition | 主要结果 |
|---|---:|---:|---|
| baseline: 300 ms, 20D, +/-0.18 rad | 0/3 | 0/3 | 三个入口均未换指 |
| TIM cooperative: 300 ms, 12D, +/-0.12 rad | 0/3 | 0/3 | 小幅 TIM 协同未扩张可行域 |
| horizon: 400 ms, 20D, +/-0.18 rad | 0/3 | 1/3 | bank 996 出现安全 transition，但轴对齐失败 |

## 边界判断

- **A. Action space 不足：未被支持。** TIM-only `+/-0.12 rad` 没有产生 transition，只能说这个小幅协同空间不足，不能证明原 20D action space 太小。
- **B. Time horizon 不足：有贡献，但不是唯一原因。** 400 ms 使 bank 996 出现 TIM release/recontact，但 strict 仍为 0。
- **C. P3 canonical 入口不足：存在明显 entry dependence。** 只有 bank 996 接近严格边界，817/948 在同预算下均未 transition。

## 最接近边界

- bank: `996`
- 条件: 400 ms, all joints, `+/-0.18 rad`
- TIM transition: `True`
- 最大高度损失: `0.744 mm`
- 轴对齐: `0.108`，门槛 `0.5`
- handoff margin: `-0.785`
- 最终失败类型: `axis_misaligned`

## 约束审计

- 未训练 PPO
- 未修改 reward
- 未生成或导出 teacher/action plan
- 未修改 O0 bank
- 三组均使用 bank `817/948/996`、population `64`、iterations `10`、3 waypoints

边界图：`O1_11_Handoff_Feasibility_Boundary_Map.png`
