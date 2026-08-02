# O1.8 Handoff Reachability Audit 修正验收报告

## 最终结论

```text
O1.8 审计管线：PASS
严格 feasible rate：0/12 = 0%
结论：A_UNREACHABLE_WITHIN_AUDITED_BOUNDS
```

这个 `A` 只表示在当前 `12` 个 P3 reset、`+-0.18 rad`、三 waypoint、`100-300 ms` 和既定 CEM 预算内没有找到全门槛解，不是动力学不可能的数学证明。

## 关键修正

初版 O1.8 在 `env.reset()` 返回后立即读取 contact sensor。该时刻 PhysX 接触传感器尚未刷新，因此 `initial_force` 可能是全零或上一 rollout 的陈旧值，污染了：

- 初始主承力 TIM 手指选择；
- old-support unload margin；
- 同步承力转移判定。

旧版 `6/12 = 50%` 因此已撤回，归档为 `pre_contact_rebuild_invalid`，不得再用于可达性或 PPO 瓶颈结论。

修正版在每次 reset 后先执行 `2` 个 zero-residual hold step 重建接触，然后清零 handoff 事件计数并开始 `100-300 ms` 审计时域。复验后初始主承力不再是零值，且能按 reset 确定 index/middle 主承力。

## 正式预算结果

| 时域 | 可行 | 失败分类 |
|---|---:|---|
| 100 ms | 0/12 | 12 个 `no_TIM_transition` |
| 200 ms | 0/12 | 12 个 `no_TIM_transition` |
| 300 ms | 0/12 | 10 个 `no_TIM_transition`，1 个 `axis_misaligned`，1 个 `throw` |
| 各时域最佳 | **0/12** | 11 个 `no_TIM_transition`，1 个 `axis_misaligned` |

正式预算为 population `64` / reset、`10` 轮 CEM。最接近的安全候选为 bank `996`：它完成了初始主承力 TIM 的 release-recontact，高度、承力和力上限通过，但轴对齐仅 `0.385 < 0.5`，handoff margin 为 `-0.229`。

## 扩预算敏感性检查

额外使用 population `128`、`30` 轮 CEM，搜索量约为正式预算的 `6` 倍，结果仍为 `0/12`：

- `10` 个 `no_TIM_transition`；
- `1` 个 `throw`；
- `1` 个 `axis_misaligned`。

扩预算结果降低了“只是原 CEM 过小”的可能性，但仍不能排除动作参数化、时域或更窄 basin 导致的搜索失败。

## 边界

- 未训练 PPO，未修改 reward 或 O0。
- 未保存 CEM 动作，未生成 teacher。
- O0 SHA-256 仍为 `73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`。
- 当前证据无法支持 `B_POLICY_LEARNING_BOTTLENECK`，也不足以宣称绝对动力学不可达。

机器可读结果：[O1_8_Handoff_Reachability_Audit.json](./O1_8_Handoff_Reachability_Audit.json)

扩预算结果：[O1_8_Handoff_Reachability_Audit_expanded_budget.json](./O1_8_Handoff_Reachability_Audit_expanded_budget.json)

可视化：[O1_8_Handoff_Reachability_对比图.png](./O1_8_Handoff_Reachability_对比图.png)
