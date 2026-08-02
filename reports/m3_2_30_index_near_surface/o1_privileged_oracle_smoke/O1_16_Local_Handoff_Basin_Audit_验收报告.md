# O1.16 Local Handoff Basin Perturbation Audit 验收报告

## 结论

O1.16 已在同一 PhysX 进程内重新搜索 `894/1155/831` 的 CEM plan，然后对每个 plan 执行结构化 reset-state 扰动重放。
共评估 `9600` 个 episode；动作 plan 和扰动状态都未序列化。

| Rank | Bank | Horizon | Nominal | Joint L1 | Joint L2 | Joint L3 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `1155` | 300 | 34.4% | 10.2% | 5.5% | 3.1% | `NARROW_LOCAL_BASIN` |
| 2 | `831` | 500 | 6.2% | 0.8% | 0.0% | 0.0% | `ISOLATED_TRAJECTORY_SOLUTION` |
| 3 | `894` | 400 | 0.0% | 0.0% | 0.0% | 0.0% | `PLAN_RECOVERY_FAILED_BASIN_NOT_MEASURED` |

判定口径：先要求 nominal plan 在本轮至少复现一次；否则标记为 `PLAN_RECOVERY_FAILED_BASIN_NOT_MEASURED`，不推断 basin。复现后，Joint-L1 `>=30%` 为可扩展局部 basin，`5%-30%` 为窄 basin，`<5%` 为近似孤立轨迹解。

本轮得到一个窄 open-loop basin（bank `1155`）、一个近似孤立轨迹解（bank `831`）；bank `894` 的独立 CEM 未恢复 nominal plan，因此其 O1.15 历史可行性仍保留，但本轮 basin 宽度未测得。

## 单因素宽度

### Bank 1155

| Factor | Largest scale @10% | Largest scale @half nominal | L1/L2/L3 |
|---|---:|---:|---|
| `joint_position_rad` | 0.005 | None | 10.9% / 2.3% / 4.7% |
| `joint_velocity_rad_s` | 0.1 | 0.1 | 54.7% / 29.7% / 39.8% |
| `pen_angular_velocity_rad_s` | 0.5 | 0.5 | 39.1% / 32.8% / 32.8% |
| `pen_linear_velocity_m_s` | 0.02 | 0.02 | 49.2% / 53.1% / 47.7% |
| `pen_position_m` | 0.00025 | 0.00025 | 28.9% / 7.0% / 5.5% |
| `pen_rotation_rad` | 0.03490658503988659 | 0.017453292519943295 | 34.4% / 26.6% / 13.3% |
| `target_context_rad` | 0.01 | 0.01 | 24.2% / 21.1% / 18.0% |
### Bank 831

| Factor | Largest scale @10% | Largest scale @half nominal | L1/L2/L3 |
|---|---:|---:|---|
| `joint_position_rad` | None | None | 0.8% / 0.0% / 0.0% |
| `joint_velocity_rad_s` | None | 0.1 | 3.1% / 5.5% / 4.7% |
| `pen_angular_velocity_rad_s` | None | 0.5 | 3.1% / 3.1% / 3.1% |
| `pen_linear_velocity_m_s` | None | 0.02 | 5.5% / 5.5% / 5.5% |
| `pen_position_m` | None | None | 1.6% / 0.8% / 0.8% |
| `pen_rotation_rad` | None | 0.017453292519943295 | 3.1% / 3.1% / 2.3% |
| `target_context_rad` | None | None | 0.0% / 0.0% / 0.0% |
### Bank 894

| Factor | Largest scale @10% | Largest scale @half nominal | L1/L2/L3 |
|---|---:|---:|---|
| `joint_position_rad` | None | None | 0.0% / 0.0% / 0.0% |
| `joint_velocity_rad_s` | None | None | 0.0% / 0.0% / 0.0% |
| `pen_angular_velocity_rad_s` | None | None | 0.0% / 0.0% / 0.0% |
| `pen_linear_velocity_m_s` | None | None | 0.0% / 0.0% / 0.0% |
| `pen_position_m` | None | None | 0.0% / 0.0% / 0.0% |
| `pen_rotation_rad` | None | None | 0.0% / 0.0% / 0.0% |
| `target_context_rad` | None | None | 0.0% / 0.0% / 0.0% |

## Joint-L2 failure taxonomy（仅 nominal plan 已恢复）

| 类型 | Episode |
|---|---:|
| `axis_misaligned` | 49 |
| `throw` | 11 |
| `recovered_strict` | 7 |
| `no_TIM_transition` | 182 |
| `contact_force_overload` | 4 |
| `penetration` | 3 |

## 解释边界

- 这是固定 open-loop plan 的局部鲁棒性下界，不是 closed-loop policy attractor 的体积。
- nominal 为零表示本轮搜索没有恢复 plan，不表示该状态动力学不可达，也不撤销 O1.15 的条件存在性证据。
- `q` 和 `q_target` 同步移动，保留原始控制上下文；`target_context` 另行审计。
- `q/dq/pen state` 的尺度均为逐分量均匀扰动上界；Joint-L1 是多组因素同时扰动，不是单一向量范数半径。
- Contact force 没有被直接设置，而是扰动后经过两步 contact rebuild 重新求解。
- 未训练 PPO，未修改 reward/O0，未生成 teacher 或 reset dataset。

## 产物

- 机器可读结果：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_16_Local_Handoff_Basin_Audit.json`
- 可视化：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_16_Local_Handoff_Basin_Map.png`
- 原始汇总：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_16_Local_Handoff_Basin_Raw.json`
