# Dex5 S0.9.4 Rollout Data Closure

## 结论

状态：`PASS_EXPANDED_COLLECTION`

成功终止帧记录已经修复，pose-delta angular velocity 已成为数据集唯一规范
运动通道。在 64 episode 回归中，环境成功数和离线严格重建均为 `49`；随后
完成 1200 条严格成功轨迹扩采，环境与离线审计均为 `1200/1200`。

## 终止帧修复

Isaac Lab 的 `DirectRLEnv.step()` 会在返回前自动 reset 已完成环境。旧 collector
因此把下一 episode 的 reset 状态误写为当前 transition 的 `next` 状态。

当前实现会在环境 `_reset_idx()` 修改 PhysX 状态之前保存：

- pen pose、linear velocity 和 instantaneous PhysX angular velocity；
- contact state、contact wrench 和 contact-force norm；
- 120 Hz physics quaternion/angular-velocity substep samples。

collector 对完成 transition 使用该 pre-reset snapshot。每个动作 transition 现在
都有属于同一 episode 的 before/after 状态，episode 边界只由 offsets 表达。

## 运动字段合同

```text
pen_angular_velocity
    canonical downstream field
    == quaternion finite-step pose-delta velocity

pen_angular_velocity_pose_delta
    explicit canonical alias

pen_angular_velocity_physx_instantaneous
    contact/impact diagnostic only
```

`pen_angular_velocity` 与 `pen_angular_velocity_pose_delta` 已逐元素验证完全相等，
最大差值为 `0.0`。原始 PhysX 角速度不再占用规范字段名。

## 64 Episode 回归

| 项目 | 结果 |
|---|---:|
| 环境 corrected success | 49/64 |
| 离线 150 ms hold 重建 | 49/64 |
| pre-reset terminal snapshot | 64/64 |
| before/after transition 完整 | 100% |
| dataset authorized | true |

旧 collector 的环境/离线成功计数差异已消失。

## 1200 条扩采

| 项目 | 结果 |
|---|---:|
| 严格成功轨迹 | 1200/1200 |
| 实际目标轴旋转 >=30 deg | 1200/1200 |
| 150 ms strict hold | 1200/1200 |
| terminal snapshot | 1200/1200 |
| transition rows | 69,600 |
| split leakage | 0 |
| train/validation/test | 891/112/197 |
| unique sources | 125/16/20 |
| 中位最大旋转 | 35.18 deg |
| P95 最大旋转 | 38.38 deg |
| dataset authorized | true |

规范数据字段均通过 finite、shape、offset、success、hold 和 split 完整性断言。
独立 `validate_s0_oracle_dataset.py` 检查结果为 `PASS`，包括 20D/16D
耦合映射精确重建、动作边界和三分割完整性。

## 诊断边界

瞬时 PhysX 角速度与四元数差分仍不一致：

```text
120 Hz omega-z correlation: 0.795
120 Hz omega RMSE: 1.759 rad/s
angular path ratio: 0.359
```

因此 `instantaneous_velocity_diagnostic_pass=false`。这不再阻断数据集，因为该
字段已明确降级为诊断量；任何训练、筛选和 replay 生成代码都必须读取规范
pose-delta 通道。powered replay 与 student 训练仍未在本阶段启动。

## 产物

- `quaternion_success_rollouts_1200.npz`
  - SHA256: `1a1a068bee196e828f787ee3926c419d756cfa34d4059d48fb3897844a0795a4`
- `quaternion_success_rollouts_1200.report.json`
  - SHA256: `942735fe924ef5cb483cbee980fc56a8140ff83ecca07004dac77f74cd9a1b3b`
- `quaternion_success_rollouts_1200.audit.json`
  - SHA256: `e74fb5f4cf7cafb09ae14176855250f5fe12a8a559b29e6cdad645c17d1f5d48`
- `quaternion_success_rollouts_1200.validation.json`
  - SHA256: `b255102b87cb0d0d3b8394ea148b181b9b4d29e61dba40ca785f259f3acf359c`
