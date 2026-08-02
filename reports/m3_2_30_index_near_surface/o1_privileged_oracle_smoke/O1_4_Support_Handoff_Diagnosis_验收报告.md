# O1.4 Support Handoff Diagnosis 验收报告

## 结论

```text
O1.4 诊断基础设施：PASS
O1.2/O1.3 handoff 技能：FAIL
当前主瓶颈：承力接力的时序与方向，不是 P3 reset 立即失稳
```

O1.2 和 O1.3 不是同一种失败。O1.2 过早卸掉拇指，食指/中指虽仍接触，但没有形成可恢复高度的有效承力；O1.3 则让 TIM 支撑近似同步塌陷，之后的拇指/食指重接触已经太晚。

## 对照设计

- 三组均从 P3 启动，每组 `256` episodes、`120` 控制步（`2 s`）。
- 使用相同 seed，覆盖相同的 `196` 个 reset bank rows。
- 接触事件需连续 `3` 步才确认，排除单步拖动。
- 诊断时关闭普通 episode termination，固定观察到 `2 s`；这些 NPZ 只是诊断轨迹，不是 teacher/reset 数据。
- 未修改 O0，未修改 reward，未新建 teacher，未进行 PPO 训练。

## 核心结果

| 指标 | Zero-action P3 | O1.2 penalty60 | O1.3 P3State60 |
|---|---:|---:|---:|
| 5 mm 高度丢失 | 0% | 100% | 100% |
| 最低高度变化均值 | -0.37 mm | -48.52 mm | -26.84 mm |
| 首次 5 mm 下降中位时刻 | 无 | 56 步 / 0.93 s | 10 步 / 0.17 s |
| TIM 至少两指重叠比 | 100% | 82.49% | 11.81% |
| 无 TIM 支撑最长区间中位数 | 0 | 0 | 77 步 / 1.28 s |
| thumb break / recontact | 0% / 0% | 100% / 0% | 100% / 79.30% |
| middle break / recontact | 0% / 0% | 70.70% / 51.56% | 100% / 10.55% |
| ballistic | 0% | 6.25% | 65.23% |

Zero-action 对照组在同一 P3 分布上 `256/256` 稳定，TIM 重叠比为 `100%`。因此，当前失败不能归因为“P3 reset 本身不可复现”。

## 承力与 Wrench

### O1.2：有接触，但没有可恢复的支撑

- thumb 在中位第 `2` 步断开，`100%` episodes 不再重接触。
- 高度下降 5 mm 前后，TIM 力幅中位数为 `0.161 -> 0.171 N`，没有瞬时归零。
- 但竖直净力代理仅 `0.0445 -> 0.0488 N`，明显低于 zero-action 稳态时约 `0.31 N` 的中位水平。
- 所以它不是“突然没有接触”，而是拇指卸载后，index-middle 夹持的承力方向/时序无法抵消下落。

### O1.3：TIM 承力塌陷，重接触过晚

- 首次 5 mm 下降前后，TIM 力幅中位数从 `2.292 N` 降到 `0.0459 N`。
- index 力从 `1.140 N` 降到 `0`，middle 从 `1.155 N` 降到 `0.030 N`。
- wrong-axis 角速度中位数反而从 `5.21` 升到 `6.20 rad/s`。
- thumb/index 虽然分别有 `79.30%/96.09%` 重接触，但高度没有恢复；这证明“重新碰到”不等于“完成承力接力”。

> Wrench 为指尖传感器力及指尖力臂构造的代理量，用于同配置横向对比，不应解释为精确重建的笔体接触 wrench。

## Handoff Failure Taxonomy

| ID | 失败类型 | 证据 | 主要出现于 |
|---|---|---|---|
| H1 | 过早拇指卸载 | thumb 第 2 步左右断开 | O1.2, O1.3 |
| H2 | 双指接触但有效支撑 wrench 不足 | TIM 接触重叠较高，仍持续下落 | O1.2 |
| H3 | TIM 同步承力塌陷 | TIM 力幅约下降 98% | O1.3 |
| H4 | 迟到的重接触，无高度恢复 | recontact 存在但 5 mm 高度门不恢复 | O1.2, O1.3 |
| H5 | 抛射/错轴运动 | ballistic 65.23%，wrong-axis 升高 | O1.3 |

## 三种假设的判定

| 假设 | 判定 | 理由 |
|---|---|---|
| handoff 动力学不可达 | **当前证据不支持** | 同分布 zero-action 稳定，且已有局部 P3->P4 可行控制证据；但本诊断不构成全局可达性证明。 |
| reward/时序信用不足 | **当前首要解释** | 两个策略都可通过“先卸拇指”获得旋转，但没有学到支撑重叠和高度恢复。 |
| reset distribution 不足 | **不支持为主因** | 同一批 P3 reset 零动作 `256/256` 稳定 2 s。 |

此处的“reward/时序信用”是最有力的当前解释，但还不能严格排除 policy representation 和 PPO optimization 的贡献。O1.4 只完成诊断，未用调 reward 来反向“证明”结论。

## 产物

- `o1_4_handoff_zero_P3_256x120.json/.npz`
- `o1_4_handoff_O1_2_penalty60_256x120.json/.npz`
- `o1_4_handoff_O1_3_P3State60_256x120.json/.npz`
- `O1_4_Support_Handoff_中位时间序列.png`
- `O1_4_Support_Handoff_Diagnosis_验收结果.json`

下一阶段若修改 objective，验收不能只看 contact/recontact；必须联合要求“支撑重叠 + 竖直承力 + 高度恢复 + 非 ballistic”。
