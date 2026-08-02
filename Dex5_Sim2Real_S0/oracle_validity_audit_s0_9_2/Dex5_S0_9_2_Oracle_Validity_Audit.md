# Dex5 S0.9.2 Oracle Trajectory Validity Audit

## 结论

```text
状态: BLOCKED_NO_TIER_A
总轨迹: 1200
Tier A: 0
Tier B: 79
Tier C: 1121
Student dataset: BLOCKED_NO_TIER_A
真机回放: BLOCKED
```

旧 success flag 只证明角速度积分、接触及保持门通过，不再作为物体旋转成功证据。
本审计只读原始 NPZ，未修改 O0、dataset、reward、policy 或真机状态。

此外，相邻四元数反推的角速度与数据集记录角速度缺乏一致性。新 rollout 在进入
任务筛选前，必须先通过 pose/velocity 运动学一致性检查。

## 核心统计

| 指标 | mean | p95 | max |
|---|---:|---:|---:|
| quaternion final deg | 11.25 | 15.26 | 25.04 |
| target-axis quaternion deg | 1.53 | 4.07 | 10.28 |
| integrated omega-z deg | 40.23 | 44.69 | 53.27 |
| pose-derived/reported omega-z correlation | -0.137 | 0.337 | 0.781 |
| quaternion/reported angular path ratio | 0.332 | 0.462 | 0.706 |

## 分层合同

- Tier A：最终四元数姿态变化 >=30 deg，目标轴一致性 >=0.70，并满足连续 150 ms
  的高度、速度、TIM 接触和 ring/pinky clear 门。
- Tier B：最终四元数姿态变化为 15-30 deg，仅用于课程或诊断，不得进入 student 成功集。
- Tier C：其余旧 success，包括高角速度积分但实际姿态变化不足的 false positive。

## 排名前十五

| episode | tier | net quaternion deg | target-axis deg | axis alignment | integrated omega deg | monotonic efficiency | strict hold ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1188 | B_PARTIAL_15_TO_30_DEG | 25.04 | -0.08 | -0.003 | 39.52 | 0.000 | 0.0 |
| 25 | B_PARTIAL_15_TO_30_DEG | 21.69 | 2.85 | 0.131 | 42.53 | 0.253 | 0.0 |
| 487 | B_PARTIAL_15_TO_30_DEG | 21.29 | 4.06 | 0.191 | 39.94 | 0.246 | 0.0 |
| 1080 | B_PARTIAL_15_TO_30_DEG | 21.26 | 4.45 | 0.209 | 36.70 | 0.337 | 0.0 |
| 753 | B_PARTIAL_15_TO_30_DEG | 19.53 | 3.72 | 0.190 | 37.20 | 0.216 | 0.0 |
| 875 | B_PARTIAL_15_TO_30_DEG | 19.34 | -0.13 | -0.007 | 39.75 | 0.000 | 0.0 |
| 840 | B_PARTIAL_15_TO_30_DEG | 19.13 | 4.89 | 0.256 | 39.44 | 0.335 | 0.0 |
| 272 | B_PARTIAL_15_TO_30_DEG | 18.56 | 4.63 | 0.250 | 37.31 | 0.437 | 0.0 |
| 179 | B_PARTIAL_15_TO_30_DEG | 18.42 | 0.36 | 0.019 | 39.17 | 0.030 | 0.0 |
| 1117 | B_PARTIAL_15_TO_30_DEG | 18.37 | 4.14 | 0.225 | 43.19 | 0.308 | 0.0 |
| 297 | B_PARTIAL_15_TO_30_DEG | 18.09 | 6.74 | 0.373 | 44.13 | 0.317 | 0.0 |
| 870 | B_PARTIAL_15_TO_30_DEG | 17.71 | 5.62 | 0.318 | 43.24 | 0.413 | 0.0 |
| 960 | B_PARTIAL_15_TO_30_DEG | 17.59 | 2.98 | 0.169 | 39.77 | 0.238 | 0.0 |
| 126 | B_PARTIAL_15_TO_30_DEG | 17.57 | 4.63 | 0.263 | 36.71 | 0.411 | 0.0 |
| 803 | B_PARTIAL_15_TO_30_DEG | 17.56 | 2.24 | 0.127 | 37.14 | 0.218 | 0.0 |

## 判定

Tier A 数量不足，因此 student 训练、四档速度包和后续 powered replay 全部保持阻断。
下一步应先修正仿真任务中的旋转状态与成功判据，再采集新的 oracle rollout；不能从
现有 Tier B/C 数据中放宽门槛凑成功样本。
