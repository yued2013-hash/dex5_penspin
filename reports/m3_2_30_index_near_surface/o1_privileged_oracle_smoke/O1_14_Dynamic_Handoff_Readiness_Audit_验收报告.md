# O1.14 Dynamic Handoff Readiness Audit 验收报告

## 结论

O1.14 已完成。对冻结 O0 中的 `396` 条 P3/train 状态施加了同一组
`31` 个固定、对称、最大 `±0.06 rad` 的 TIM 小动作 probe，
并在 `50/100 ms` 记录轴响应、承力响应、高度加速度和接触转换。

预先固定的短窗联合物理门在四个锚点上分离了 996 与三个失败锚点；996 有 `1/30` 个非零 probe 通过，1245/1171/1203 均为 0。这是局部 action-conditioned 证据，不是 400 ms 严格 handoff 的可行性分类器。
全部 P3 中仍有 `37/396` 个状态至少通过一个非零 probe，因此只能形成待后续 CEM 和 clone replay 验证的候选层。

## 四个代表状态

| Bank | O1.12 已知结局 | 轴响应 P90 | 轴对齐 P90 | 交接 margin P90 | 高度 P10 (mm) | TIM 响应 | 不安全 | 动态分数 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `996` | O1.12 strict post success | -0.478 | 0.381 | 16.204 | -1.418 | 54.8% | 0.0% | 0.441 |
| `1245` | axis failure | -0.475 | 0.419 | 11.760 | -0.735 | 51.6% | 0.0% | 0.460 |
| `1171` | axis failure | -0.349 | 0.351 | 22.565 | -0.667 | 54.8% | 0.0% | 0.426 |
| `1203` | transition replay failure | -0.907 | 0.318 | 15.834 | -1.552 | 54.8% | 0.0% | 0.410 |

`996` 在全部 P3 中的连续动态响应分数排名为 `143/396`，它没有靠连续分数单独脱颖而出。
按非零 short-admissible probe 数量排名时，996 并列第 `5`。
两种分数都不是成功概率，也不能用来修改 P3 reset 分布。

## 边界

- 未训练 PPO，未修改 reward，未生成 teacher。
- O0 bank 前后 SHA-256 均为 `73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`。
- Probe 是预先固定的通用小扰动，不是 CEM 优化轨迹。
- 本轮只能用于提出后续独立 parent/source 的 CEM 候选，不能筛选、替换或重标 O0 P3。

## 产物

- 原始响应：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_14_Dynamic_Probe_Responses.npz`
- 机器可读结论：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_14_Dynamic_Handoff_Readiness_Audit.json`
- 动态图：`/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/O1_14_Dynamic_Readiness_Map.png`
