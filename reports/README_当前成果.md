# Dex5 转笔报告目录

## 当前正式成果

- `m3_2_30_index_near_surface/P1到P5可靠数据模型可视化总汇_v2/`
  P1-P5 全流程入口、可靠性等级、代表动态图和证据索引。
- `m3_2_30_index_near_surface/当前任务可靠数据模型总览_v1/`
  当前任务、双入口、严格评测和跨引擎风险摘要。
- `m3_2_30_index_near_surface/canonical_p1_p5_v9_P4拇指下托笔端/`
  当前 P1-P5 canonical 姿态。
- `m3_2_30_index_near_surface/P3与P3到P4训练结果_v9正式评估/`
  当前 P3 保持与 P3 到制动 P4 的 PhysX 评测。

## 可继续训练的数据

- `m3_2_30_index_near_surface/p4_p5_curriculum/`
  P4 到 P5 teacher、learned cache、策略评测和扰动课程。
- `m3_2_30_index_near_surface/p5_p1_curriculum/`
  P5 到 P1 后缀训练数据。
- `m3_2_30_index_near_surface/no_reset_chain/`
  后半圈无重置缓存、phase6-18 数据与严格复核。
- `manus_mujoco_presets/short_primitive_reward_m3_2_23/`
  P1 到 P2/B23 早期动态入口 cache。

## 待确认

- 当前 P1 到 P2 接触语义重建。
- 当前 P2 到 P3 连续 PhysX 严格验证。
- 拇指彻底越侧后的对侧拇食指夹持。
- phase18 中指继续撤回与最终 P1 保持。
- 当前语义 P1 到 P5 到 P1 整圈无重置闭环。

## 保留历史参考

- `manus_mujoco_presets/p1_p2_micro_rebuild_m3_2_18/`
- `manus_mujoco_presets/b12_ring_under_recapture_m3_2_32/`
- `manus_mujoco_presets/narrow_021_five_step_chain_m3_2_33/`
- `manus_mujoco_presets/continuous_p1_p2_p3_m3_2_34/`
- `m3_1_hold_oracle/` 与 `m3_1_hold_oracle_smoke/`

这些文件只作为 teacher、姿态语义或失败诊断参考，不代表当前严格门成功。

## 清理记录

- `清理记录_2026-07-22/待删除文件.json`：执行前固定清单。
- `清理记录_2026-07-22/执行结果.json`：实际删除数量和大小。

本次采用保守清理：保留正式评测、teacher/reset/cache、当前失败边界、源码和策略日志；删除旧 canonical、重复 smoke、筛查副本和已被正式结果替代的探索目录。
