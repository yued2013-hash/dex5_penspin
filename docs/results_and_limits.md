# 实验结果与评估记录

| 实验 | 测量结果 | 设置与来源 |
| --- | --- | --- |
| S0.11 Student闭环 | 152/197次严格短角度成功；184/197次达到30° | Isaac Lab，20个测试来源；[评估报告](../Dex5_Sim2Real_S0/student_s0_11/student_closed_loop_eval.json) |
| Student BC | 验证MSE 0.000780，测试MSE 0.000654 | [训练报告](../Dex5_Sim2Real_S0/student_s0_11/student_training_report.json) |
| 混合参考零残差 | 最大365.69°，42.28s，末5秒对向夹持100% | 单参考、固定初态、MuJoCo；[基准报告](../experiments/rl13/results/baseline.json) |
| 基准接触几何 | 全程穿透峰值2.676mm | [几何诊断](../experiments/rl13/results/quality_audit.json) |
| 大残差PPO | 训练32,768步；所选检查点32,768步；106.13°、3.96s、穿透峰值2.252mm | 上限0.006 rad；[报告](../experiments/rl13/results/training_report.json) |
| 小残差PPO | 训练32,768步；所选检查点8,192步；112.65°、4.02s、穿透峰值3.314mm | 上限0.00006 rad；[报告](../experiments/rl13/results/micro_training_report.json) |

## 评估实现

- 转角使用连续四元数姿态差分，记录目标轴变化、最大转角与最终姿态。
- 接触统计包括对向夹持、持续至少3个控制步的消抖重接触和有效支撑交接。
- 终态夹持采用末5秒时间窗；物理质量单独记录全程穿透峰值及0.5mm质量门槛。
- Student闭环按来源隔离测试；MuJoCo参考与残差实验使用同一固定初态进行配对比较。
- 每个实验保存环境、模型参数、种子、训练预算、检查点和结果文件。

## 复核环境

完整MuJoCo参考回放的认证环境为Windows。CI分别执行Windows参考回放检查及跨平台接口、同平台状态恢复检查。`verify_replay.py`用于同平台逐元素复核；OS数值延续差异记录在实验档案中。

## 查看结果

播放[参考辅助视频](showcase/media/full-turn-demo.mp4)，在[交互展示](showcase/index.html#results)切换Student、真机适配、动作诊断和残差PPO；曲线可查看实际转角、对向夹持状态及两种残差幅度的检查点评估回报。
