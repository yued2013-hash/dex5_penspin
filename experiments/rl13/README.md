# RL13 · 全流程残差强化学习

独立、仅仿真的 MuJoCo 支线。原项目 `isaac/`、`sim2real/`、O0/S0 数据不被本模块导入或修改。

当前结果：已经运行两次各32,768步的独立PPO训练，共65,536步。两种残差幅度都没有超过整圈基准。训练权重与失败日志完整保留；推荐展示使用参考辅助基准，不能称为已成功训练的端到端全程策略。

## 运行

建议独立虚拟环境，已在 Windows / Python3.14 / CPU 验证：

```bash
pip install -r experiments/rl13/requirements.txt
python -m experiments.rl13.validate_reference
python -m experiments.rl13.evaluate --tag baseline
python -m experiments.rl13.train
python -m experiments.rl13.train_micro
python -m experiments.rl13.export_showcase
python scripts/serve_showcase.py
```

仓库包含已有训练记录。训练入口会拒绝覆盖已有同名日志；如需重训，复制该实验模块到独立工作副本，并为结果建立新的版本目录。不要删除当前记录后冒充同一轮实验。

重放已经训练的候选：

```bash
python -m experiments.rl13.evaluate --checkpoint experiments/rl13/checkpoints/ppo_best.zip --tag rerun_large
```

小幅度候选需使用 `residual_limit=0.00006`，Python入口：

```python
from stable_baselines3 import PPO
from experiments.rl13.evaluate import evaluate
model = PPO.load('experiments/rl13/checkpoints/micro_best.zip')
report = evaluate(model, 'rerun_micro', residual_limit=0.00006)
```

## 设计

- 参考：V12连续物理回放，共2,114步、42.28秒，仅1个轨迹来源。
- 动作：16维，参考目标加有界PPO残差，低通alpha=0.15；残差上限消融为0.006与0.00006 rad。
- 观测：101维，包含关节状态、模拟笔状态、相位、参考控制、残差历史及接触力；这是特权仿真策略，不能直接部署到无笔状态感知的实机。
- 训练：4个环境，256步rollout，128×128网络，同种子20261101，同32,768步预算；一半窗口从真实初态开始，另一半从通过验证的阶段起点开始。
- 评估：始终从真实初态连续积分，中途不切换/重置状态；不把训练窗口成功当整圈成功。
- 未验证：跨来源测试、跨物体泛化、随机初态鲁棒性、实机部署。

## 结果解释

`success` / `behavior_success` 是整圈及终态夹持门槛；`strict_success` 还要求全程穿透≤0.5mm。初始训练日志产生于增加独立几何质量字段之前，其行为结果原样保留，严格质量解释见 `results/quality_audit.json`，不要只读取一个旧布尔值。

当前基准观察到365.69°整圈与末段夹持，但全程最大穿透2.68mm，严格质量失败。大幅度所选模型运行3.96秒、最大106.13°后失效；小幅度所选模型运行4.02秒、最大112.65°后失效。因此没有把新策略晋级成推荐演示策略。

状态校验为本MuJoCo模型专用：检查有限数值、完整积分状态、50步保存控制延续、10步有界动作响应。它不是PhysX接触恢复认证，也不是随机重置分布验证。原项目冻结的Isaac数据仍按其原始R0–R3规则管理。

训练前v1检查报告原样保存在 `reset_validation_pretraining_v1.json`。训练后补充的v2几何审计明确使用初态穿透≤1mm、软关节限位偏差≤0.01rad的仿真诊断范围，10个起点仍全部通过。这些起点门槛不同于全程≤0.5mm严格质量门槛，也不构成硬件许可。

方法、奖励与研究边界见 [方法文档](../../docs/rl13_method.md)，面试叙事见 [讲解提纲](../../docs/interview_guide.md)。
