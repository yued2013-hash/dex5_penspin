# 残差PPO实验报告

## 设置

MuJoCo / SB3 PPO，CPU，种子20261101，4个并行环境；128×128网络，rollout 256，batch 256，5个优化epoch，学习率0.0001，gamma 0.995，训练窗口512步。两次运行各32,768步，对比残差上限。

| 控制器 | 训练交互步 | 所选检查点步数 | 最大实际转角 | 执行时长 | 穿透峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 混合参考，零残差 | — | — | 365.69° | 42.28s | 2.676mm |
| 大幅度残差，0.006 rad | 32,768 | 32,768 | 106.13° | 3.96s | 2.252mm |
| 小幅度残差，0.00006 rad | 32,768 | 8,192 | 112.65° | 4.02s | 3.314mm |

参考辅助基准的末5秒对向夹持比例为100%。转角、终态保持和几何质量分别记录，完整字段见各评估JSON。

## 保存与复核

[实验档案](../experiments/rl13/runs/README.md)保存配置、原始报告、检查点清单与哈希索引；状态和动作数据沿原实验路径归档。

```bash
pip install -r experiments/rl13/requirements.txt
python -m experiments.rl13.validate_reference
python -m experiments.rl13.verify_replay
```

Windows执行完整参考回放复核；Linux CI检查接口与同平台状态恢复。两种残差幅度的检查点回报可在[展示页](showcase/index.html#results)中查看。
