# Dex5-PenSpin

目标：复现并适配《Lessons from Learning to Spin "Pens"》的核心方法，在 Dex5 真机上实现笔状物体连续多圈稳定转笔。

主任务书：

- [Dex5-PenSpin 复现任务书](docs/dex5_penspin_reproduction_plan.md)
- [跨机器安装与逐阶段复现](docs/reproducibility.md)

## 快速复现

```bash
git clone https://github.com/yued2013-hash/dex5_penspin.git
cd dex5_penspin
./bootstrap.sh
./scripts/reproduce.sh all
```

环境固定为 Isaac Sim `5.1.0.0`、Isaac Lab commit
`b4c321024792976150ca55fddb26fa34480d974e`、PyTorch
`2.7.0+cu128` 和 skrl `2.1.0`。最小 checkpoint/rollout 包由 Git LFS
管理，bootstrap 会安装用户级 Git LFS 并拉取资产。

## 工程目录

```text
dex5_penspin/
├── assets/          # 图片、视频、标定图、论文图示摘录
├── artifacts/       # Git LFS 管理的最小复现 checkpoint/rollout
├── configs/         # 任务、仿真、训练、真机参数配置
├── data/
│   ├── raw/         # 原始真机/仿真轨迹
│   └── processed/   # 转换后的训练数据
├── docs/            # 任务书、读论文笔记、实验规范
├── models/          # policy checkpoint
├── real/            # Dex5 真机采集、replay、执行接口
├── reports/         # 阶段实验报告
├── scripts/         # 一键运行脚本
└── sim/             # Dex5 / pen 仿真环境与训练代码
```

## 当前阶段

先完成任务定义与工程设计，再进入仿真环境建设：

```text
Dex5 MDP 定义
-> 数据格式
-> 安全边界
-> 仿真 oracle
-> student policy
-> 真机 open-loop replay
-> real fine-tuning
```
