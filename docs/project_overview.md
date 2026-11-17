# Dex5 PenSpin：方法与工程实现

围绕Dex5灵巧手的笔操作，组织仿真强化学习、数据与蒸馏、硬件适配和动作诊断四个模块。

| 模块 | 代码入口 | 实现内容 |
| --- | --- | --- |
| 仿真学习 | `isaac/` | Oracle PPO、奖励、耦合约束及姿态评估 |
| 数据与蒸馏 | `sim2real/`、`Dex5_Sim2Real_S0/` | 四元数轨迹校验、来源隔离、340维Student观测、行为克隆 |
| 硬件迁移 | `sim2real/dex5_hardware_adapter.py` | 20维SDK目标、16独立通道、方向、偏置、迟滞与时序 |
| 动作诊断 | `sim/`、`experiments/rl13/`、`docs/showcase/` | MuJoCo回放、接触分析、残差控制与实验可视化 |

## 训练与数据流程

Isaac Lab中的特权Oracle使用笔位姿和接触信息学习短角度控制；保存的轨迹经过实际四元数转角、轴对齐、接触与保持联合校验。Student从验证轨迹学习340维观测历史到20维有界动作的映射。

数据按 `source_trajectory_id` 分组划分，包含1,200个episode和69,600个状态转移，训练/验证/测试分别为891/112/197个episode。闭环测试取得152/197次严格短角度成功；训练与评估记录见[数据统计](data_statistics.md)。

## 硬件适配

20维绝对URDF关节目标经厂商SDK转换，关联16个独立控制通道。适配层保存关节方向、偏置、分方向分段逆映射及动作历史，处理60Hz策略与100Hz指令链。实机工作覆盖空手标定、静态负载保持、受控轨迹前缀执行及反馈瞬态分析，详见[实机迁移](sim2real_case_study.md)。

## MuJoCo回放与控制实验

`sim/render_s0_oracle_trajectory_gallery_mujoco.py`实现保存的Isaac关节状态和笔位姿到MuJoCo模型的状态回放渲染，按数据版本与SHA-256选择轨迹。

`experiments/rl13/`使用MuJoCo物理规划前缀与PPO收尾生成的参考进行连续动力学回放，并开展有界残差PPO对照。模块保存独立的关节顺序、观测、控制频率、增益与耦合参数；MuJoCo耦合设置约0.829，实机局部辨识值约0.29。

## 统一入口

```bash
python scripts/project.py showcase
python scripts/project.py evidence
python scripts/project.py audit
```

展示页集中查看仿真视频、Student结果、标定记录、残差幅度对照与训练曲线。数据、模型、配置和原始报告沿实验路径归档。
