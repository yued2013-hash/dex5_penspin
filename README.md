<div align="center">

# Dex5 PenSpin · Sim-to-Real

**面向耦合关节灵巧手的强化学习训练、策略蒸馏与真机迁移**

Isaac Lab · Oracle PPO · Student BC · Hardware Calibration

[项目展示](docs/showcase/index.html) · [迁移分析](docs/sim2real_case_study.md) · [面试讲解](docs/interview_guide.md) · [原工程复现](docs/reproducibility.md)

![Dex5真实MuJoCo仿真演示：规划辅助片段](docs/showcase/media/penspin-preview.gif)

*补充MuJoCo物理演示，包含规划辅助与PPO收尾；不是下表Isaac学生策略的整圈输出，也不是真机视频。*

</div>

## 项目目标与路线

将笔状物体的灵巧操作从仿真训练推进到Dex5实机。核心路线是：**仿真特权Oracle强化学习 → 非特权Student策略蒸馏 → 硬件标定与受控回放 → 迁移误差诊断**。

项目重点包括关节耦合建模、接触与奖励设计、数据来源隔离、20维SDK目标到16独立控制通道的适配，以及仿真与实机控制时序对齐。目前已完成短角度仿真学习与学生策略验证，并进行了真机受控回放和局部动作验证；实机连续转笔仍未稳定实现。

## 已取得的成果

|模块|结果|适用范围|
|---|---|---|
|Student仿真闭环|152/197次严格短角度成功，约77.2%|Isaac历史评估，197个episode来自20个测试来源|
|旋转能力|184/197次达到至少30°实际姿态转动|不是整圈或实机成功率|
|学生训练数据|1200条验证轨迹，69,600个transition|按source分组，891/112/197个训练/验证/测试episode|
|硬件适配|20维绝对URDF关节目标，16独立控制通道|保留厂商SDK的URDF到电机转换|
|实机迁移|完成标定、受控回放及局部转动验证|依据原阶段记录；连续转笔稳定性仍不足|

数据来源：[学生闭环评估](Dex5_Sim2Real_S0/student_s0_11/student_closed_loop_eval.json)、[训练报告](Dex5_Sim2Real_S0/student_s0_11/student_training_report.json)、[阶段合同与记录](AGENTS.md)。这些是项目已有实验记录，本轮文档整理未重跑Isaac或真机。

## 方法与工程实现

```mermaid
flowchart LR
  A[Dex5与笔的仿真建模] --> B[特权Oracle强化学习]
  B --> C[姿态与接触校验的轨迹数据]
  C --> D[非特权Student行为克隆]
  D --> E[仿真闭环评估]
  C --> F[受控实机轨迹回放]
  F --> G[关节映射 / 耦合 / 时延诊断]
  G --> H[后续迁移优化]
```

- **学习与数据**：Oracle利用仿真状态学习；Student读取340维可部署观测历史，预测有界20维虚拟动作。按轨迹来源划分数据，避免相邻状态泄漏。
- **评估可信度**：依据相邻四元数的实际姿态变化验证转动；结合目标轴、接触、穿透和150ms稳定保持，避免只积分角速度造成“假转动”。
- **硬件接口**：处理耦合关节、方向与偏置、迟滞、动作插值及反馈监控；逻辑控制量不是电机轴角度，继续通过厂商SDK执行。
- **迁移策略**：先做静态保持和受控回放，再讨论自主闭环与真实数据适配。当前未完成Student实机自主闭环验证。

## 为什么实机效果仍不理想？

实际迁移中存在关节耦合、控制映射、负载下反馈瞬态以及策略频率与SDK指令频率不一致等工程问题。它们是需要联合排查的迁移瓶颈，不能仅凭效果不好就断定“完全由Dex5设置造成”。

已有记录包含局部PIP/DIP响应比约0.29、14–31ms可检测起动延迟、60Hz策略到100Hz指令链，以及负载回放中的独立关节反馈瞬态。这些结果对应特定标定条件，不是硬件的通用极限。详见[实机迁移案例分析](docs/sim2real_case_study.md)。

## 仿真视频与补充研究

[完整仿真视频](docs/showcase/media/baseline-v12.mp4)展示真实MuJoCo接触动力学中的转动、拇指换位和接回。该演示采用规划参考及PPO收尾，与原Isaac Oracle→Student支线分别标记；其几何穿透和单来源限制见[实验边界](docs/results_and_limits.md)。

新增全流程残差PPO作为[补充实验](experiments/rl13/README.md)保留，其两次训练尚未超过基准，不作为原项目迁移成功的证据。

## 展示与复现

```bash
python scripts/serve_showcase.py
```

打开 `http://127.0.0.1:8831/docs/showcase/`，可查看视频、Student结果、实机迁移分析和补充实验。

- 原Isaac环境与复现入口：[跨机器复现](docs/reproducibility.md)、[任务设计](docs/dex5_penspin_reproduction_plan.md)。
- 补充MuJoCo支线：[实验说明](experiments/rl13/README.md)。完整参考回放目前仅Windows认证。
- 面试材料：[90秒开场、工程贡献与常见追问](docs/interview_guide.md)。

历史训练、标定和实机数据独立保留。当前下载副本中部分历史大文件是Git LFS指针，复现原Isaac实验需按原文档获取资产。

## 参考

[Lessons from Learning to Spin “Pens”](https://arxiv.org/abs/2407.18902)及[原代码](https://github.com/HaozhiQi/penspin)提供Oracle→Student与仿真到实机的研究背景。本项目展示Dex5适配和工程验证，不宣称算法首创或已实现论文中的实机多物体多圈效果。
