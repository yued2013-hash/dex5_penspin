# Dex5 S0.10 Student Dataset Spec

## 状态

```text
S0.10 Student Data Contract and BC Preparation: PASS
Dataset audit: PASS
BC interface smoke: PASS
BC training: NOT RUN
Hardware control: NOT RUN
```

本阶段只建立可部署 student 的数据与模型接口。没有修改 oracle、O0 或四元数成功定义，也没有执行反向传播、优化器更新、checkpoint 写入或真机控制。

## 输入来源

- 输入：`quaternion_validated_rollouts_1200_v2.npz`
- SHA256：`50621081e145675912de565d2097bdc8b9a439cebb4e2868ee5fb84d5fdc61cd`
- 轨迹：1200 条，全部通过既有 quaternion-consistent success gate
- transition：69600 条
- 数据授权标志：`dataset_authorized=true`

旧的非 `v2` 文件不属于 S0.10 输入。`v2` 在同一 PhysX step 同步保存 student observation，并为 episode 终止 transition 保存自动 reset 之前的 `next_obs`。

## Transition Contract

每条样本为：

```text
(student_obs, action, next_obs, success)
```

| 字段 | 形状 | 语义 |
|---|---:|---|
| `student_obs` | `[340]` | 当前 4 帧可部署观测历史 |
| `action` | `[20]` | 范围 `[-1, 1]` 的 virtual normalized delta action |
| `next_obs` | `[340]` | 执行动作后的下一观测；终止帧使用 pre-reset snapshot |
| `success` | `[1]` | 仅成功 episode 的最后一个 transition 为 true |

数据另含 `terminal`、`episode_id`、`source_trajectory_id`、split 与 episode offsets。它们只用于训练采样、序列边界和审计，不属于部署 observation。

## Student Observation

单帧为 85D，按 newest-to-oldest 堆叠 4 帧，总计 340D：

| 单帧字段 | 维度 | 真机来源 |
|---|---:|---|
| joint position | 20 | Dex5 joint feedback |
| joint velocity | 20 | Dex5 joint feedback |
| previous virtual action | 20 | deployment adapter 内部状态 |
| previous absolute target | 20 | deployment adapter 内部状态 |
| fingertip force | 5 | Dex5 force channel；标定仍沿用硬件验收约束 |

明确禁止进入 student observation 的 privileged 信息包括：pen pose/velocity、PhysX instantaneous angular velocity、oracle observation、质量、摩擦、质心和 world-frame contact wrench。

## Action Contract

Student 输出 20D virtual normalized delta action，经现有硬件兼容链执行：

```text
20D virtual action
  -> virtual target integration and limits
  -> PIP/DIP coupling projection
  -> 16D logical actuator target
  -> 20D SDK absolute command representation
```

成品数据最大绝对 action 为 `1.0`。适配器耦合误差和 20→16→20 重构误差均为 `0.0 rad`。

## Split Contract

split 按 `source_trajectory_id` 完成，禁止逐行随机拆分：

| Split | Episodes | Rows | Unique sources |
|---|---:|---:|---:|
| train | 891 | 51621 | 125 |
| validation | 112 | 6519 | 16 |
| test | 197 | 11460 | 20 |

source trajectory 跨 split 泄漏数为 `0`。观测均值和标准差只由 train rows 计算，标准差下限为 `1e-6`。

## 审计结果

- 字段严格白名单：PASS
- privileged field name 与 observation schema 泄漏：0
- 非终止 `next_obs == 下一行 student_obs`：最大误差 `0.0`
- `next_obs` 最新帧 previous action 与当前 action 一致：PASS
- 1200 个 terminal，1200 个 success，success 仅出现在 terminal：PASS
- finite、shape、action bounds、normalization 重算：PASS
- dataset SHA256 与 manifest：PASS

独立审计见 `student_dataset_validation.json`。

## BC Interface

当前入口定义一个 `340 -> 256 -> 256 -> 20` 的 MLP，隐藏层使用 ELU，输出使用 tanh。固定种子下，仅运行 256 条 train 和 validation 样本的 forward 与 MSE loss：

```text
train MSE smoke:      0.0653499141
validation MSE smoke: 0.0710152388
parameter count:      158228
```

这些 loss 来自随机初始化模型，只证明数据加载、归一化、模型 forward 和 loss 接口可用，不表示 student 已获得任何技能。

## 产物

- `student_bc_dataset.npz`: BC 数据集
- `student_dataset_manifest.json`: 机器可读 manifest
- `student_dataset_validation.json`: 独立审计
- `bc_smoke_report.json`: forward/loss smoke
- `sim2real/build_s0_10_student_dataset.py`: 数据构建入口
- `sim2real/validate_s0_10_student_dataset.py`: 独立验证入口
- `sim2real/smoke_s0_10_bc.py`: BC 接口 smoke

## 阶段边界

S0.10 完成不授权正式 BC 训练。下一阶段若要训练，必须单独批准 optimizer/backward/checkpoint，并继续保持 source-level split、部署观测白名单和现有 quaternion success definition 不变。
