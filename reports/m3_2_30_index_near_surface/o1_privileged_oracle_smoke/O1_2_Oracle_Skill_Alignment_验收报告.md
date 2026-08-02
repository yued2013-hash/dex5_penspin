# O1.2 Oracle Skill Alignment 验收报告

## 结论

**O1.2 工程与对照实验：PASS。Reliable finger gait 验收：FAIL。**

O1.2 证明了简单放大 axis reward 不能解决问题。当前最有价值的方向是 P3-only 探索，它已将 ballistic 压低到 `0.20%`，并产生了 middle 指的持续 break/re-contact；但因拇指全部释放、掉笔和轴向不足，qualified gait 仍为 `0%`。

## 实验设置

- 单 seed：`20260730`
- 本机 RTX 5060 Laptop
- `256` 个并行环境
- A/B/P3-only：各 `30` iterations
- P3-only penalty recovery 与 P3Axis：各 `60` iterations
- 确定性评估：`256 env x 240 steps`
- O0 bank SHA-256：`73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`

共用 shaping：有任意有效接触时才给正向 rotation reward，加入 axis-aligned rotation、linear speed、wrong-axis、high-angular-speed 和 ballistic penalty。未加入指定手指接触模式或 P1-P5 waypoint reward。

## 对照结果

| 候选 | Raw gait | Qualified gait | Raw 180 deg | Failure | Ballistic | Axis alignment | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| O1.1 baseline | 29.16% | 0.00% | 72.99% | 60.64% | 46.51% | 约 0.37-0.44 | 转动 exploit |
| Aligned A, strict guard | 0.00% | 0.00% | 0.00% | 100% | 85.54% | 不合格 | 过早终止 |
| Aligned B, strict guard | 0.09% | 0.00% | 0.00% | 100% | 91.10% | 不合格 | 过早终止 |
| P3-only, strict guard | 13.70% | 0.00% | 12.75% | 89.15% | 59.70% | 0.478 | 有弱迹象，仍失稳 |
| **P3-only penalty60** | **35.16%** | **0.00%** | **99.80%** | **54.88%** | **0.20%** | **0.553** | **当前最佳诊断候选** |
| P3Axis60 | 3.40% | 0.00% | 0.00% | 100% | 96.95% | 0.571 | 轴向 reward 被再次 exploit |

## 最佳候选的真实行为

`P3-only penalty60` 的 512 个完成 episode 中：

- middle break/re-contact gait：`178/512 = 34.77%`
- raw 180 deg：`511/512 = 99.80%`
- ballistic：`1/512 = 0.20%`
- contact loss：`3/512 = 0.59%`
- penetration failure：`0`
- 最大穿透：`0.232 mm`
- drop：`278/512 = 54.30%`
- thumb break：`512/512 = 100%`，无 re-contact
- 平均 directional axis alignment：`0.553`，未达 `0.70`
- qualified gait：`0/512`

因此这不是合格 finger gait，但它已将上一阶段的 ballistic exploit 与真正的 middle 接触切换分离开。

## 失败的轴向加强实验

P3Axis60 增加了显式 axis-quality reward 和高度下降惩罚，但策略选择高速脱离接触：ballistic `96.95%`、contact loss `96.69%`。这说明不应继续单独放大 axis reward。

## 下一瓶颈

下一步应针对两个行为约束，而不是再加训练步数：

1. 拇指支撑连续性：允许换指，但避免 thumb 在全部 episode 中只断不回。
2. 笔 COM/高度可恢复性：避免直接使用可被高速飞行利用的常数 dense reward，改用分段安全约束或 constrained objective。

在 qualified gait `>0` 之前，不进入多 seed、DR、5090/H200 或长训练。

## 证据

- `o1_2_P3Only_penalty60_final_eval256.json`
- `o1_2_P3Only_penalty60_final_eval256.md`
- `o1_2_P3Axis60_final_eval256.json`
- `o1_2_P3Axis60_final_eval256.md`
- `O1_2_Oracle_Skill_Alignment_验收结果.json`
