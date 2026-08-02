# O1.7 Termination-Invariant Learning Ablation 验收报告

## 结论

```text
O1.7 实验管线：PASS
Termination-return loophole：CLOSED
Reliable finger gait：FAIL
分类：TERMINATION_CREDIT_NOT_SUFFICIENT
```

O1.7 证明 termination credit assignment 是 O1.6 “早死少吃惩罚”投机的原因，但它不是 P3 承力交接失败的充分解释。关闭该漏洞后，三组策略均能运行完整时域，但 qualified gait、严格 TIM support handoff 和 height recovery 仍全部为零。

## 实验契约

| 项目 | A Fixed horizon | B Fixed + terminal penalty | C Recovery grace |
|---|---:|---:|---:|
| Reset | P3-only | P3-only | P3-only |
| Observation | 172D temporal | 172D temporal | 172D temporal |
| Baseline reward | 保持 | 保持 + 一次 terminal `-20` | 保持 |
| Seed | 20260730 | 20260730 | 20260730 |
| Envs | 256 | 256 | 256 |
| PPO iterations | 60 | 60 | 60 |
| Training transitions | 368,640 | 368,640 | 368,640 |
| Evaluation transitions | 61,440 | 61,440 | 61,440 |
| Early termination | 禁用 | 禁用 | 失败持续 12 步后 |

三组使用同一 O0 bank：

```text
SHA-256: 73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65
```

无 teacher、无 contact-pattern reward、无接触顺序监督，未扩大训练。

## 训练前回报顺序审计

| 环境 | Zero action | Random | Intentional release | 顺序 |
|---|---:|---:|---:|---|
| A | -18.20 | -152.21 | -664.48 | PASS |
| B | -18.20 | -160.34 | -684.48 | PASS |
| C | -18.91 | -84.40 | -84.19 | PASS |

A/B 三类控制均运行 `119` 步。C 的 intentional release 约 `17.24` 步后终止，但回报仍显著低于 zero action，因此“主动提前失败更划算”的梯度方向已被关闭。

## 正式评估

每组使用 `256 env x 240 control steps`，完成 `512` 个确定性 episode。

| 指标 | A Fixed | B Terminal | C Grace |
|---|---:|---:|---:|
| 平均 episode 步数 | 119.0 | 119.0 | 119.0 |
| Raw gait | 0.59% | 0.00% | 1.56% |
| Qualified gait | 0.00% | 0.00% | 0.00% |
| Strict TIM handoff | 0.00% | 0.00% | 0.00% |
| Height drop >=5 mm | 0.00% | 0.00% | 100.00% |
| Height recovery | 0.00% | 0.00% | 0.00% |
| Thumb recontact | 0.00% | 0.00% | 1.56% |
| TIM overlap | 100.00% | 100.00% | 100.00% |
| Ballistic | 0.00% | 0.00% | 0.59% |
| Contact loss | 0.00% | 0.00% | 0.00% |
| Failure | 0.20% | 0.00% | 0.20% |
| 方向轴对齐 | 0.149 | 0.351 | 0.415 |
| 最大穿透 | 0.187 mm | 0.187 mm | 0.597 mm |

## 行为解读

### A/B：稳定但没有交接

A/B 不再 ballistic 或接触丢失，但两者都是 thumb 很早释放、index+middle 继续全程支撑，thumb recontact 为零。A 的少量 raw gait 来自 ring/pinky 接触抖动，不是 TIM 承力交接。

因此 TIM overlap 约 100% 只表示 index+middle 没有断开，不代表完成了 finger gait。

### C：出现重接尝试，但高度不可恢复

C 的 thumb attempt 为 `17.38%`，thumb recontact 为 `1.56%`，说明 grace window 确实给了策略尝试重接的时间。但所有 episode 都下降超过 5 mm，平均最低高度损失约 `27.53 mm`，恢复率为零。这是“重新碰到”，不是“恢复承力”。

## 最终判断

```text
termination credit assignment
  -> 可以解释 O1.6 的早死投机
  -> 不能单独产生承力交接

P3 handoff 当前瓶颈
  -> 仍然是释放后的支撑恢复和有效动力学交接
  -> 不是 episode 太短或提前终止一个因素造成
```

O1.7 是有效的否定性实验：它排除了 termination credit 作为单一主因。当前不应增加训练预算，也不应使用 5090/H200 扩训。

## 下一步建议

下一阶段应先做小规模 handoff reachability 审计，比较：

1. index+middle 长期支撑是否把 policy 困在静态局部最优；
2. thumb recontact 时的法向力、支撑力矩和高度恢复是否在当前 action/PD 限制内可达；
3. 是否需要 termination-invariant potential difference，奖励“交接后恢复支撑能力”，而不是指定哪根手指接触。

## 产物

- `o1_7_A/B/C_return_order.json`
- `o1_7_A_fixed_horizon_eval256.json/.md`
- `o1_7_B_fixed_horizon_terminal_eval256.json/.md`
- `o1_7_C_recovery_grace_eval256.json/.md`
- `O1_7_Termination_Invariant_对比结果.json`
- `O1_7_Termination_Invariant_对比图.png`

## 复验记录

- 环境、评估、return audit 和对比脚本 `py_compile`：通过。
- 项目测试：`7 passed`。
- 三组 `agent.yaml`：除 experiment name 外完全一致。
- A/B env 差异：仅 terminal penalty `0 -> 20`。
- A/C env 差异：仅 termination mode `fixed_horizon -> recovery_grace`。
- 全部 O1.7 JSON：可解析。
- 冻结 O0 bank SHA-256：未变。
- 后台训练/评估进程：无遗留。
