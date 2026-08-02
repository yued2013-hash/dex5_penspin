# S0 Quaternion Replay Selection Audit

## 结论

```text
状态: BLOCKED_NO_QUALIFYING_TRAJECTORY
轨迹总数: 1200
四元数门槛: max和final均 >= 30.0 deg
合格轨迹: 0
真机速度包: NOT GENERATED
```

## 排名前十

| episode | split | quaternion max deg | quaternion final deg | integrated omega-z deg | quaternion world-z component deg |
|---:|---|---:|---:|---:|---:|
| 1188 | train | 25.04 | 25.04 | 39.52 | 0.80 |
| 25 | train | 21.69 | 21.69 | 42.53 | 3.49 |
| 487 | train | 21.29 | 21.29 | 39.94 | 4.06 |
| 1080 | test | 21.26 | 21.26 | 36.70 | 4.74 |
| 753 | test | 19.53 | 19.53 | 37.20 | 4.16 |
| 875 | train | 19.34 | 19.34 | 39.75 | 0.19 |
| 840 | train | 19.13 | 19.13 | 39.44 | 5.11 |
| 272 | test | 18.56 | 18.56 | 37.31 | 5.11 |
| 179 | train | 18.42 | 18.42 | 39.17 | 0.59 |
| 1117 | train | 18.37 | 18.37 | 43.19 | 4.14 |

## 边界

`integrated omega-z` 不再单独构成成功证据。只有四元数实际姿态变化通过门槛，
才允许生成 0.12/0.18/0.24/0.36 rad/s 包。当前若为阻断状态，不得继续真机回放。
