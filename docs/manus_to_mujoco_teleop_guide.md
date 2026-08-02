# Manus 手套遥操作 Dex5-PenSpin MuJoCo 指南

目标：

```text
Manus 手套 -> Dex5 20 维目标关节角 -> Dex5-PenSpin MuJoCo viewer
```

这个流程只控制 MuJoCo 仿真，不直接控制 Dex5 真机。

## 1. 为什么要用 Manus 遥操作

键盘调 `custom-qpos` 太慢，也很难摆出自然手型。

Manus 手套的优势是：

```text
你用自己的手直接摆姿态
retargeting 节点把 Manus 手型转成 Dex5 20 维 qpos
MuJoCo 里实时显示 Dex5 手和笔
满意时保存 hand_qpos + pen pose
```

这一步不是训练 RL，而是在做 reset 设计：

```text
找到一个人眼认可的多指指尖托笔初始状态。
```

## 2. 当前使用的链路

我们优先使用已有的 ROS2 retargeting 节点：

```text
/manus_glove_0
  -> manus_mujoco_hybrid_node
  -> /_dex5_mujoco/command
  -> dex5_penspin/sim/teleop_m2_manus_ros.py
  -> MuJoCo viewer
```

关键点：

```text
manus_mujoco_hybrid_node 只发布 MuJoCo command topic
不会直接向真实 Dex5 下发 /dex5_1/set_position
```

## 3. 注意 Python 环境

ROS Jazzy 使用 Python 3.12。

你的 conda `(base)` 是 Python 3.13，不能直接 import `rclpy`。

所以 Manus-MuJoCo 脚本要用：

```text
/usr/bin/python3
```

不要用：

```text
python
```

## 4. 终端 A：启动 Manus 数据发布

打开一个新终端：

```bash
source /home/linkerhand/ofdemo_ros2_ws/install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

检查是否有数据：

```bash
source /home/linkerhand/ofdemo_ros2_ws/install/setup.bash
ros2 topic echo /manus_glove_0 --once
```

或者只看频率：

```bash
ros2 topic hz /manus_glove_0
```

如果没有数据，先确认：

```text
Manus Core 是否启动
手套是否连接
ROS topic 是否真的是 /manus_glove_0
```

## 5. 终端 B：启动 Manus -> Dex5 retargeting

打开第二个终端：

```bash
source /home/linkerhand/ofdemo_ros2_ws/install/setup.bash

ros2 run unitree_dex5_1_control manus_mujoco_hybrid_node \
  --ros-args \
  -p manus_topic:=/manus_glove_0 \
  -p command_topic:=/_dex5_mujoco/command \
  -p publish_rate:=100.0 \
  -p retarget_rate:=30.0 \
  -p target_smoothing_alpha:=0.2
```

检查是否输出 Dex5 command：

```bash
ros2 topic echo /_dex5_mujoco/command --once
```

如果这里没有输出，通常说明：

```text
/manus_glove_0 没有数据
raw_nodes 不完整
retargeting 节点拒绝了当前 Manus 帧
```

## 6. 终端 C：启动 Dex5-PenSpin MuJoCo viewer

打开第三个终端：

```bash
source /home/linkerhand/ofdemo_ros2_ws/install/setup.bash
cd "/home/linkerhand/桌面/deep learning"

/usr/bin/python3 dex5_penspin/sim/teleop_m2_manus_ros.py \
  --command-topic /_dex5_mujoco/command \
  --hand-mode kinematic \
  --smoothing-alpha 0.35
```

窗口打开后：

```text
Manus 手套动 -> MuJoCo Dex5 手动
```

按键：

```text
SPACE  暂停/运行笔的物理
P      重置笔到初始位置
S      保存当前手型、笔状态和截图
H      打印帮助
```

推荐使用方式：

```text
1. 先保持 physics paused。
2. 用 Manus 手套摆出“指尖托笔”手型。
3. 按 P 重置笔到手指上方。
4. 按 SPACE，让笔自然落下。
5. 观察是否落在多个指尖，而不是中段/掌面。
6. 再按 SPACE 暂停，继续用手套调整。
7. 满意后按 S 保存。
```

保存目录：

```text
dex5_penspin/reports/manus_mujoco_presets/
```

保存后会生成：

```text
manus_时间戳.json
manus_时间戳.png
```

如果退出时忘记按 `S`，脚本也会自动保存：

```text
manus_exit_autosave_时间戳.json
```

## 7. 判断一个 Manus reset 是否合格

可以接受：

```text
至少 3 根手指参与支撑
主要接触发生在指尖附近
笔身大体水平
拇指是轻支撑，而不是把笔顶飞
笔不是主要躺在掌面上
```

不能接受：

```text
两根手指中段夹住笔
笔主要靠掌面稳定
笔很快竖起来
拇指把笔推出去
食指/小指完全不参与
```

## 8. 常见问题

### rclpy import 报 Python 版本错误

原因：

```text
conda Python 3.13 与 ROS Jazzy Python 3.12 不匹配。
```

解决：

```bash
source /home/linkerhand/ofdemo_ros2_ws/install/setup.bash
/usr/bin/python3 dex5_penspin/sim/teleop_m2_manus_ros.py
```

### viewer 一直显示 waiting for Manus/Dex5 target messages

检查：

```bash
ros2 topic echo /manus_glove_0 --once
ros2 topic echo /_dex5_mujoco/command --once
```

如果 `/manus_glove_0` 没数据，问题在 Manus 数据发布。

如果 `/manus_glove_0` 有数据但 `/_dex5_mujoco/command` 没数据，问题在 retargeting 节点。

### 手型太抖

先提高平滑：

```bash
/usr/bin/python3 dex5_penspin/sim/teleop_m2_manus_ros.py \
  --smoothing-alpha 0.15
```

`smoothing-alpha` 越小越稳，但延迟越大。

### 手跟得太慢

提高平滑响应：

```bash
/usr/bin/python3 dex5_penspin/sim/teleop_m2_manus_ros.py \
  --smoothing-alpha 0.6
```

### 想走 controller 的 applied_position

不推荐作为第一选择，因为可能涉及真实 Dex5 controller。

如果你明确只是在安全环境中使用，可以让 viewer 订阅：

```bash
/usr/bin/python3 dex5_penspin/sim/teleop_m2_manus_ros.py \
  --subscribe-float \
  --float-topic /dex5_1/applied_position
```

## 9. 下一步

当你保存一个看起来合理的 Manus preset 后，告诉我文件名，例如：

```text
我保存了 dex5_penspin/reports/manus_mujoco_presets/manus_20260709_xxxxxx_00.json
```

我会帮你：

```text
1. 读取 hand_qpos 和 pen pose
2. 回放 hold 测试
3. 统计接触分布
4. 判断是否是多指指尖托笔
5. 决定是否作为 M2.6 reset preset
```
