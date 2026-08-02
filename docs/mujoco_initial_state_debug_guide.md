# MuJoCo 初始状态与 Dex5-PenSpin 手动调试指南

这份文档的目标不是让你一次调出转笔，而是帮你建立最基本的调试能力：

```text
我改了哪个状态？
MuJoCo 里发生了什么？
笔到底是被指尖托住，还是被中段/掌面卡住？
下一步应该改哪一个量？
```

## 1. 先理解：MuJoCo 里什么叫“初始状态”

在 MuJoCo 里，初始状态主要由三类变量决定：

```text
data.qpos  位置状态，包括关节角、自由物体位置和姿态
data.qvel  速度状态，包括关节速度、自由物体线速度和角速度
data.ctrl  控制器目标，这里是 Dex5 的 position actuator 目标角
```

当前 Dex5-PenSpin 模型中：

```text
qpos[0:20]   Dex5 20 个手指关节角
qpos[20:23]  笔的位置 xyz
qpos[23:27]  笔的姿态四元数 qw qx qy qz
```

因为我们用了 position actuator，所以只改 `qpos` 还不够，还要同步改 `ctrl`：

```python
data.qpos[0:20] = hand_qpos
data.ctrl[:] = hand_qpos
mujoco.mj_forward(model, data)
```

否则电机会把手拉回旧目标。

## 2. 当前 viewer 支持的直接控制参数

文件：

```text
dex5_penspin/sim/view_m2_mujoco.py
```

基础命令：

```bash
python dex5_penspin/sim/view_m2_mujoco.py \
  --policy hold \
  --duration 20 \
  --realtime
```

可以手动设置：

```text
--custom-qpos   Dex5 20 个关节初始角
--pen-init-pos  笔的位置 xyz
--pen-quat      笔的姿态 qw qx qy qz
--pen-qvel      笔的速度 vx vy vz wx wy wz
```

示例：

```bash
python dex5_penspin/sim/view_m2_mujoco.py \
  --policy hold \
  --duration 20 \
  --realtime \
  --pen-init-pos -0.025 0.075 0.13 \
  --pen-quat 1 0 0 0 \
  --custom-qpos \
  0.00 0.70 0.45 0.15 \
  0.00 0.70 0.45 0.15 \
  0.00 0.70 0.45 0.15 \
  0.00 0.70 0.45 0.15 \
  0.10 0.55 0.40 0.20
```

## 3. Dex5 20 维 qpos 怎么读

当前顺序：

```text
0  Roll_21R      四指 1 横向展开
1  Pitch_22R     四指 1 根部弯曲
2  Pitch_23R     四指 1 中节弯曲
3  Pitch_24R     四指 1 末节弯曲

4  Roll_31R      四指 2 横向展开
5  Pitch_32R     四指 2 根部弯曲
6  Pitch_33R     四指 2 中节弯曲
7  Pitch_34R     四指 2 末节弯曲

8  Roll_41R      四指 3 横向展开
9  Pitch_42R     四指 3 根部弯曲
10 Pitch_43R     四指 3 中节弯曲
11 Pitch_44R     四指 3 末节弯曲

12 Roll_51R      四指 4 横向展开
13 Pitch_52R     四指 4 根部弯曲
14 Pitch_53R     四指 4 中节弯曲
15 Pitch_54R     四指 4 末节弯曲

16 Yaw_11R       拇指 yaw
17 Roll_12R      拇指 roll
18 Pitch_13R     拇指根部/中部弯曲
19 Pitch_14R     拇指末节弯曲
```

调试时先不用记名字，只记这个规律：

```text
每根普通手指 4 个数：
  Roll, MCP, PIP, DIP

MCP/PIP/DIP 越大，手指越弯。
DIP 太大时，笔容易碰到中段或被卡住，而不是落在指尖末端。
```

## 4. 新手调试顺序

不要一上来调 20 个关节。推荐顺序：

```text
第一步：只调普通四指，拇指先放松
第二步：让四个指尖在 viewer 里形成一条“浅托架”
第三步：把笔放在指尖上方 1-2 cm
第四步：运行 hold，看笔自然落下接触哪里
第五步：确认不是中段卡住，再加入拇指
第六步：最后才加 finger_gait 动作
```

## 5. 推荐从这个手型开始

这个手型比 M2.5 更保守，目标是让末节不要过度卷曲，避免中段夹笔：

```text
普通四指：
  Roll = 0.00
  MCP  = 0.65
  PIP  = 0.40
  DIP  = 0.10

拇指：
  Yaw   = 0.10
  Roll  = 0.45
  Pitch = 0.30
  Tip   = 0.12
```

命令：

```bash
python dex5_penspin/sim/view_m2_mujoco.py \
  --policy hold \
  --duration 20 \
  --realtime \
  --pen-init-pos -0.025 0.075 0.115 \
  --pen-quat 1 0 0 0 \
  --custom-qpos \
  0.00 0.65 0.40 0.10 \
  0.00 0.65 0.40 0.10 \
  0.00 0.65 0.40 0.10 \
  0.00 0.65 0.40 0.10 \
  0.10 0.45 0.30 0.12
```

如果笔从指尖前方滑走：

```text
把 pen y 稍微调小或调大：0.065 / 0.075 / 0.085 逐个试。
```

如果笔落不到手上：

```text
降低 pen z：0.115 -> 0.105。
```

如果笔一开始就穿插或爆开：

```text
提高 pen z：0.115 -> 0.125 或 0.135。
```

## 6. 观察时看什么

每次打开 viewer，按这个检查：

```text
1. 笔是否先落到绿色 fingertip proxy 附近？
2. 是否接触四指末端，而不是指节中段？
3. 是否只有两根手指卡住？
4. 拇指是否只是轻轻参与，而不是把笔顶飞？
5. 笔的两端高度是否越来越不一致？
6. 笔是否主要靠掌面/大 mesh 托住？
```

可以接受：

```text
笔短暂碰到指节，但最后稳定在多个指尖附近。
至少 3 根手指参与支撑。
笔身大体水平。
```

不能接受：

```text
笔被两根手指中段夹住。
笔主要落在掌面。
笔很快竖起来。
拇指把笔顶到外侧。
```

## 7. 每次只改一个量

这是新手最重要的调试习惯。

错误方式：

```text
同时改 pen_x、pen_y、pen_z、MCP、PIP、DIP、thumb。
```

正确方式：

```text
先固定 qpos，只试 pen_z。
再固定 pen_z，只试 pen_y。
再固定笔位置，只试 DIP。
```

推荐小步长：

```text
pen x/y/z: 每次改 0.005 或 0.010 m
MCP/PIP/DIP: 每次改 0.05 或 0.10 rad
thumb: 每次改 0.05 rad
```

## 8. 调参直觉

如果笔碰到手指中段：

```text
降低 PIP 或 DIP。
尤其先降低 DIP。
```

如果笔从指尖上方滚走：

```text
略微增加 MCP 或 PIP，让指尖抬高。
或者调整 pen_y，让笔落到指尖正上方。
```

如果笔竖起来：

```text
说明支撑点太偏一端。
调整 pen_x，让笔中心更接近多指支撑中心。
或者让更多手指参与，而不是两指支撑。
```

如果只有中指/无名指接触：

```text
调整 Roll_21R / Roll_51R。
让食指和小指向笔靠近。
```

如果拇指把笔顶飞：

```text
降低 thumb Roll/Pitch。
先让拇指只作为轻接触，不要夹紧。
```

## 9. 建议记录表

每次调试记一行：

```text
编号:
command:
现象:
接触位置:
是否中段夹住:
是否掌面支撑:
是否至少 3 指参与:
是否水平:
下一步:
```

示例：

```text
编号: manual_001
现象: 笔落在中指/无名指末端，但食指没参与
接触位置: middle/ring fingertip
是否中段夹住: 否
是否掌面支撑: 否
是否至少 3 指参与: 否
是否水平: 基本水平
下一步: 调 Roll_21R 或 pen_x，让食指参与
```

## 10. 当前结论

M2.5 的自动搜索不应继续直接作为正确方向，因为它被视觉验证为：

```text
更像两根手指中段夹住/卡住笔，
不是论文中希望的多指指尖托住笔。
```

下一步应该改成：

```text
手动设置可信 reset
用 viewer 视觉确认
再把确认过的 reset 写成 preset
最后在 preset 上调 finger_gait
```

## 11. 交互式遥操作摆手型

现在已经新增一个键盘遥操作工具：

```text
dex5_penspin/sim/teleop_m2_initial_state.py
```

启动：

```bash
cd "/home/linkerhand/桌面/deep learning"

python dex5_penspin/sim/teleop_m2_initial_state.py
```

这个工具的逻辑是：

```text
默认物理暂停。
你先摆手指。
按 SPACE 后物理运行，笔会自然落下。
再按 SPACE 暂停，继续调。
满意后按 S 保存当前 hand_qpos、pen state 和截图。
```

按键：

```text
1 / 2 / 3 / 4 / 5
  选择手指：食指 / 中指 / 无名指 / 小指 / 拇指

Q / W / E / R
  选择当前手指里的关节槽位
  普通手指：Roll / MCP / PIP / DIP
  拇指：Yaw / Roll / Pitch / Tip

A / D
  当前关节减小 / 增大

Z / C
  当前整根手指整体减小 / 增大

P
  重置笔到初始位置

SPACE
  暂停 / 运行物理

S
  保存当前位姿和截图

H
  在终端打印帮助
```

保存位置：

```text
dex5_penspin/reports/manual_initial_state_presets/
```

保存后会生成：

```text
manual_时间戳.json
manual_时间戳.png
```

为了防止忘记按 `S`，工具现在会自动保存：

```text
autosave_latest.json
  每次调关节或重置笔时自动更新。

manual_exit_autosave_时间戳.json
manual_exit_autosave_时间戳.png
  关闭 viewer 窗口时自动保存。
```

如果你忘了按 `S`，优先找：

```text
dex5_penspin/reports/manual_initial_state_presets/autosave_latest.json
```

其中 json 里包含：

```text
hand_qpos
joint_order
pen_pos
pen_quat
pen_qvel
replay command
```

你保存后可以直接告诉我：

```text
我保存了 manual_xxx.json
```

我就可以读取该文件，帮你判断：

```text
1. 这个 hand_qpos 是否合理
2. 是否真的多指指尖托住
3. 笔是否会竖起
4. 是否值得作为 reset preset
5. 下一步该调哪个关节
```
