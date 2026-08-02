# M0/M1: Official PenSpin Repo and Dex5 Interface Audit

> Date: 2026-07-08  
> Project root: `/home/linkerhand/桌面/deep learning/dex5_penspin`  
> Official repo clone: `external/penspin`  
> Dex5 ROS2 workspace: `/home/linkerhand/ofdemo_ros2_ws`

## 1. M0 Conclusion

The official `penspin` repository is a four-stage sim-to-real project, not a single deploy script.

```text
simulation oracle RL
-> student policy distillation
-> real open-loop replay and success filtering
-> real-data fine-tuning
```

For Dex5, the most useful reproduction route is:

```text
first copy the data/control shape
then replace Allegro/xArm-specific parts with Dex5 interfaces
then build Dex5 simulation and real replay
finally move toward closed-loop multi-round real pen spinning
```

## 2. Official Repo Snapshot

Repository:

```text
https://github.com/HaozhiQi/penspin.git
```

Local clone:

```text
dex5_penspin/external/penspin
```

Current shallow clone commit:

```text
5035c52
```

Top-level structure:

```text
assets/        robot and object assets, including Allegro URDF and pencil/cylinder objects
cache/         grasp caches
configs/      Hydra task/train configs
docs/         install and baseline notes
penspin/      simulation tasks, PPO, DemonTrain, utility code
real/         real data, real fine-tuning, robot replay/controller code
scripts/      shell entrypoints for teacher/student/finetune/visualization
tools/        object generation scripts
train.py      main simulation training entrypoint
gen_grasp.py  grasp cache generation entrypoint
```

## 3. Official Pipeline Modules

### Step 1: Oracle Policy in Simulation

Official entry:

```bash
scripts/train_teacher.sh 0 42 output_name
```

Important files:

```text
train.py
penspin/tasks/allegro_hand_hora.py
penspin/algo/ppo/
configs/task/AllegroHandHora.yaml
configs/train/AllegroHandHora.yaml
scripts/train_teacher.sh
```

Key settings observed:

```text
task: AllegroHandHora
algorithm: PPO
object: cylinder_pencil-5-7
rotation_axis: +z
sim dt: 0.005 s, 200 Hz
controlFrequencyInv: 10, about 20 Hz policy control
Allegro action dim: 16
action_scale: 0.04167
torque_limit: 0.7
episodeLength: 400
privileged info: object orientation, object angular velocity, fingertip position, tactile, point cloud
```

Meaning for Dex5:

```text
The official oracle can use privileged simulator state.
Dex5 oracle can also use pen pose, pen angular velocity, contact, and simulator-only physical parameters.
The real student/deploy policy must not depend on impossible real observations unless we build those sensors.
```

### Step 2: Student Policy Pretraining

Official entry:

```bash
scripts/train_student_sim.sh train.ppo.is_demon=True train.demon_path=ORACLE_CHECKPOINT_PATH
```

Important files:

```text
penspin/algo/ppo/demon.py
scripts/train_student_sim.sh
configs/train/AllegroHandHora.yaml
```

Key settings observed:

```text
algorithm: DemonTrain
student input mode: proprio
proprio history length: 30
distill: true
use_l1: true
teacher rollout is used as demonstration data
```

Meaning for Dex5:

```text
Dex5 student should first imitate successful oracle trajectories.
A practical first student observation can be:
  qpos, dq, target_qpos, last_action, tactile/force, optional pen endpoint estimate
```

### Step 3: Real Open-Loop Replay

Official entry:

```bash
python real/robot_controller/teacher_replay.py --data-collect --exp=0 --replay_data_dir=REPLAY_DATA_DIR
```

Important files:

```text
real/robot_controller/teacher_replay.py
real/robot_controller/real_robot/robot.py
```

Official replay data records:

```text
action
qpos
current_target
tactile
rgb_ori
rgb_c2d
depth
pc
```

Official processed HDF5 may include:

```text
qpos
action
current_target_qpos
rgb_ori
rgb_c2d
depth
pc
obj_ends
```

Meaning for Dex5:

```text
This stage maps cleanly to Dex5:
  policy/trajectory target_qpos -> /dex5_1/set_position
  measured qpos/dq -> /dex5_1/joint_state
  applied target -> /dex5_1/applied_position
  touch/force -> /dex5_1/touch_data and /dex5_1/fingertip_forces
```

The missing part is object tracking:

```text
official repo uses rgb/depth/pc/obj_ends
Dex5 control repo does not provide pen pose or pen endpoints
```

### Step 4: Real-Data Fine-Tuning

Official entry:

```bash
scripts/finetune_ppo.sh --real-dataset-folder=REAL_DATA_PATH --checkpoint-path=YOUR_CHECKPOINTPATH
```

Important files:

```text
real/finetune_ppo.py
real/dataset/ppo_dataset.py
real/agent/ppo_agent.py
```

Observed training style:

```text
This real fine-tuning script is supervised/data-driven fine-tuning over real trajectories.
It loads real_data.h5, chunks sequences, and trains with action labels.
It is not online RL by itself.
```

Meaning for Dex5:

```text
Before risky online RL, use successful Dex5 replay data for supervised real adaptation.
Only after safe replay and reliable tracking should we consider small-noise online RL.
```

## 4. Dex5 ROS2 Control Interface

Dex5 control package:

```text
/home/linkerhand/ofdemo_ros2_ws/src/unitree_dex5_1_control
remote: https://github.com/lvlvturancao/unitree_dex5_1_control.git
commit: 831f0a262a68b12bac47d8d410b018e8a95c4ae3
branch: main
```

The workspace contains local uncommitted changes. This audit only reads it and does not modify it.

Launch:

```bash
ros2 launch unitree_dex5_1_control dex5_1_control.launch.py
./start_dex5_control.sh
```

Controller node:

```text
/dex5_1_controller
```

Default command publish rate:

```text
100 Hz
```

Dex5 MuJoCo bridge reference:

```text
sim_rate: 500 Hz
publish_rate: 100 Hz
```

## 5. Dex5 Topics

### Observations

| Topic | Type | Role |
|---|---|---|
| `/dex5_1/joint_state` | `unitree_dex5_1_control/msg/Dex51JointState` | 20D joint position, velocity, torque/force fields |
| `/dex5_1/touch_data` | `unitree_dex5_1_control/msg/Dex51TouchData` | touch pressures, contact states, touch positions |
| `/dex5_1/fingertip_forces` | `unitree_dex5_1_control/msg/Dex51FingertipForces` | normal/tangential fingertip forces in N |
| `/paxini/fingertip_forces` | `unitree_dex5_1_control/msg/Dex51FingertipForces` | optional Paxini fingertip force source |
| `/dex5_1/applied_position` | `std_msgs/msg/Float64MultiArray` | current applied 20D target qpos |

### Commands

| Topic | Type | Shape | Role |
|---|---|---:|---|
| `/dex5_1/set_position` | `std_msgs/msg/Float64MultiArray` | 20 | target joint positions, rad |
| `/dex5_1/set_force` | `std_msgs/msg/Float64MultiArray` | 20 | joint force/torque command |
| `/dex5_1/set_hybrid` | `std_msgs/msg/Float64MultiArray` | 40 | first 20 position, last 20 force |
| `/dex5_1/set_manus_position` | `std_msgs/msg/Float64MultiArray` | 20 | Manus 20 DOF input, rad |

## 6. Dex5 Joint Order

The 20D command/order is:

```text
Index : Roll_21R  Pitch_22R  Pitch_23R  Pitch_24R
Middle: Roll_31R  Pitch_32R  Pitch_33R  Pitch_34R
Ring  : Roll_41R  Pitch_42R  Pitch_43R  Pitch_44R
Pinky : Roll_51R  Pitch_52R  Pitch_53R  Pitch_54R
Thumb : Yaw_11R   Roll_12R   Pitch_13R  Pitch_14R
```

The controller maps 20 URDF joints to 16 physical motors:

```text
[0,1,2,2, 3,4,5,5, 6,7,8,8, 9,10,11,11, 12,13,14,15]
```

Implication:

```text
Four-finger DIP/PIP pairs share a physical motor.
Dex5 policy can output 20D commands for compatibility, but the safety filter must enforce shared-joint consistency.
```

## 7. Filled M1 Configuration

Updated file:

```text
dex5_penspin/configs/dex5_penspin_task.yaml
```

Important decisions:

```text
control_hz: 100.0
episode_seconds initial: 10.0
action type: delta_target_qpos
action dim: 20
initial action delta limit: 0.03 rad/step
initial max joint velocity cap: 2.0 rad/s
initial max force compensation tau: 0.2
```

Important caveat:

```text
The action/velocity/force limits above are initial conservative software limits.
They are not final hardware-certified values.
They must be validated with low-speed replay before autonomous policy rollout.
```

## 8. Direct Mapping: Official PenSpin -> Dex5

| Official PenSpin Concept | Allegro/xArm Implementation | Dex5 Replacement |
|---|---|---|
| 16D Allegro qpos | `RealRobot.get_observation()[:16]` | `/dex5_1/joint_state.finger_positions`, 20D |
| 16D target qpos | `RealRobot.set_action(final_action)` | publish 20D to `/dex5_1/set_position` |
| action | `final_action - cur_pos` | `target_qpos - qpos`, 20D |
| tactile/force | Allegro FSR/tactile path | `/dex5_1/touch_data`, `/dex5_1/fingertip_forces`, `/paxini/fingertip_forces` |
| real trajectory | pickle/HDF5 real data | Dex5 HDF5 with qpos, dq, action, target_qpos, object state |
| obj_ends | RGB-D + Tracking-SAM | external Dex5 RGB-D/marker tracking still required |
| oracle sim env | IsaacGym AllegroHandHora | Dex5 simulation env to be built |

## 9. Current Blocking Gaps

### Object State

Dex5 controller does not provide:

```text
pen pose
pen axis
pen angular velocity
pen rotation phase
pen endpoints
```

This blocks:

```text
rotation reward
drop detection by object state
success/failure counting
closed-loop object-aware policy
```

Minimum next solution:

```text
start with video/manual success labels for open-loop replay
then add pen endpoint tracking via RGB-D, marker, or a simplified external tracker
```

### Simulation Asset

Official repo uses:

```text
assets/round_tip/allegro_hand_right_fsr_round_dense.urdf
assets/cylinder/pencil-5-7
IsaacGym AllegroHandHora task
```

Dex5 needs:

```text
Dex5 URDF/MJCF import
Dex5 joint limit and shared motor handling
pen object inertia/friction/contact tuning
Dex5 task class or MuJoCo/Isaac equivalent
```

### Real Safety

Before real policy rollout:

```text
manual stop must be ready
command clamp must run in policy executor
joint limit clamp must use Dex5 joint ranges
velocity limit must be measured on hardware
first rollout should be open-loop replay at reduced speed
```

## 10. M2 Recommended Next Steps

1. Build `real/dex5_policy_executor.py`.

It should:

```text
subscribe /dex5_1/joint_state
subscribe /dex5_1/applied_position
optionally subscribe touch/force topics
publish /dex5_1/set_position
apply action clamp and joint limit clamp
record HDF5-compatible rollout data
```

2. Build `scripts/inspect_dex5_topics.py`.

It should print:

```text
topic availability
message frequency
array dimensions
first non-empty sample
```

3. Convert Dex5 recorder NPZ to project HDF5.

Starting point:

```text
/home/linkerhand/ofdemo_ros2_ws/src/unitree_dex5_1_control/unitree_dex5_1_control/scripts/dex5_episode_recorder.py
```

Target format:

```text
qpos
dq
action
target_qpos
timestamps
success
failure_reason
```

4. Design object tracking choice.

Options:

```text
marker tracking on pen ends
RGB-D endpoint tracking
SAM-style segmentation plus endpoint fitting
motion-capture if available
```

For fastest closed-loop learning, marker-based endpoint tracking is the most practical first choice.

