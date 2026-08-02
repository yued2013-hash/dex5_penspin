# 跨机器安装与逐阶段复现

## 1. 固定的软件栈

| 组件 | 固定版本 |
|---|---|
| Ubuntu | 22.04 推荐 |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| torchvision / torchaudio | 0.22.0 / 2.7.0 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | `b4c321024792976150ca55fddb26fa34480d974e` |
| skrl | 2.1.0 |
| Git LFS | 3.7.1（用户目录安装） |

需要 NVIDIA GPU、可用的 CUDA 驱动和 Conda。安装使用 NVIDIA Isaac Sim
即表示需要阅读并接受其 EULA；脚本通过 `ACCEPT_EULA=Y` 启动 Kit。

## 2. 新机器安装

```bash
git clone https://github.com/yued2013-hash/dex5_penspin.git
cd dex5_penspin
./bootstrap.sh
```

`bootstrap.sh` 会：

1. 在 Conda 环境 `env_isaaclab` 中安装 Python 依赖；
2. 安装 cu128 的 PyTorch 2.7；
3. 安装 Isaac Sim 5.1；
4. 将 Isaac Lab 克隆到 `.deps/IsaacLab` 并检出固定 commit；
5. 安装 Git LFS 3.7.1 到 `~/.local/bin`，下载最小复现资产；
6. 校验版本、CUDA 和 Isaac Sim import。

首次安装体积较大。`.deps`、Conda 环境和下载缓存不提交 Git。

## 3. 路径变量

仓库不依赖任何用户主目录的绝对路径。可按需覆盖：

```bash
export DEX5_PROJECT_ROOT="$PWD"
export ISAACLAB_ROOT="$PWD/.deps/IsaacLab"
export ISAACLAB_LOG_ROOT="$ISAACLAB_ROOT/logs/skrl"
export DEX5_ARTIFACT_ROOT="$PWD/artifacts"
export DEX5_SDK_ROOT=/path/to/unitree_dex5_1_control
export DEX5_ROS_WS=/path/to/ros2_workspace
```

`DEX5_SDK_ROOT` 和 `DEX5_ROS_WS` 只用于真机接口；仿真 smoke 不需要它们。

## 4. 逐阶段命令

### E0：版本与 GPU

```bash
./scripts/reproduce.sh env
```

### E1：离线确定性复现

验证四个 LFS 文件的 SHA256，重新生成 O1.23 failure-manifold 报告，并运行
对应单元测试。此阶段不启动 Isaac Sim。

```bash
./scripts/reproduce.sh offline
```

### E2：Isaac Lab 环境 smoke

创建真实 Dex5+pen privileged-oracle 环境，运行 16 个并行环境、8 个控制步，
检查 observation/reward 有限值和 reset bank hash。

```bash
./scripts/reproduce.sh isaac-smoke
```

### E3：重新生成 O1.23 rollout

使用两个 LFS checkpoint，各运行 256 个环境、120 步，再执行离线分析：

```bash
./scripts/reproduce.sh rollout
```

输出默认位于 `outputs/reproduction/`。E3 受 GPU、PhysX 和驱动差异影响，
不要求逐浮点位相同；E1 使用发布的固定 rollout，结果必须精确通过。

一次执行 E0-E2：

```bash
./scripts/reproduce.sh all
```

## 5. 最小数据包

Git LFS 发布以下必要资产：

- O1.22 A/B 两份 `agent_1440.pt`；
- O1.23 A/B 两份 256x120 rollout；
- `artifacts/o1_23/manifest.sha256` 完整性清单。

O0 冻结 reset bank 体积较小，仍由普通 Git 跟踪。历史 checkpoint、完整训练
日志和大规模 teacher/cache 不属于最小复现包；需要时应放在 GitHub Release
或共享存储，并在 release 中记录 SHA256 和生成命令。

## 6. 常见问题

- LFS 文件只有几行文本：运行 `~/.local/bin/git-lfs install --local && git lfs pull`。
- 找不到 Conda：先安装 Miniconda，或确保 `conda` 在 PATH 中。
- Isaac Lab 位于其他目录：运行前设置 `ISAACLAB_ROOT`。
- 真机 SDK 不在默认位置：设置 `DEX5_SDK_ROOT`，不要修改源码路径。
