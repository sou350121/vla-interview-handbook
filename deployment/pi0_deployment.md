# Pi0 真机部署指南 (Pi0 Real-World Deployment Guide)

> **基于官方 OpenPI 仓库**: 本指南基于 Physical Intelligence 开源的 `openpi` 架构编写。
> **适用版本**: π0 (Base), π0-FAST (High Frequency).

## 1. 真机硬件配置 (Real-World Hardware Configuration)

> **官方参考**: 基于 Physical Intelligence 内部使用的 DROID 和 ALOHA 平台配置。

### 1.1 机器人平台 (Supported Robots)
虽然 Pi0 是通用的，但官方仓库 `openpi` 重点适配了以下平台：
- **DROID (Single Arm)**: 基于 **Franka Emika Panda** 机械臂。
    - *特点*: 高精度力控，适合精细操作。
- **ALOHA (Dual Arm)**: 基于 **ViperX 300** (Trossen Robotics) 或自研低成本机械臂。
    - *特点*: 双臂协同，适合双手操作 (如叠衣服)。
- **Trossen WidowX**: 低成本桌面级机械臂，适合入门和测试。

### 1.2 视觉系统 (Vision System)
Pi0 极其依赖多视角视觉输入。标准的官方推荐配置是 **4-Camera Setup**：

| 视角 (View) | 推荐相机 | 安装位置 | 作用 |
| :--- | :--- | :--- | :--- |
| **High Cam** | Logitech C920 / RealSense D435 | 机器人正上方 (俯视) | 全局规划，定位物体位置。 |
| **Low Cam** | Logitech C920 | 机器人正前方 (平视) | 补充视角，观察物体侧面细节。 |
| **Left Wrist** | Arducam / RealSense | 左手腕部 | **关键视角**。用于对准物体，解决遮挡问题。 |
| **Right Wrist** | Arducam / RealSense | 右手腕部 | 同上。 |

> **注意**: 所有相机必须进行 **外参标定 (Extrinsic Calibration)**，确保坐标系对齐。

### 1.3 计算单元架构 (Compute Architecture)
为了解决 Pi0 (3B VLM) 推理重的问题，**强烈推荐** 采用 "大脑-小脑" 分离架构：

- **大脑 (Server)**: 高性能工作站
    - **GPU**: **NVIDIA RTX 4090 (24GB)** 或 A6000。
    - **任务**: 运行 Pi0 模型，接收图像，生成 Action Chunk (50Hz)。
    - **连接**: 通过千兆以太网 (LAN) 连接机器人。

- **小脑 (Client)**: 机载电脑 (Onboard Computer)
    - **设备**: **Intel NUC** 或 **Jetson Orin**。
    - **任务**: 读取相机数据，发送给 Server；接收 Action，控制电机。
    - **优势**: 机载电脑只需要负责 I/O，不需要跑大模型，大大降低了机载硬件成本和散热压力。

## 2. 软件环境搭建 (Software Setup)

推荐使用 Ubuntu 22.04 + Python 3.11。

### 2.1 安装 OpenPI
使用 `uv` 进行依赖管理 (官方推荐)：

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库 (包含子模块)
git clone --recursive https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 3. 创建虚拟环境并安装
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2.2 下载模型权重
模型权重托管在 Hugging Face：

```bash
# 需要先登录 Hugging Face
huggingface-cli login

# 下载 π0 Base 模型
huggingface-cli download physical-intelligence/pi0 --local-dir checkpoints/pi0
```

## 3. 部署架构：Remote Inference (推荐)

由于 Pi0 计算量较大，在机载电脑 (如 Orin) 上直接运行可能无法达到 50Hz 的控制频率。官方推荐 **Server-Client** 架构。

### 3.1 架构图
```mermaid
graph LR
    Robot[机器人 (Client)] -- 图像/状态 (WebSocket) --> Server[高性能工作站 (Server)]
    Server -- 动作 (50Hz) --> Robot
```

### 3.2 Server 端 (4090 工作站)
启动推理服务器：
```bash
python -m openpi.serving.server --model_path checkpoints/pi0 --port 8000
```

### 3.3 Client 端 (机器人/Orin)
连接服务器并执行动作：
```python
from openpi.serving.client import Pi0Client

client = Pi0Client(url="ws://192.168.1.100:8000")

while True:
    # 1. 获取机器人图像
    img = robot.get_camera_image()
    
    # 2. 发送请求并获取动作 (Blocking or Async)
    action = client.predict(image=img, instruction="Pick up the apple")
    
    # 3. 执行动作
    robot.execute(action)
```

## 4. 微调工作流 (Fine-tuning Workflow)

如果你需要让 Pi0 学会一个新技能 (例如：折叠一种从未见过的毛巾)，可以使用 LoRA 进行快速微调。

### 4.1 数据收集
- **数据量**: 推荐 **1-20 小时** 的示范数据。
- **格式**: 必须转换为 **RLDS (Reinforcement Learning Datasets)** 格式。
- **工具**: 使用 `openpi` 提供的 `rlds_converter`。

### 4.2 启动训练
使用 LoRA 配置文件启动训练：

```bash
python -m openpi.training.train \
    --config configs/pi0_lora.yaml \
    --data_path /path/to/your/rlds_data \
    --exp_name fold_towel_experiment
```

### 4.3 训练时间参考 (RTX 4090)
- **1 小时数据**: 约 2-4 小时训练时间。
- **10 小时数据**: 约 12-24 小时训练时间。

## 5. 常见问题 (FAQ)

### Q1: 为什么我在 Orin 上跑只有 5Hz?
**A**: Pi0 是 3B VLM，Orin 的推理能力有限。请开启 **RTC (Real-Time Chunking)** 或使用 Remote Inference。

### Q2: 什么是 RTC (Real-Time Chunking)?
**A**: 这是一种在端侧部署时的优化技术。模型一次推理生成多个未来的动作块 (Chunk)，机器人依次执行这些动作，从而掩盖推理延迟。

### Q3: Pi0 支持多卡训练吗?
**A**: 支持。使用 `torchrun` 启动多卡训练，可以显著减少显存压力 (Model Parallelism)。

## 6. RealSense 避坑指南 (Troubleshooting)

> **问题**: "RealSense 內容爆炸" (USB Bandwidth Overflow / Frame Drops) 是 4-Camera Setup 最常见的问题。

### 6.1 现象
- 报错: `RuntimeError: Frame didn't arrive within 5000`
- 报错: `RS2_USB_STATUS_OVERFLOW`
- 现象: 相机掉线，或者 FPS 极低 (只有 15fps 甚至更低)。

### 6.2 根本原因 (Root Cause)
- **USB 带宽瓶颈**: 4 个 RealSense D435 跑 640x480@30fps 需要约 1.5Gbps 带宽。如果它们插在同一个 USB Root Hub 上，会瞬间挤爆总线。
- **供电不足**: 4 个相机同时启动瞬间电流很大，可能导致电压骤降，相机重启。

### 6.3 解决方案 (Solutions)

#### 方案 A: 物理隔离 (Physical Isolation) [推荐]
不要把所有相机插在机箱前面板或主板自带的 USB 口上 (它们通常共享一个 Controller)。
1.  **购买 PCIe USB 扩展卡**: 推荐带有 **独立控制器 (Independent Controllers)** 的扩展卡 (如 StarTech 4-Port Dedicated 5Gbps)。
2.  **分配**: 
    - High Cam + Low Cam -> 主板 USB 口
    - Left Wrist + Right Wrist -> PCIe 扩展卡

#### 方案 B: 降低规格 (Reduce Specs)
如果无法加硬件，必须降低数据量：
1.  **降低分辨率**: 从 848x480 降级到 **640x480** 甚至 **424x240** (Pi0 对分辨率不敏感)。
2.  **降低帧率**: 强制设置为 **15fps** 或 **30fps** (不要用 60fps)。
3.  **关闭深度流**: 如果只用 RGB，务必在 Launch 文件中关闭 Depth Stream。

#### 方案 C: 硬件同步 (Hardware Sync)
多相机之间会有红外干扰 (IR Interference)。
1.  **Master-Slave 模式**: 设置一个相机为 Master (发送触发信号)，其他为 Slave。
2.  **Sync Cable**: 使用同步线连接各相机的 Sync 接口。
3.  **Librealsense 设置**:
    ```python
    # Master
    sensor.set_option(rs.option.inter_cam_sync_mode, 1)
    # Slave
    sensor.set_option(rs.option.inter_cam_sync_mode, 2)
    ```
