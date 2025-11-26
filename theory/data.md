# 数据处理 (Data Processing)

在 VLA 模型的训练中，数据是核心壁垒。本章介绍机器人学习中通用的数据格式和处理策略。

## 1. 主流数据格式对比 (Mainstream Data Formats)

在 VLA 领域，数据格式的选择直接影响训练效率和生态兼容性。目前主要有三种主流格式：

### 1.1. RLDS (Robotics Language-Image Datasets)
- **生态位**: **Google / Open X-Embodiment 标准**。
- **底层**: 基于 `TensorFlow Datasets (TFDS)` 和 `ProtoBuf`。
- **物理格式**: **`.tfrecord`** 文件。
    - 这是一种基于行 (Row-based) 的二进制序列化格式，将数据序列化为 Protocol Buffers 消息。
- **特点**:
    - **序列化**: 适合大规模分布式读取，Google TPU 友好。
    - **标准化**: 强制定义了 `observation`, `action`, `language` 的标准接口。
    - **流式读取**: 支持云端存储 (GCS) 的流式训练，无需下载整个数据集。
- **适用场景**: 使用 TPU 训练，或基于 RT-1/RT-2/Octo 架构开发时。

### 1.2. LeRobot Dataset (Hugging Face)
- **生态位**: **PyTorch / Open Source 社区新标准**。
- **底层**: 基于 `Parquet` (列式存储) 和 `Hugging Face Datasets` (Apache Arrow)。
- **物理格式**: **`.parquet`** 文件。
    - 这是一种基于列 (Column-based) 的存储格式，压缩率极高，读取特定列（如只读 Action 不读 Image）非常快。
- **特点**:
    - **可视化**: 在 Hugging Face 网页端可直接预览视频和元数据。
    - **轻量级**: 不依赖 TensorFlow，安装简单 (`pip install lerobot`)。
    - **PyTorch 原生**: 数据加载器直接输出 PyTorch Tensors。
- **适用场景**: 使用 GPU 训练，基于 OpenVLA/ACT/Diffusion Policy 开发新项目时。

### 1.3. HDF5 / Robomimic
- **生态位**: **传统科研 / 仿真数据标准**。
- **底层**: `HDF5` (Hierarchical Data Format)。
- **物理格式**: **`.hdf5`** 或 **`.h5`** 文件。
    - 类似于一个"文件系统"，内部可以像文件夹一样组织数据 (Groups/Datasets)。
- **特点**:
    - **单文件**: 整个数据集通常是一个巨大的二进制文件。
    - **随机访问**: 支持高效的随机索引读取 (Random Access)。
    - **结构灵活**: 类似于文件系统的层级结构。
- **缺点**: 不适合超大规模数据集 (TB 级别)，难以流式读取。
- **适用场景**: 仿真环境 (MuJoCo) 数据收集，小规模真机实验。

### 📊 格式对比表

| 特性 | RLDS | LeRobot | HDF5 |
| :--- | :--- | :--- | :--- |
| **背书机构** | Google DeepMind | Hugging Face | Stanford (Robomimic) |
| **核心依赖** | TensorFlow | PyTorch / Arrow | h5py |
| **存储格式** | TFRecord (序列化) | Parquet (列式) | HDF5 (层级) |
| **流式读取** | ⭐⭐⭐ (原生支持) | ⭐⭐ (支持) | ⭐ (困难) |
| **生态兼容** | Open X-Embodiment | Transformers / Hub | Simulators |
| **推荐指数** | ⭐⭐⭐ (大规模/TPU) | ⭐⭐⭐ (新项目首选) | ⭐⭐ (科研/仿真) |

---

## 2. 代码示例：如何加载数据

### 2.1. Loading RLDS (TensorFlow)
```python
import tensorflow_datasets as tfds

# 加载 Open X-Embodiment 中的 fractal 数据
ds = tfds.load('fractal20220817_data', split='train')

for episode in ds.take(1):
    steps = episode['steps']
    for step in steps:
        image = step['observation']['image']
        action = step['action']
        # 需要手动转换为 PyTorch Tensor 如果不用 TF
```

### 2.2. Loading LeRobot (PyTorch)
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 直接从 Hugging Face Hub 加载
dataset = LeRobotDataset("lerobot/pusht")

# 像标准的 PyTorch Dataset 一样使用
item = dataset[0]
image = item['observation.image']  # 自动归一化并转为 Tensor (C, H, W)
action = item['action']
print(f"Action shape: {action.shape}")
```

## 3. PyTorch 完整训练流程 (PyTorch Training Pipeline)

在 PyTorch 中训练 VLA 模型，数据流通常遵循以下模式：`Dataset` -> `DataLoader` -> `Model`。

### 3.1. 核心组件
1.  **Dataset**: 负责读取磁盘上的数据 (RLDS/Parquet)，并进行预处理 (Resize, Normalize)。
2.  **Processor/Transform**: 处理多模态数据。
    -   **Image**: `Resize((224, 224))`, `Normalize(mean, std)`.
    -   **Text**: Tokenizer (如 Llama Tokenizer) 将指令转为 Input IDs.
    -   **Action**: 归一化到 [-1, 1].
3.  **DataLoader**: 将多个样本打包成 Batch。需要自定义 `collate_fn` 来处理变长序列 (Padding)。

### 3.2. 代码实战 (Pseudo-code)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

class VLADataset(Dataset):
    def __init__(self, data_path, processor):
        self.data = load_data(data_path) # e.g., LeRobotDataset
        self.processor = processor

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 获取原始数据
        image = item['observation.image'] # (C, H, W)
        text = item['language_instruction'] # "Pick up the apple"
        action = item['action'] # (Time, Action_Dim)
        
        # 2. 多模态预处理 (关键步骤!)
        # VLA 模型通常需要同时输入图像和文本
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True
        )
        
        # 3. 返回字典
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels": action # 动作作为监督信号
        }

# 4. 训练循环
dataset = VLADataset(path, processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = OpenVLAModel.from_pretrained("openvla/openvla-7b")

for batch in dataloader:
    # 将数据送入 GPU
    input_ids = batch["input_ids"].cuda()
    pixel_values = batch["pixel_values"].cuda()
    actions = batch["labels"].cuda()
    
    # 前向传播
    # VLA 模型通常计算 Action MSE Loss 或 Cross-Entropy Loss (如果是 Tokenized Action)
    loss = model(
        input_ids=input_ids, 
        pixel_values=pixel_values, 
        labels=actions
    ).loss
    
    loss.backward()
    optimizer.step()
```

### 3.3. 常见坑点 (Pitfalls)
-   **数据类型**: 确保 Action 是 `float32` (对于 Diffusion/Regression) 或 `long` (对于 Tokenization)。
-   **图像通道**: PyTorch 默认是 `(C, H, W)`，而有些读取库 (如 OpenCV/PIL) 可能是 `(H, W, C)`，务必检查 `permute`。
-   **归一化**: 动作必须使用**统计数据 (Statistics)** 进行归一化 (e.g., min-max 或 mean-std)。**推理时必须使用相同的统计数据反归一化**。

---

## 4. 数据加权与平衡 (Data Weighting & Balancing)
在训练通用 VLA 模型时，通常会混合多种数据集。不同数据集的质量、规模和难度差异巨大，直接混合训练效果往往不佳。

### 常见策略
1. **按数据集规模加权**:
    - 简单的按比例采样，但这会导致大规模数据集 (通常是简单的重复任务) 主导训练，模型学不到复杂任务。
2. **按任务难度加权**:
    - 给包含复杂操作 (e.g., 使用工具, 长序列) 的数据集更高的权重。
3. **成功率过滤 (Success Filtering)**:
    - 仅使用 `is_terminal=True` 且 `reward=1` 的成功轨迹进行 BC (Behavior Cloning) 训练。
    - 对于失败轨迹，可以用于对比学习 (Contrastive Learning) 或作为负样本。
4. **Co-training with Web Data**:
    - 在训练批次 (Batch) 中，固定比例 (e.g., 50%) 混合 VQA (Visual Question Answering) 或 Captioning 数据。
    - **目的**: 维持 VLM backbone 的视觉语言理解能力，防止过拟合到机器人数据分布上 (Catastrophic Forgetting)。

## 4. 数据收集工具链 (Data Collection Tools)

高质量的数据源于高效的收集工具。

### 4.1. 遥操作 (Teleoperation)
- **VR 头显 (Vision Pro / Quest 3)**:
    - **优势**: 沉浸感强，能收集 6-DoF 姿态，适合灵巧手操作。
    - **方案**: ALOHA (VR版), AnyTeleop。
- **主从臂 (Leader-Follower Arms)**:
    - **优势**: 力反馈真实，操作精度极高。
    - **方案**: ALOHA (使用 WidowX 作为主臂), GELLO (低成本 3D 打印主臂)。
- **手柄/3D 鼠标**:
    - **优势**: 成本低，易获取。
    - **劣势**: 难以控制高自由度 (如灵巧手)。

### 4.2. 自动化收集 (Autonomous Collection)
- **Scripted Policy**: 在仿真或简单场景中，用硬编码脚本生成数据。
- **Self-Replay**: 机器人回放成功的轨迹，并添加噪声进行数据增强。

## 5. 动作空间对齐 (Action Space Alignment)
不同机器人的动作空间不同 (e.g., 7-DoF 机械臂 vs 14-DoF 双臂 vs 四足)。

- **归一化 (Normalization)**: 将所有动作维度归一化到 [-1, 1] 或 [0, 1]。
- **Proprioception Padding**: 对于自由度较少的机器人，用 0 填充剩余维度。
- **相对控制 vs 绝对控制**:
    - **Delta Action**: 预测当前状态的增量 (dx, dy, dz)。泛化性更好。
    - **Absolute Action**: 预测绝对坐标。精度更高，但依赖标定。
    - **趋势**: VLA 模型通常偏向于使用 **Delta Action (End-effector velocity/pose delta)**。

## 7. 面试高频考点
1.  **数据格式**: RLDS 和 LeRobot 格式有什么区别？为什么 PyTorch 用户现在倾向于 LeRobot？(答: LeRobot 去除了 TF 依赖，原生支持 PyTorch，且基于 Parquet 存储效率高)
2.  **数据流**: 在 VLA 训练中，Processor 的作用是什么？(答: 同时处理图像归一化和文本 Tokenization，确保多模态对齐)
3.  **数据平衡**: 如果我有 1000 条简单的 Pick-Place 数据和 100 条复杂的 Assembly 数据，应该怎么训练？(答: 重采样 Assembly 数据，提高其在 Batch 中的比例)
4.  **Action Space**: 为什么要用 Delta Action？(答: 减少对绝对坐标的依赖，更容易迁移到不同位置或不同机器人)
5.  **数据收集**: 相比于 VR 遥操作，主从臂 (Leader-Follower) 有什么优缺点？(答: 主从臂有力反馈，精度高，但成本高且不仅限于异构机器人映射)


---
[← Back to Theory](./README.md)
