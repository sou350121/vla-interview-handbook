# 数据处理 (Data Processing)

在 VLA 模型的训练中，数据是核心壁垒。本章介绍机器人学习中通用的数据格式和处理策略。

## 1. RLDS (Robotics Language-Image Datasets)
> **定义**: Google DeepMind 推出的一种用于机器人学习的标准数据格式，旨在统一不同数据集的接口，方便大规模训练。

### 核心结构
RLDS 将数据组织为 `Episodes` (回合) 和 `Steps` (步)。
- **Episode**: 一次完整的任务执行过程 (e.g., 抓取一个苹果)。包含元数据 (Metadata) 如任务描述、成功/失败标签。
- **Step**: 每一个时间步的数据。
    - `observation`:
        - `image`: 摄像头图像 (RGB, Depth).
        - `state`: 机器人本体状态 (关节角, 夹爪开度, TCP 坐标).
        - `language_instruction`: 语言指令 (e.g., "Pick up the apple").
    - `action`: 机器人执行的动作 (e.g., 目标关节角, 速度, 扭矩).
    - `reward`: 奖励值 (通常用于 RL).
    - `is_terminal`: 是否结束.

### 为什么使用 RLDS?
1. **标准化**: 解决了不同数据集 (Bridge, RT-1, Language-Table) 格式不一致的问题。
2. **TFDS 集成**: 基于 TensorFlow Datasets，支持高效的流式读取 (Streaming) 和 缓存 (Caching)。
3. **Open X-Embodiment**: 全球最大的机器人数据集 OXE 就是基于 RLDS 格式发布的。

### 代码示例 (Loading RLDS)
```python
import tensorflow_datasets as tfds

# 加载数据集
ds = tfds.load('fractal20220817_data', split='train')

for episode in ds.take(1):
    steps = episode['steps']
    for step in steps:
        image = step['observation']['image']
        action = step['action']
        # Training logic here...
```

## 2. 数据加权与平衡 (Data Weighting & Balancing)
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

## 3. 动作空间对齐 (Action Space Alignment)
不同机器人的动作空间不同 (e.g., 7-DoF 机械臂 vs 14-DoF 双臂 vs 四足)。

- **归一化 (Normalization)**: 将所有动作维度归一化到 [-1, 1] 或 [0, 1]。
- **Proprioception Padding**: 对于自由度较少的机器人，用 0 填充剩余维度。
- **相对控制 vs 绝对控制**:
    - **Delta Action**: 预测当前状态的增量 (dx, dy, dz)。泛化性更好。
    - **Absolute Action**: 预测绝对坐标。精度更高，但依赖标定。
    - **趋势**: VLA 模型通常偏向于使用 **Delta Action (End-effector velocity/pose delta)**。

## 面试高频考点
1. **RLDS**: 简述 RLDS 的数据结构。如何处理不同频率的数据？(答: 插值或下采样)
2. **数据平衡**: 如果我有 1000 条简单的 Pick-Place 数据和 100 条复杂的 Assembly 数据，应该怎么训练？(答: 重采样 Assembly 数据，提高其在 Batch 中的比例)
3. **Action Space**: 为什么要用 Delta Action？(答: 减少对绝对坐标的依赖，更容易迁移到不同位置或不同机器人)


---
[← Back to Theory](./README.md)
