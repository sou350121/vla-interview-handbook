# 面试题库 (Question Bank)

本题库涵盖了 VLA 算法岗面试的高频问题，分为概念题、场景题和代码题。

## 1. 概念题 (Conceptual Questions)

### Q1: 解释一下 RT-2 的 Co-fine-tuning 策略，为什么它很重要？
- **参考答案**:
    - RT-2 将机器人动作编码为文本 Token，与互联网 VQA 数据混合训练。
    - **重要性**: 纯机器人数据量太小，容易导致模型遗忘预训练 VLM 的语义知识 (Catastrophic Forgetting)。混合训练让模型既能听懂 "抓恐龙" (语义)，又能输出动作 (控制)，实现了 Zero-shot 泛化。

### Q2: VLA 模型中，Action Tokenization 和 Continuous Regression 有什么区别？
- **参考答案**:
    - **Tokenization (离散化)**: 将连续动作空间划分为 bins (e.g., 256个)，作为分类问题处理。
        - *优点*: 可以建模多模态分布 (Multimodal Distribution)，即同一个状态下可能有多种合理的动作。Transformer 擅长处理离散序列。
        - *缺点*: 精度受限于 bin 的数量，高频控制可能不平滑。
    - **Regression (回归)**: 直接预测连续数值 (MSE Loss)。
        - *优点*: 精度高，适合精细操作。
        - *缺点*: 假设动作分布是单峰高斯 (Unimodal Gaussian)，难以处理多解情况 (e.g., 从左边抓还是右边抓)。

### Q3: 什么是 Sim-to-Real Gap？如何解决？
- **参考答案**:
    - 仿真与真机在视觉 (光照、纹理) 和动力学 (摩擦、质量) 上的差异。
    - **解决方法**:
        1. **Domain Randomization**: 在仿真中随机化各种参数，让真机成为分布中的一种。
        2. **Domain Adaptation**: 使用 GAN 或特征对齐技术。
        3. **System Identification**: 辨识真机参数反馈给仿真。

### Q4: Transformer vs CNN 在 VLA 中的选择：什么时候用 ViT，什么时候用 ResNet？
- **参考答案**:
    - **ViT (优选)**:
        - **多模态统一**: 需要将视觉、语言、动作全部 Tokenize 并拼接 (e.g., RT-2, OpenVLA)。
        - **全局上下文**: 需要理解画面中远距离的物体关系 (e.g., "桌子左边的杯子和右边的壶")。
        - **Scaling Law**: 有大量数据时 ViT 效果更好。
    - **ResNet (优选)**:
        - **小样本**: 数据不足时 CNN 的归纳偏置有助于快速收敛。
        - **纹理特征**: 需要高频细节（如 GelSight 触觉传感器）。
        - **推理速度**: CNN 通常比 ViT 轻量，适合边缘设备。

### Q5: 什么是触觉 VLA (Tactile VLA)？为什么视觉不够？
- **参考答案**:
    - **Tactile VLA**: 融合了触觉传感器 (e.g., GelSight) 的 VLA 模型，能够感知接触、材质、滑移等视觉无法获取的信息。
    - **为什么视觉不够**:
        1. **遇挡 (Occlusion)**: 机械手抓取时，手掌会挡住摄像头。
        2. **物理属性**: 视觉无法直接判断软硬、摩擦力。
        3. **微米级控制**: 触觉传感器提供更高精度的反馈。
    - **代表模型**: VLA-Touch (2025), OmniVTLA (2025)。

### Q6: VLA-Touch 和 OmniVTLA 有什么区别？
- **参考答案**:
    - **VLA-Touch**: 双层反馈机制。
        - **High-level**: 使用 Tactile-Language Model (TLM) 将触觉信号翻译成语言，辅助 VLM 决策。
        - **Low-level**: 通过 FiLM 将触觉特征注入 Diffusion Policy。
        - **优势**: 无需重训整个 VLA，即插即用。
    - **OmniVTLA**: 统一模型 (Unified Tokenization)。
        - **架构**: 将视觉、触觉、语言全部 Token 化，输入同一个 Transformer。
        - **语义对齐**: 使用 InfoNCE Loss 拉近触觉 Embedding 与材质描述文本的距离。
        - **优势**: 能执行跨模态推理 (e.g., "Pick up the softest object")。

## 2. 场景题 (Scenario Questions)

### S1: 只有 100 条真机演示数据，如何训练一个鲁棒的抓取策略？
- **参考答案**:
    - **数据增强**: 旋转、裁剪、颜色抖动。
    - **Sim-to-Real**: 先在仿真中训练一个基础策略，用这 100 条数据做 Fine-tuning。
    - **Co-training**: 混合大规模开源数据集 (如 OXE) 进行训练，但这 100 条数据赋予更高的采样权重。
    - **使用预训练模型**: 基于 OpenVLA 或 RT-1 预训练权重进行微调 (LoRA)。

### S2: 机器人抓取透明物体 (Transparent Object) 总是失败，怎么办？
- **参考答案**:
    - **传感器层面**: RGB-D 相机的深度光在透明物体上会失效 (穿透/反射)。考虑使用立体视觉 (Stereo) 或 补全深度图 (Depth Completion)。
    - **算法层面**:
        - 在训练数据中加入大量透明物体。
        - 使用 **末端触觉 (Tactile Sensor)**: 视觉可能看不准，但摸到了就知道。
        - **多视角融合 (Multi-view Fusion)**: 移动机械臂从不同角度观察，利用镜面反射特征。

## 3. 代码题 (Coding Questions)

### C1: 实现一个简单的 PID 控制器
```python
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        return output
```

### C2: 计算两个 3D 点之间的欧氏距离 (NumPy)
```python
import numpy as np

def distance(p1, p2):
    # p1, p2: np.array [x, y, z]
    return np.linalg.norm(p1 - p2)
```

### C3: 旋转矩阵转欧拉角 (概念)
- 面试官可能会问转换公式或万向节死锁问题。建议复习 `scipy.spatial.transform.Rotation` 的用法。


---
[← Back to Question Bank](./README.md)
