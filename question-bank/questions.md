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
