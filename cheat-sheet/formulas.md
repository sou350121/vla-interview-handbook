# 核心公式速查 (Core Formulas)

面试中可能会要求手推或解释核心算法公式。

## 1. Transformer / Attention
VLA 的核心骨架。

### Scaled Dot-Product Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
- **Q (Query)**: 当前的观察/指令 (e.g., "Pick up apple").
- **K (Key)**: 记忆中的特征.
- **V (Value)**: 实际提取的信息.
- **$\sqrt{d_k}$**: 缩放因子，防止点积过大导致 Softmax 梯度消失。

## 2. Diffusion Policy (扩散策略)
Octo 等模型使用的动作生成方式。

### 前向过程 (Forward Process) - 加噪
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

### 逆向过程 (Reverse Process) - 去噪 (生成动作)
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
- 机器人根据当前的观察 $O$，从高斯噪声 $x_T$ 中逐步去噪，生成动作序列 $x_0$。

## 3. LoRA (Low-Rank Adaptation)
OpenVLA 微调时的核心技术。

$$
W' = W + \Delta W = W + BA
$$
- $W \in \mathbb{R}^{d \times k}$: 冻结的预训练权重。
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$: 可训练的低秩矩阵 ($r \ll d, k$)。
- **优势**: 显存占用极小，训练速度快。

## 4. 机器人学基础 (Robotics Basics)

### 坐标变换 (Coordinate Transformation)
$$
^A P = ^A T_B \cdot ^B P
$$
- $^A T_B$: 齐次变换矩阵 (4x4)，描述坐标系 B 相对于 A 的位姿。
- 包含旋转矩阵 $R$ (3x3) 和平移向量 $t$ (3x1)。

### 四元数 (Quaternion)
用于表示旋转，避免万向节死锁 (Gimbal Lock)。
$$
q = w + xi + yj + zk
$$
- 单位四元数满足 $w^2 + x^2 + y^2 + z^2 = 1$。
- **面试题**: 如何计算两个四元数的插值？(答: Slerp - 球面线性插值)

## 5. 评价指标 (Metrics)

### Success Rate (成功率)
$$
SR = \frac{\text{Number of Successful Episodes}}{\text{Total Episodes}}
$$

### Control Frequency (控制频率)
$$
f = \frac{1}{\Delta t} \quad (\text{Hz})
$$
- VLA 通常运行在 3Hz - 10Hz (高层规划)。
- 底层关节控制通常需要 500Hz - 1000Hz (由 PD 控制器处理)。
