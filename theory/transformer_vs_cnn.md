# Transformer vs CNN: 核心架构对比

在 VLA (Vision-Language-Action) 面试中，理解 Backbone (骨干网络) 的差异至关重要。虽然现在的趋势是 Transformer (ViT) 一统天下，但 CNN (ResNet, EfficientNet) 依然在 RT-1 等经典模型中扮演重要角色。

## 1. 核心差异一览表

| 特性 | CNN (卷积神经网络) | Transformer (自注意力机制) |
| :--- | :--- | :--- |
| **核心操作** | Convolution (卷积) | Self-Attention (自注意力) |
| **感受野 (Receptive Field)** | 局部 (Local) -> 逐渐扩大 | 全局 (Global) |
| **归纳偏置 (Inductive Bias)** | 强 (平移不变性, 局部性) | 弱 (需海量数据学习) |
| **计算复杂度** | $O(N)$ (线性) | $O(N^2)$ (平方级) |
| **并行性** | 高 | 极高 |
| **擅长领域** | 图像纹理, 边缘检测 | 语义理解, 长距离依赖 |
| **代表模型** | ResNet, EfficientNet | ViT, BERT, GPT |

## 2. CNN (Convolutional Neural Networks)

### 原理
CNN 模仿生物视觉皮层，通过**滑动窗口 (Sliding Window)** 提取特征。
- **局部性 (Locality)**: 每次只看一小块区域 (e.g., 3x3 像素)。
- **平移不变性 (Translation Invariance)**: 猫在图片左上角还是右下角，识别出的特征是一样的。
- **层级结构**: 浅层学边缘/纹理，深层学形状/物体。

### 在 VLA 中的应用
- **RT-1**: 使用 **EfficientNet-B3** 作为视觉编码器。
- **优势**: 训练收敛快，对小数据集友好 (因为有很强的归纳偏置)。
- **劣势**: 难以捕捉长距离关系 (比如：桌子左边的杯子和桌子右边的壶之间的关系，CNN 需要堆叠很多层才能"看"到两者)。

## 3. Transformer (Attention Is All You Need)

### 原理
Transformer 抛弃了卷积，完全依赖 **Self-Attention (自注意力机制)**。
- **全局感受野**: 每一个 Token (像素块) 都能直接"关注"到图像中的其他所有 Token。
- **动态权重**: 卷积核的权重是固定的 (训练好后)，而 Attention Map 是根据输入动态生成的。

### 在 VLA 中的应用
- **RT-2 / OpenVLA / Pi0**: 使用 **ViT (Vision Transformer)** 或 **SigLIP**。
- **优势**:
    - **多模态统一**: 图像 Patch 和文本 Token 可以被同等对待，直接拼接输入 Transformer。
    - **Scaling Law**: 数据越多，模型越大，效果越好 (由弱归纳偏置决定)。
- **劣势**: 训练极其昂贵，需要海量数据 (JFT-300M, LAION-5B) 才能超越 CNN。

## 4. 为什么 VLA 转向 Transformer?

1.  **多模态融合**: 机器人需要同时处理视觉 (Vision) 和语言 (Language)。Transformer 是目前唯一能完美统一这两种模态的架构 (Early Fusion)。
2.  **语义理解**: 机器人不再只是"执行动作"，而是需要"理解环境"。Transformer 在语义提取上远强于 CNN。
3.  **时序建模**: 动作序列 (Action Sequence) 本质上是时间序列。Transformer (GPT 风格) 天生适合处理序列预测问题。

## 5. 深度解析: ViT & SigLIP 技术细节
在 OpenVLA 和 Pi0 等现代模型中，ViT (Vision Transformer) 通常搭配 **SigLIP** 预训练目标使用。

### 5.1 Vision Transformer (ViT) 核心组件
ViT 将图像视为一系列 Patch 的序列，完全摒弃了卷积。

1.  **Patchify (切片)**:
    - 输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 被切分为 $N$ 个 $P \times P$ 的 Patch。
    - $N = (H \times W) / P^2$。例如 $224 \times 224$ 图像，Patch 大小 16，则 $N = 196$。
    - 每个 Patch 被展平并通过线性映射 (Linear Projection) 转换为 $D$ 维向量。

2.  **Positional Embedding (位置编码)**:
    - 由于 Transformer 是置换不变的 (Permutation Invariant)，必须手动注入位置信息。
    - ViT 使用可学习的 1D 位置编码，直接加到 Patch Embedding 上。

3.  **CLS Token vs Average Pooling**:
    - **CLS Token**: 原始 ViT 在序列开头加一个特殊的 `[CLS]` Token，其输出作为整张图的特征 (BERT 风格)。
    - **Average Pooling**: 现代 ViT (如 SigLIP) 往往去掉 CLS Token，直接对所有 Patch 的输出取平均 (Global Average Pooling)，效果更稳健。

### 5.2 SigLIP (Sigmoid Loss for Language Image Pre-training)
OpenVLA 的视觉编码器使用的是 **SigLIP** (来自 Google DeepMind)，而非传统的 CLIP。

#### 为什么不用 CLIP (Softmax Loss)?
传统的 CLIP 使用 **InfoNCE Loss** (基于 Softmax)，需要维护巨大的负样本对 (Negative Pairs)。
$$
L_{CLIP} = -\log \frac{e^{sim(I, T)}}{\sum_{j} e^{sim(I, T_j)}}
$$
- **缺点**: 必须进行全局归一化 (Softmax)，导致必须在所有 GPU 之间同步巨大的 Batch，通信开销极大。

#### SigLIP 的创新 (Sigmoid Loss)
SigLIP 将多分类问题转化为 **二分类问题 (Binary Classification)**。对每一对 (Image, Text)，独立判断它们是否匹配。
$$
L_{SigLIP} = - \log \sigma(sim(I, T)) - \sum_{j \neq i} \log (1 - \sigma(sim(I, T_j)))
$$
- **优势**:
    1.  **无需全局同步**: 每个 GPU 可以只处理自己的负样本，无需跨卡通信 Softmax 分母。
    2.  **Batch Size 独立**: 性能不再强依赖于超大 Batch Size。
    3.  **更高效**: 在相同计算资源下，SigLIP 的 Zero-shot 准确率显著高于 CLIP。

## 6. 面试常见问题

**Q: 为什么 ViT 需要比 ResNet 更多的数据?**
A: 因为 ViT 缺乏 **归纳偏置 (Inductive Bias)**。CNN 天生知道"相邻像素相关"和"平移不变"，而 ViT 必须从数据中自己学习这些规律。

**Q: 什么是 Patchify?**
A: 将一张图片切成一个个小方块 (e.g., 16x16 像素)，拉平成向量，作为 Transformer 的输入 Token。这相当于 NLP 中的分词 (Tokenization)。
