# FAST: 高效动作 Token 化

> [!IMPORTANT]
> **FAST** (Frequency-space Action Sequence Tokenization，频域动作序列 Token 化) 是 **Physical Intelligence** 开发的一种高效动作 Token 化方法，专为解决 VLA 模型中连续动作转换为离散 token 的难题而设计。

## 1. 概述
在 VLA 模型中，动作的表示方式至关重要。传统的离散化方法（如简单分桶）在处理高频、灵巧的机器人操作时效果不佳。FAST 通过**离散余弦变换 (DCT)** 和**字节对编码 (BPE)** 的组合，实现了高效的动作压缩和 token 化。

-   **开发者**: Physical Intelligence
-   **论文**: [FAST: Efficient Action Tokenization for VLA Models (arXiv:2501.09747)](https://arxiv.org/abs/2501.09747)
-   **核心目标**: 将连续的机器人动作序列压缩为紧凑的离散 token，同时保持高频动作的精度。
-   **应用**: 已成功集成到 **OpenVLA** 中，显著提升训练速度（**最高 5 倍加速**）。

## 2. 核心问题：为什么需要 FAST？
传统 VLA 模型中的动作 Token 化面临几个挑战：

### 2.1. 简单分桶的局限性
-   **Token 数量爆炸**: 对于 7-DoF 机械臂，如果每个关节分成 256 个 bin，总 token 数可达 256^7，导致难以学习。
-   **高频动作丢失**: 简单分桶无法捕捉平滑、高频的轨迹变化（如快速折叠衣物、精细抓取）。

### 2.2. 连续动作的自相关性
-   机器人动作在时间上高度自相关（t 和 t+1 的动作非常相似）。
-   自回归模型（如 Transformer）在处理这种高相关性数据时效率低下。

FAST 通过**频域变换**解决了这些问题。

## 3. FAST 的核心技术

### 3.1. 离散余弦变换 (DCT)
FAST 借鉴了 **JPEG 图像压缩**的思想，使用 DCT 将时域的动作序列转换到频域。

**工作原理**:
1.  **输入**: 一段连续动作序列（例如，10 个时间步的 7-DoF 关节角度，形状为 `[10, 7]`）。
2.  **DCT 变换**: 对每个维度应用 DCT，将时域信号分解为频率分量。
3.  **低频保留**: 由于机器人动作大多是平滑的，主要能量集中在**低频分量**。FAST 只保留前 K 个低频系数（例如，K=4），丢弃高频噪声。
4.  **量化**: 将 DCT 系数量化为离散值，类似于 JPEG 的量化表。

**类比**:
-   **时域**: 音乐的波形（每个时刻的振幅）。
-   **频域 (DCT)**: 音乐的频谱（每个频率成分的强度）。
-   **压缩**: 只保留人耳能听到的频率，丢弃超高频。

### 3.2. 字节对编码 (BPE)
经过 DCT 量化后的系数序列仍然可能很长。FAST 进一步使用 **BPE**（类似 GPT 的 token 化方法）压缩这些系数。

**工作原理**:
1.  **初始词汇表**: 所有可能的量化 DCT 系数（例如，0-255）。
2.  **迭代合并**: 统计训练数据中最常见的系数对（例如，`[3, 5]`），将其合并为一个新 token（例如，`token_256 = [3, 5]`）。
3.  **重复**: 不断合并，直到词汇表达到目标大小（例如，8000 个 token）。

**效果**: 一个 10 步的动作序列可能被压缩为 **2-3 个 token**，而不是 70 个（10 × 7）。

### 3.3. FAST+ (Universal Tokenizer)
FAST+ 是在 **100 万+真实机器人动作序列**上预训练的通用 token 化器。
-   **跨平台**: 适用于不同的机器人（机械臂、人形机器人、移动机器人）。
-   **跨频率**: 适应不同的控制频率（10Hz 到 100Hz）。
-   **开箱即用**: 无需为每个新任务重新训练 tokenizer。

## 4. FAST 的优势
| 特性 | 简单分桶 | FAST (DCT + BPE) |
| :--- | :--- | :--- |
| **Token 数量** | 高（256^7）| **低（2-3 个 token/序列）** |
| **高频精度** | 差（抖动）| **强（平滑）** |
| **训练速度** | 慢 | **快（5 倍加速）** |
| **泛化能力** | 弱 | **强（FAST+ 跨任务泛化）** |

## 5. 在 OpenVLA 中的应用
FAST 已被集成到 **OpenVLA** 框架中：
-   **训练**: 使用 FAST tokenizer 将 Open X-Embodiment 数据集中的动作序列 token 化。
-   **推理**: 生成的 token 通过逆 DCT 转换回连续动作。
-   **效果**: 在折叠衣物、清理桌子等高频任务上，成功率提升 **30-50%**。

## 6. 与其他动作表示的对比
| 方法 | 原理 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **分桶 (Binning)** | 将连续值分成离散区间 | 简单 | Token 爆炸，精度差 |
| **扩散 (Diffusion)** | 通过去噪生成动作 | 平滑，高精度 | 推理慢（多步去噪）|
| **流匹配 (Flow Matching)** | ODE 求解器生成轨迹 | 快速，高质量 | 需要额外训练头 |
| **FAST (DCT + BPE)** | 频域压缩 + Token 化 | **快速，兼容自回归模型** | 需要预训练 tokenizer |

## 7. 面试要点
-   **DCT 是核心**: 记住 "像 JPEG 压缩图片一样压缩动作轨迹"。
-   **BPE 进一步压缩**: 类似 GPT 的 token 化，将 DCT 系数压缩为少量 token。
-   **5 倍加速**: FAST 使 OpenVLA 的训练速度提升 5 倍。
-   **FAST+ 是通用 tokenizer**: 在 100 万+真实机器人数据上预训练，跨平台泛化。
-   **适合自回归模型**: FAST 的 token 输出可以直接喂给 Transformer，无需修改架构。

## 8. 参考资源
-   **论文**: [FAST: Efficient Action Tokenization for VLA Models (arXiv:2501.09747)](https://arxiv.org/abs/2501.09747)
-   **官方博客**: [Physical Intelligence - FAST](https://physicalintelligence.company/blog/fast)
-   **GitHub**: [OpenVLA](https://github.com/openvla/openvla)
-   **Hugging Face**: [FAST+ Tokenizer](https://huggingface.co/pi0/FAST-plus)

