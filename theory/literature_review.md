# VLA 文献核心技术归纳 (Literature Technical Review)

本章节对 VLA 领域的核心文献进行**深度技术归纳**，适合面试前快速复习模型细节。

## 1. Diffusion Policy (Chi et al., RSS 2023)
> **论文**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)

- **核心问题**: 解决传统 MSE 回归在多模态分布 (Multimodal Distribution) 下的平均值问题 (即"撞墙"问题)。
- **核心技术**: **DDPM (Denoising Diffusion Probabilistic Models)**。将动作生成建模为从高斯噪声中逐步去噪的过程。
- **Backbone**:
    - **CNN-based**: 1D Temporal CNN (类似 U-Net)，适合短时序。
    - **Transformer-based**: DiT (Diffusion Transformer)，适合长时序。
- **Action Space**: **连续空间 (Continuous)**。无离散化误差，精度极高。
- **Inference**: 迭代去噪。原始 DDPM 需 100 步，使用 **DDIM** 可加速至 10-15 步。
- **Deep Dive**:
    - **EBM 视角**: Diffusion 实际上是在学习能量地貌 (Energy Landscape)，相比 MSE 的单峰平均，它能捕捉多模态分布 (Multimodal Distribution)。
    - **Conditioning**: 通过 **FiLM** 层将语言/图像特征注入 U-Net。
- **Key Contribution**: 首次将生成式 AI (Generative AI) 引入机器人控制，完美解决了多解问题，并在高精度任务 (如穿针) 上表现卓越。

## 2. RT-2 (Google DeepMind, 2023)
> **论文**: [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)

- **核心问题**: 如何让机器人拥有互联网级别的语义理解能力 (泛化到未见过的物体/指令)。
- **核心技术**: **VLA (Vision-Language-Action)** = VLM + Action Tokens。
- **Backbone**: **PaLI-X (55B)** 或 **PaLM-E (12B)**。
- **Action Tokenization**:
    - **Uniform Discretization**: 将动作维度归一化并切分为 **256 个 Bins**。
    - **Text Mapping**: 将这些 Bins 映射为特殊的文本 Token (如 "1", "128")，与自然语言共享词表。
- **Training**: **Co-fine-tuning** (混合微调)。同时训练互联网 VQA 数据 (保持语义) 和机器人操作数据 (学习控制)。
- **Key Contribution**: 涌现出 **Semantic Reasoning** (语义推理) 能力。例如听到 "pick up the extinct animal" 能抓起恐龙玩具，尽管训练数据里只有 "pick up dinosaur"。

## 3. OpenVLA (Stanford, 2024)
> **论文**: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)

- **核心问题**: 复现 RT-2 的能力，但完全开源且高效。
- **核心技术**: **Parameter-Efficient Fine-Tuning (LoRA)**。
- **Backbone**:
    - **Language**: **Llama 2 7B**。
    - **Vision**: **SigLIP** (比 CLIP 更强的视觉编码器)。
    - **Projector**: 2-layer MLP (将视觉 Embedding 映射到语言空间)。
- **Action Output**:
    - 不同于 RT-2 直接输出文本，OpenVLA 使用专门的 **Action Head** (Linear Layer) 预测去离散化的动作 Token。
    - 依然是 **256-bin Discretization**。
- **Optimization**: 支持 **4-bit Quantization (QLoRA)**，使得 7B 模型可以在消费级显卡 (如 RTX 3090/4090) 上运行。
- **Key Contribution**: 提供了第一个性能接近闭源 SOTA 的开源 VLA 模型，并构建了完整的开源训练/部署生态。

## 4. Pi0 (Physical Intelligence, 2024)
> **论文**: [π0: A Generalist Robot Foundation Model](https://www.physicalintelligence.company/blog/pi0)

- **核心问题**: 解决 VLM 推理速度慢、难以进行高频 (50Hz) 连续控制的问题。
- **核心技术**: **Flow Matching (流匹配)**。
- **Backbone**: **PaliGemma 3B** (Google 的轻量级 VLM)。
- **Action Space**: **连续空间 (Continuous)**。
    - 不同于 RT-2/OpenVLA 的离散 Token，Pi0 输出连续动作，避免了量化误差。
- **Inference**: 使用 ODE Solver (常微分方程求解器)。相比 Diffusion 的随机游走，Flow Matching 走直线，**1-10 步**即可生成高质量动作。
- **Deep Dive**:
    - **OT-CFM**: 基于 Optimal Transport 构造直线路径 (Wasserstein Geodesic)。
    - **ODE Solver**: 训练时学习向量场，推理时使用 **Euler** (极速) 或 **Heun** (高精) 求解。
- **Key Contribution**: 结合了 VLM 的语义理解和 Flow Matching 的高频精细控制，实现了"大脑"与"小脑"的统一。

## 5. Pi0.5 (Physical Intelligence, 2025)
> **核心定位**: **Open-World Explorer (开放世界探险家)**

- **核心问题**: 解决机器人无法在从未见过的环境 (Open World) 中泛化的问题。
- **核心技术**: **Unified Model with Hierarchical Inference**。
- **架构创新**:
    - **Latent Thought**: 模型内部生成隐式的高层语义子任务 (Semantic Subtask)，再解码为底层动作。
    - **Hybrid Architecture**: 训练时使用 **FAST Tokenizer** (离散) 加速，推理时使用 **Flow Matching** (连续) 微调。
- **Data Strategy**: **Co-training**。混合 Robot Data (高质量) + Internet Videos (世界模型) + Simulation Data (长序列逻辑)。
- **Key Contribution**: 实现了跨形态 (Cross-Embodiment) 的 Zero-shot 迁移，并显著提升了长序列任务的成功率。

## 6. Pi0.6 (Physical Intelligence, 2025)
> **核心定位**: **Self-Improving Master (自我进化大师)**

- **核心问题**: 如何超越人类示教的上限，实现极致的熟练度 (Proficiency)。
- **核心技术**: **Recap Algorithm (Offline RL)**。
- **架构升级**:
    - **5B Backbone**: 更强的语义理解。
    - **Action Expert**: 独立的高频动作生成模块 (小脑)，专门负责精细操作。
- **Recap 机制**:
    - 学习失败轨迹 (Failure Cases)，通过 Offline RL 抑制错误动作，奖励成功动作。
    - 实现了 **Data-Driven Self-Improvement**。
- **Key Contribution**: 证明了机器人可以通过自我复盘 (Recap) 在操作速度和鲁棒性上超越人类专家。

## 8. FAST (Physical Intelligence, 2025)
> **论文**: [FAST: Efficient Action Tokenization for VLA Models (arXiv:2501.09747)](https://arxiv.org/abs/2501.09747)

- **核心问题**: 传统动作 token 化方法（简单分桶）无法处理高频、灵巧的机器人操作。
- **核心技术**: **Frequency-space Action Sequence Tokenization (DCT + BPE)**。
- **工作原理**:
    - **DCT (离散余弦变换)**: 将时域动作序列转换到频域，只保留低频系数（压缩比 2.5:1）。
    - **BPE (字节对编码)**: 类似 GPT，将常见 DCT 系数组合合并为单个 token（压缩比 2.3:1）。
- **效果**: 一个 10 步动作序列从 70 个 token 压缩为 **2-3 个 token**。
- **FAST+**: 在 100 万+真实机器人数据上预训练的通用 tokenizer，跨平台泛化。
- **Key Contribution**: 使 OpenVLA 训练速度提升 **5 倍**，同时保持高频动作精度。

## 9. Galaxea G0 (星海图智能, 2024)
> **论文**: [Galaxea Open-World Dataset and G0 Dual-System VLA Model (arXiv:2509.00576)](https://arxiv.org/abs/2509.00576)

- **核心问题**: 单一 VLA 模型难以同时处理长时域任务的高层规划和低层控制。
- **核心技术**: **Dual-System Architecture (双系统架构)**。
- **架构设计**:
    - **G0-VLM**: 负责多模态规划和高层推理（大脑）。
    - **G0-VLA**: 负责细粒度执行和低层控制（小脑）。
- **训练策略**: **三阶段课程学习**
    1. 跨具身预训练（学习通用世界知识）
    2. 单具身预训练（适配特定机器人）← 核心阶段
    3. 任务后训练（精调复杂技能）
- **Galaxea Open-World Dataset**: 500+ 小时，50 个真实场景，统一具身（R1 Lite），精确子任务标注。
- **Key Contribution**: 在长时域移动操作任务上表现突出，泛化能力强，可解释性高（子任务可见）。

## 10. Knowledge Insulation (Physical Intelligence, 2024)
> **技术**: Pi0 的梯度隔离训练方法

- **核心问题**: VLA 微调时，新增的连续动作专家会破坏 VLM 的预训练语义知识（灾难性遗忘）。
- **核心技术**: **Gradient Isolation (梯度隔离)**。
- **工作原理**:
    - **VLM 分支**: 学习离散动作 token（保持语义理解）。
    - **动作专家分支**: 学习连续动作（使用 `.detach()` 阻止梯度回传到 VLM）。
- **效果**: VLM 的语义知识被"绝缘"保护，同时动作专家独立学习连续控制。
- **Key Contribution**: 防止灾难性遗忘，加速训练，提升泛化能力，为持续学习打好基础。

## 总结对比表 (Summary Table)

| 特性 | Diffusion Policy | RT-2 | OpenVLA | Pi0 | WALL-OSS | Galaxea G0 | FAST |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n| **核心机制** | Denoising | Token Prediction | Token + LoRA | Flow Matching | **Uni-CoT + Dual Heads** | **Dual-System** | **DCT + BPE** |\n| **动作空间** | 连续 | 离散 (256) | 离散 (256) | 连续 | 连续 + 离散 | 连续 | **离散 (极度压缩)** |\n| **Backbone** | CNN/ViT | PaLI-X (55B) | Llama 2 (7B) | PaliGemma (3B) | VLM | **VLM + VLA** | **Tokenizer** |\n| **推理速度** | 慢 (100 步) | 极慢 | 中等 | 快 (1-10 步) | 快/精 | 稍慢 (两阶段) | **极快 (5x)** |\n| **语义能力** | 弱 | 极强 | 强 | 强 | **强 (CoT)** | **强 (分离 VLM)** | N/A |\n| **适用场景** | 精细操作 | 高层规划 | 通用操作 | 通用控制 | 长序列推理 | **长时域移动操作** | **高频 Token 化** |\n| **核心优势** | 多模态分布 | 语义涌现 | 开源生态 | 高效推理 | **统一思维链** | **分层解耦** | **压缩效率** |


---
[← Back to Theory](./README.md)

