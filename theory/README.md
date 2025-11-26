# 理论基础 (Theory)

本模块涵盖了机器 VLA 算法岗面试所需的核心理论知识，从基础架构到前沿算法。

## 目录

### 1. 基础架构 (Foundations)
构建 VLA 模型的基石，包括骨干网络选择、动作表示和数据处理。
- **[VLA 核心架构 (VLA Core Architectures)](./vla_arch.md)**: RT-1, RT-2, OpenVLA, WALL-OSS 等主流模型概览。
- **[Backbone 对比: Transformer vs CNN](./transformer_vs_cnn.md)**: 为什么 ViT 成为主流？ResNet 还有机会吗？
- **[动作生成范式 (Action Representations)](./action_representations.md)**: Tokenization (离散) vs Diffusion (连续) vs Flow Matching (流匹配)。
- **[FAST 动作 Token 化](./fast.md)**: DCT + BPE 压缩，5 倍训练加速，OpenVLA 的秘密武器。
- **[数据处理 (Data Processing)](./data.md)**: RLDS 格式、数据加权与平衡策略。

### 2. 核心算法 (Core Algorithms)
深入理解驱动 VLA 的数学原理。
- **[扩散策略详解 (Diffusion Policy)](./diffusion_policy.md)**: DDPM/DDIM, Noise Schedulers, EBM 视角, FiLM Conditioning。
- **[Pi0 代码解构 (Flow Matching)](./pi0_flow_matching.md)**: Time Embeddings, CFG, OT-CFM, ODE Solvers (Euler/Heun)。

### 3. 模型深度解析 (Model Deep Dives)
针对 Physical Intelligence (Pi) 系列模型的详细拆解。
- **[Pi0.5 模型解剖 (Unified Model)](./pi0_5_dissection.md)**: 统一高层规划与底层控制，开放世界泛化。
- [Pi0.6 模型解剖 (Recap RL)](./pi0_6_dissection.md): 引入 Offline RL (Recap) 进行自我进化。
- **[WALL-OSS 深度解析 (Uni-CoT)](./wall_oss.md)**: 统一跨层思维链 (Uni-CoT) 与流匹配控制 (Flow Matching) 的完美结合。

### 4. 前沿专题 (Advanced Topics)
面试加分项，展示对最新技术的追踪。
- **[触觉感知与 VLA (Tactile VLA)](./tactile_vla.md)**: VLA-Touch, OmniVTLA, 视触觉融合技术。
- **[Flash Attention 优化原理](./flash_attention.md)**: 如何解决 Transformer 的内存瓶颈，实现长序列推理。

### 5. 综合综述 (Literature Review)
- **[核心文献技术归纳](./literature_review.md)**: 汇总了 Diffusion Policy, Pi0, Wall-X 等关键论文的技术细节。

## 学习建议
- **初学者**: 先阅读 **基础架构** 部分，理解 VLA 的基本范式 (Tokenization, Co-fine-tuning)。
- **进阶**: 深入 **核心算法**，掌握 Diffusion 和 Flow Matching 的数学原理。
- **前沿**: 关注 **前沿专题** 和 **模型深度解析**，特别是 Pi 系列和触觉 VLA，这是大厂面试的差异化竞争点。

## 推荐阅读论文
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.xxxxx)
