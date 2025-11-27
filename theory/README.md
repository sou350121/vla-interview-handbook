# 理论基础 (Theory)

本模块涵盖了机器 VLA 算法岗面试所需的核心理论知识，从基础架构到前沿算法。

# VLA 理论与核心算法 (Theory & Algorithms)

本章节深入探讨 Vision-Language-Action (VLA) 模型的理论基础、核心算法以及前沿模型架构。

> **学习建议**: 建议按照以下 **"数据 -> 架构 -> 算法 -> 模型"** 的逻辑顺序进行学习。

---

## 📚 Part 1: Foundations (基础基石)
*万丈高楼平地起，数据与动作空间是 VLA 的根基。*

- **[数据处理 (Data Processing)](./data.md)**
    - 主流格式对比 (RLDS vs LeRobot vs HDF5)。
    - PyTorch 训练流水线与数据加载。
    - 数据收集工具链 (VR vs Leader-Follower)。
- **[联合训练 (Co-training)](./co_training.md)**
    - 为什么需要混合互联网数据？(防止灾难性遗忘)。
    - 实施策略：数据配比与 Loss Masking。
- **[动作空间 (Action Representations)](./action_representations.md)**
    - 连续控制 (Continuous) vs 离散 Token (Discrete)。
    - 相对控制 (Delta) vs 绝对控制 (Absolute)。

## 🧠 Part 2: Architecture & Algorithms (架构与算法)
*理解模型是如何"思考"和"决策"的。*

### 核心架构
- **[VLA 架构概览 (VLA Architectures)](./vla_arch.md)**: VLM Backbone + Action Head 的主流设计范式。
- **[Transformer vs CNN](./transformer_vs_cnn.md)**: 为什么 Transformer 统治了机器人学习？

### 生成策略 (Policy Generation)
- **[Diffusion Policy](./diffusion_policy.md)**: 基于扩散模型的动作生成，解决多模态分布问题。
- **[Flow Matching (π0)](./pi0_flow_matching.md)**: 比 Diffusion 更快、更稳定的生成模型，π0 的核心。
- **[FAST (Action Tokenization)](./fast.md)**: 基于频率空间 (DCT) 的动作 Tokenization 技术。

### 效率优化
- **[Flash Attention](./flash_attention.md)**: 如何解决长序列 Transformer 的计算瓶颈。

## 🚀 Part 3: Advanced Topics (进阶专题)
*解决特定场景下的难题。*

- **[知识绝缘 (Knowledge Insulation)](./knowledge_insulation.md)**: 如何在微调时保护 VLM 的通用常识？
- **[触觉感知 (Tactile VLA)](./tactile_vla.md)**: 引入触觉模态，实现更精细的操作 (e.g., 盲盒摸索)。

## 🦁 Part 4: Model Zoo (模型详解)
*SOTA 模型的深度剖析与实战案例。*

> **[文献综述 (Literature Review)](./literature_review.md)**: **(必读)** VLA 发展史与主流模型全景图。

- **Physical Intelligence (π0 系列)**
    - **[π0.5 解析](./pi0_5_dissection.md)**: Flow Matching + VLA 的早期探索。
    - **[π0.6 解析](./pi0_6_dissection.md)**: 性能更强的迭代版本。
- **X Square (自变量)**
    - **[WALL-OSS](./wall_oss.md)**: 基于 Uni-CoT 的通用具身大模型。
- **Galaxea AI (星海图)**
    - **[Galaxea G0](./galaxea_g0.md)**: 独特的"小脑+大脑"双系统架构。

---
[← Back to Root](../README.md)

## 学习建议
- **初学者**: 先阅读 **基础架构** 部分，理解 VLA 的基本范式 (Tokenization, Co-fine-tuning)。
- **进阶**: 深入 **核心算法**，掌握 Diffusion 和 Flow Matching 的数学原理。
- **前沿**: 关注 **前沿专题** 和 **模型深度解析**，特别是 Pi 系列和触觉 VLA，这是大厂面试的差异化竞争点。

## 推荐阅读论文
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.xxxxx)
