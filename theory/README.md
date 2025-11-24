# 理论基础 (Theory)

本模块涵盖了机器 VLA 算法岗面试所需的核心理论知识。

## 目录
1. **[VLA 核心架构 (VLA Core Architectures)](./vla_arch.md)**
    - RT-1, RT-2
    - OpenVLA, Octo
    - **Physical Intelligence (Pi) Models (π0, π0.5, π0.6)**
    - **[Pi0 代码解构 (Flow Matching)](./pi0_flow_matching.md)**
    - **[Pi0.5 模型解剖 (Unified Model & Generalization)](./pi0_5_dissection.md)**
    - **[Pi0.6 模型解剖 (Recap RL & Self-Improvement)](./pi0_6_dissection.md)**
    - **[动作生成范式对比 (Tokenization vs Diffusion vs Flow)](./action_representations.md)**
    - **[扩散策略详解 (Diffusion Policy)](./diffusion_policy.md)**
    - **[核心文献技术归纳 (Literature Review)](./literature_review.md)** [New]
2. **[数据处理 (Data Processing)](./data.md)**
    - RLDS 格式详解
    - 数据加权与平衡策略
    - 动作空间对齐

## 学习建议
- **初学者**: 先阅读 RT-1 和 RT-2 的部分，理解 VLA 的基本范式 (Tokenization, Co-fine-tuning)。
- **进阶**: 重点关注 OpenVLA 的架构细节 (Llama + SigLIP)，这是目前开源界的基准。
- **前沿**: 了解 Pi 模型的最新进展 (RL 强化, 异构数据)，这是大厂面试的加分项。

## 推荐阅读论文
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
