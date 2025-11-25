# VLA (Vision-Language-Action) 算法岗面试手册

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **专注机器 VLA (Vision-Language-Action) 算法岗位的面试准备指南。**
> 从理论基础到真机部署，从灵巧手选型到 Sim-to-Real 实战。

## 📖 项目简介 (Introduction)

随着具身智能 (Embodied AI) 的爆发，VLA (Vision-Language-Action) 模型成为连接数字世界与物理世界的关键。本项目旨在为致力于进入该领域的算法工程师提供一份**全中文、实战导向**的面试与学习手册。

不同于通用的 CV/NLP 面试指南，本项目**聚焦于 Robotics 特有的挑战**：
- **Action Tokenization**: 如何将连续动作离散化？
- **Sim-to-Real**: 如何跨越仿真与真机的鸿沟？
- **Deployment**: 如何在边缘设备 (Jetson) 上部署大模型？
- **Hardware**: 灵巧手与机械臂的选型与控制。

## 🗺️ 路线图 (Roadmap)

本项目包含以下核心模块：

### 1. [理论基础 (Theory)](./theory/)
- **VLA 核心架构**: RT-1, RT-2, OpenVLA, Octo, Pi0 (Pi-Zero), Wall-X.
- **Backbone 对比**: [Transformer (ViT) vs CNN (ResNet)](./theory/transformer_vs_cnn.md) (含 SigLIP, Patchify).
- **Pi 系列解剖**: [Pi0 代码解构 (Flow Matching)](./theory/pi0_flow_matching.md) (含 OT-CFM, ODE Solvers), [Pi0.5 (Unified Model)](./theory/pi0_5_dissection.md), [Pi0.6 (Recap RL)](./theory/pi0_6_dissection.md).
- **动作生成范式**: [离散化 vs 扩散 (Diffusion Policy) vs 流匹配 (Flow Matching)](./theory/action_representations.md).
- **触觉感知**: [Tactile VLA](./theory/tactile_vla.md) (GelSight, VLA-Touch).
- **扩散策略深度**: [Diffusion Policy 详解](./theory/diffusion_policy.md) (含 EBM 视角, FiLM, Noise Schedulers).
- **性能优化**: [Flash Attention 原理](./theory/flash_attention.md) (Kernel Fusion, Recomputation).
- **文献综述**: [核心文献技术归纳](./theory/literature_review.md).
- **多模态大模型**: CLIP, LLaVA, Flamingo 原理回顾.
- **数据处理**: [RLDS 格式, 异构数据 Co-training](./theory/data.md).

### 2. [真机与部署 (Deployment)](./deployment/)
- **Pi0 真机部署**: [硬件配置与 Remote Inference](./deployment/pi0_deployment.md).
- **灵巧手实战**: [避坑指南 (通讯, 散热, 线缆)](./deployment/dexterous_hand_guide.md).
- **模型优化**: 量化 (AWQ, GPTQ), 剪枝.
- **边缘计算**: TensorRT, ONNX Runtime, vLLM 部署.
- **Sim-to-Real**: Domain Randomization, 迁移学习.
- **硬件选型**: **灵巧手 (Dexterous Hands)** 深度解析与价格参考.

### 3. [速查表 (Cheat Sheet)](./cheat-sheet/)
- **关键论文时间线**: 经典与最新 (近半年) 论文一览.
- **核心公式**: Attention, Diffusion, Control Theory.
- **模型对比**: 参数量, 训练数据, 性能指标.

### 4. [系统设计 (System Design)](./system-design/)
- **数据闭环**: [Data Pipeline Design](./system-design/data_pipeline.md) (Auto-labeling, Active Learning).
- **云端架构**: [Cloud Infrastructure](./system-design/cloud_infrastructure.md) (Distributed Training, Fleet Management).

### 5. [题库与实战 (Question Bank)](./question-bank/)
- **概念题**: 理论深度考察.
- **场景题**: "给定 100 条数据如何训练?"
- **代码题**: 坐标变换, 基础控制算法实现.
- **考官视角**: 面试官看重什么能力？

## 🚀 快速开始 (Getting Started)

建议按照以下顺序阅读：
1. 阅读 [理论基础](./theory/README.md) 建立 VLA 知识体系。
2. 查看 [硬件选型](./deployment/hardware.md) 了解行业现状与设备成本。
3. 浏览 [速查表](./cheat-sheet/README.md) 复习核心概念。
4. 挑战 [题库](./question-bank/README.md) 进行模拟面试。

## 🤝 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
- 补充最新的 VLA 论文解读。
- 分享你的真机部署经验。
- 提供更多面试真题。

## 📄 许可证 (License)

MIT License
