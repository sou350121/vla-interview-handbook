# VLA (Vision-Language-Action) 算法岗面试手册

![VLA Handbook Banner](./assets/banner.png)


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

## ✨ 项目亮点 (Highlights)

1. **全中文内容**: 所有文档均使用简体中文编写，专业术语保留英文对照。
2. **最新技术覆盖**:
    - 包含了 **Physical Intelligence (Pi)** 的 π0, π0.5, π0.6 模型解析。
    - 涵盖了 **OpenVLA**, **WALL-OSS** (X Square), **Galaxea G0** (星海图) 等开源 SOTA 模型。
    - 详解了 **FAST** 动作 Token 化（DCT + BPE，5倍训练加速）。
    - 深入讲解 **Knowledge Insulation**（梯度隔离，防止灾难性遗忘）。
3. **硬件选型指南**:
    - 重点加强了 **灵巧手 (Dexterous Hands)** 的介绍。
    - 提供了 **Unitree, Agibot, Fourier** 等中国头部机器人公司的详细参数与价格参考。
    - 新增 **国际机器人公司** 和 **亚洲机器人公司** 对比表。
4. **实战导向**:
    - 提供了 **Sim-to-Real** 的具体技术路线 (Domain Randomization, Co-training)。
    - 提供了 **边缘部署** 的实战代码片段 (vLLM, Quantization)。

## 📂 项目结构 (Project Structure)

```
/opt/vla-interview-handbook/
├── README.md                   # 项目主页 (Introduction & Roadmap)
├── theory/                     # 理论基础
│   ├── README.md               # 索引
│   ├── vla_arch.md             # VLA 核心架构 (RT-1, RT-2, OpenVLA, Pi, WALL-OSS)
│   ├── transformer_vs_cnn.md   # Backbone 对比 (ViT vs ResNet, SigLIP)
│   ├── action_representations.md # 动作生成范式 (Tokenization vs Diffusion vs Flow)
│   ├── fast.md                 # FAST 动作 Token 化 (DCT + BPE, 5倍加速)
│   ├── diffusion_policy.md     # 扩散策略详解 (DDPM, DDIM, EBM)
│   ├── flash_attention.md      # 性能优化 (Kernel Fusion)
│   ├── literature_review.md    # 核心文献技术归纳 (包含10个模型对比)
│   ├── pi0_flow_matching.md    # Pi0 代码解构 (Flow Matching)
│   ├── pi0_5_dissection.md     # Pi0.5 模型解剖 (Unified Model)
│   ├── pi0_6_dissection.md     # Pi0.6 模型解剖 (Recap RL)
│   ├── wall_oss.md             # WALL-OSS 深度解析 (Uni-CoT, X Square Robot)
│   ├── galaxea_g0.md           # Galaxea G0 双系统 VLA (星海图智能)
│   ├── knowledge_insulation.md # 知识绝缘技术 (防止灾难性遗忘)
│   ├── tactile_vla.md          # 触觉感知与 VLA
│   └── data.md                 # 数据处理 (RLDS, Co-training)
├── deployment/                 # 真机与部署
│   ├── README.md               # 索引
│   ├── hardware.md             # 硬件选型 (灵巧手, 机械臂)
│   ├── calibration.md          # 相机标定指南
│   ├── pi0_deployment.md       # Pi0 真机部署
│   ├── dexterous_hand_guide.md # 灵巧手部署实战
│   ├── optimization.md         # 模型优化 (量化, TensorRT)
│   └── sim_to_real.md          # Sim-to-Real 技术
├── system-design/              # 系统设计
│   ├── README.md               # 索引
│   ├── data_pipeline.md        # 数据闭环设计
│   ├── cloud_infrastructure.md # 云端基础设施
│   └── evaluation.md           # 评估系统设计
├── cheat-sheet/                # 速查表
│   ├── README.md               # 索引
│   ├── timeline.md             # 关键论文时间线
│   └── formulas.md             # 核心公式
└── question-bank/              # 题库与实战
    ├── README.md               # 索引
    ├── questions.md            # 面试真题
    ├── openvla_finetuning.md   # OpenVLA 微调实战
    └── interviewer_guide.md    # 考官视角指南
```

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
