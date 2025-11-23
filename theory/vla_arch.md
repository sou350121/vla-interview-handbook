# VLA 核心架构 (VLA Core Architectures)

本章节深入解析 Vision-Language-Action (VLA) 模型的核心架构演进，从 Google 的 RT 系列到开源的 OpenVLA，再到最新的 Physical Intelligence (Pi) 模型。

## 1. RT-1 (Robotics Transformer 1)
> **核心思想**: 将机器人控制建模为 Token 生成问题，使用 Transformer 处理多模态输入。

- **架构**: EfficientNet (Vision) + FiLM (Language conditioning) + TokenLearner + Transformer.
- **Action Tokenization**:
    - 将连续的动作空间 (x, y, z, roll, pitch, yaw, gripper) 离散化为 256 个 bins。
    - 输出是一个离散的 Token 序列，代表动作维度。
- **关键点**:
    - **TokenLearner**: 显著减少了视觉 Token 的数量 (e.g., 81 -> 8)，提高了推理速度。
    - **数据规模**: 130k episodes (17 months of data).
    - **局限性**: 泛化能力有限，主要依赖于大规模收集的机器人数据，缺乏互联网知识的迁移。

## 2. RT-2 (Robotics Transformer 2)
> **核心思想**: VLA = VLM + Action Tokens. 直接微调预训练的 VLM (如 PaLI-X, PaLM-E) 用于机器人控制。

- **架构**: 基于大规模 VLM (Vision-Language Model) 微调。
- **Co-fine-tuning**:
    - 混合训练：互联网 VQA 数据 + 机器人操作数据。
    - 机器人动作被表示为特殊的文本 Token (e.g., "1 128 90 ...")，与自然语言 Token 共享词表。
- **关键点**:
    - **涌现能力 (Emergent Capabilities)**: 能够理解未见过的指令 (e.g., "pick up the extinct animal" -> 抓恐龙玩具)，得益于 VLM 的语义理解能力。
    - **Chain-of-Thought**: 支持简单的推理步骤。
    - **局限性**: 推理速度慢 (大模型)，难以在边缘端实时运行。

## 3. OpenVLA / Octo
> **核心思想**: 开源、高效、基于 Diffusion 或 Llama 的 VLA 策略。

### Octo
- **架构**: 基于 Diffusion Policy 的 Transformer 架构。
- **特点**: 支持多种观察空间 (Proprioception, Images) 和动作空间。
- **训练**: 在 Open X-Embodiment (OXE) 数据集上训练。

### OpenVLA
- **架构**: 基于 Llama 2 (7B) + DINOv2 / SigLIP (Vision Encoder)。
- **Action Head**:
    - 并没有直接输出文本 Token，而是使用专门的 Action Head (Linear Layer) 预测去离散化的动作 Token。
    - 使用 **Action Detokenization** 还原为连续动作。
- **优化**: 支持 4-bit 量化 (QLoRA) 训练和推理，适合消费级显卡。

## 4. Physical Intelligence (Pi) Models
> **核心思想**: 通用机器人基础模型 (Generalist Robot Foundation Models)，强调跨形态 (Cross-Embodiment) 和物理世界的理解。

### π0 (Pi-Zero)
- **定位**: 基础 VLA 模型，旨在解决通用的物理操作问题。**已开源 (Open Weights)**。
- **架构特点**:
    - 类似于 RT 系列，但更强调对物理动力学的理解。
    - 可能采用了更高效的 Action Tokenizer，以适应高频控制需求。
- **数据**: 混合了多种机器人的数据 (Arms, Quadrupeds, Humanoids)。

### π0.5 (Pi-Zero-Point-Five)
- **核心升级**: **Open-world Generalization** (开放世界泛化)。
- **异构数据训练**:
    - 引入了大量的互联网视频数据 (YouTube) 和模拟数据。
    - 使用 **Cross-Embodiment Alignment** 技术，将不同机器人的动作空间对齐到统一的潜空间 (Latent Space)。
- **能力**: 能够处理未见过的物体和场景，Sim-to-Real 能力显著增强。

### π0.6 (Pi-Zero-Point-Six)
- **核心升级**: **RL (Reinforcement Learning) 强化** 与 **Action Expert**。
- **Backbone**: 升级为 **Gemma 3** (Google 最新开源模型) 作为语言基座，增强了指令遵循和推理能力。
- **Action Expert**:
    - 引入了专门的 "Action Expert" 模块（可能是 Mixture-of-Experts, MoE 架构），专门处理精细操作。
    - 解决了大语言模型在精细运动控制上的 "手笨" 问题。
- **RLHF for Robotics**:
    - 使用人类反馈 (Human Feedback) 或 成功/失败 信号进行强化学习微调。
    - 显著提升了长序列任务的成功率。

## 5. 模型对比总结 (Model Comparison)

| 模型 | 基础架构 | Action 输出 | 训练数据 | 优势 | 劣势 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RT-1** | EfficientNet + Transformer | Discrete Tokens | Robot Data Only | 推理快，稳定性高 | 泛化差，无语义推理 |
| **RT-2** | PaLI-X / PaLM-E | Text Tokens | Web + Robot Data | 语义理解强，Zero-shot | 推理慢，闭源 |
| **OpenVLA** | Llama 2 + SigLIP | Action Head | OXE Dataset | 开源，支持量化，易部署 | 性能略逊于闭源 SOTA |
| **π0.6** | Gemma 3 + Action Expert | Specialized | Cross-Embodiment + RL | 泛化强，精细操作好，**已开源** | 训练极其昂贵 |

## 面试高频考点
1. **Action Tokenization**: 为什么要离散化？连续回归 (Regression) 有什么问题？(答: 多模态分布处理能力)
2. **Co-fine-tuning**: 为什么要混合互联网数据？(答: 保持 VLM 的语义能力，防止灾难性遗忘)
3. **Sim-to-Real**: OpenVLA 如何在真机上部署？(答: 量化，VLM 蒸馏)
