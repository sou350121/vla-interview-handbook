# VLA 模型全方位对比 (Model Comparison)

本表汇总了主流 VLA 模型的关键架构与特性，适合面试时快速回忆模型差异。

| 特性 | RT-1 | RT-2 | OpenVLA | Octo | Pi0 (π0) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **机构** | Google | Google DeepMind | Stanford | Berkeley | Physical Intelligence |
| **发布时间** | 2022.12 | 2023.07 | 2024.06 | 2023.10 | 2024.10 |
| **Backbone (Vision)** | EfficientNet-B3 | ViT-22B (PaLI-X) | SigLIP (ViT-L) | ViT-Base/Small | PaliGemma-3B (SigLIP) |
| **Backbone (Lang)** | Universal Sentence Encoder | PaLM-E / PaLI-X | Llama 2 7B | Transformer | Gemma 2B |
| **Action Space** | **Discrete** (Tokenized) | **Discrete** (Tokenized) | **Discrete** (Tokenized) | **Continuous** | **Continuous** |
| **Generation** | Classification (Softmax) | Classification (Softmax) | Classification (Softmax) | **Diffusion** (DDIM) | **Flow Matching** (ODE) |
| **Training** | BC (Behavior Cloning) | Co-fine-tuning | LoRA Fine-tuning | Diffusion Training | Flow Matching |
| **Inference Speed** | Fast (3Hz) | Slow (1-3Hz) | Medium (5-10Hz w/ Quant) | Slow (Diffusion) | **Fast** (10-50Hz) |
| **Pros** | 稳定，工业验证 | 语义理解极强，泛化好 | 开源 SOTA，生态好 | 动作平滑，多模态 | **高频控制**，物理理解强 |
| **Cons** | 泛化差，离散动作抖动 | 闭源，推理昂贵 | 离散动作，7B 仍较重 | 推理慢，语义弱 | 需新硬件支持 |

## 核心差异总结

1.  **RT-2 vs OpenVLA**:
    *   两者都使用 **Discrete Action Tokens**。
    *   RT-2 是闭源的超大模型 (55B+)，OpenVLA 是开源的 7B 模型 (Llama 2)，通过 LoRA 高效微调。

2.  **Diffusion (Octo) vs Flow (Pi0)**:
    *   两者都输出 **Continuous Actions**，精度高。
    *   Diffusion 走随机路径，推理慢；Flow Matching 走直线，推理快且支持高频控制。

3.  **Why Pi0?**:
    *   它是目前唯一结合了 **VLM 强语义** (PaliGemma) 和 **Flow Matching 高频控制** 的模型，解决了"脑子慢手快"的矛盾。
