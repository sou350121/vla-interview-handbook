# 触觉感知与 VLA (Tactile VLA)

在机器人操作中，视觉 (Vision) 往往不足以完成所有任务。对于接触密集型 (Contact-rich) 任务（如在黑暗中摸索物体、精密装配、判断物体材质），**触觉 (Tactile)** 是不可或缺的模态。

2024-2025 年，VLA 领域开始爆发 "Vision-Tactile-Language-Action" 的研究，旨在赋予机器人"触觉语义"理解能力。

## 1. 为什么需要触觉? (Why Tactile?)
- **视觉遮挡 (Occlusion)**: 当机械手抓取物体时，手掌会挡住摄像头视线。此时只有触觉能提供反馈。
- **物理属性感知**: 视觉无法直接判断物体的软硬、摩擦力、重量。
- **微米级控制**: 视觉通常有毫米级误差，而触觉传感器 (如 GelSight) 可以提供微米级的纹理信息。

## 2. 核心传感器技术

### 2.1 主流触觉传感器详细参数对比

| 传感器 | GelSight Mini | DIGIT | GelSight 360 | BioTac | OptoForce | ReSkin |
|--------|---------------|-------|--------------|--------|-----------|---------|
| **厂商/机构** | MIT (GelSight Inc.) | Meta AI | GelSight Inc. | SynTouch | OptoForce Ltd. | Meta AI / ReSkin |
| **类型** | 光学触觉 (Vision-based) | 光学触觉 | 光学触觉 | 电阻式 + 流体 | 力/力矩传感器 | 磁性触觉 |
| **分辨率** | 640×480 (VGA) | 320×240 (QVGA) | 640×480 | N/A (19 电极) | N/A (力矢量) | N/A (磁场) |
| **空间精度** | ~30 μm | ~50 μm | ~40 μm | 1 mm | ~0.1 N | ~1 mm |
| **采样率** | 30 Hz (USB 2.0) / 60 Hz (USB 3.0) | 60 Hz | 30 Hz | 100 Hz | 100-1000 Hz | 50 Hz |
| **接触面积** | 15×15 mm | 18×14 mm | 20×20 mm | Φ 8 mm (圆形) | Φ 30 mm | 2×2 mm (单元) |
| **力量范围** | 0-5 N | 0-3 N | 0-10 N | 0-20 N | 0-150 N | 0-1 N |
| **输出数据** | RGB 图像 + 深度图 | RGB 图像 | RGB 图像 + 法向量 | 19 维向量 (压力/震动/温度) | 3D 力 + 3D 力矩 | 3D 磁场向量 |
| **材质感知** | ✅ 高 (纹理级) | ✅ 高 | ✅ 高 | ⚠️ 中 (电导率) | ❌ 低 | ⚠️ 中 |
| **滑移检测** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **硬件成本** | ~$500 USD | ~$300 USD | ~$800 USD | ~$2,500 USD | ~$1,200 USD | ~$50 USD (耗材) |
| **耐用性** | ⚠️ 中 (弹性体需定期更换) | ⚠️ 中 | ⚠️ 中 | ✅ 高 | ✅ 高 | ❌ 低 (易磨损) |
| **集成难度** | ⚠️ 中 (需 USB + 照明) | 🟢 低 (USB 即插即用) | ⚠️ 中 | 🔴 高 (专用驱动) | 🟢 低 (ROS 支持) | 🟢 低 (I2C/SPI) |
| **典型应用** | 精密装配、材质识别 | 抓取反馈、VLA 研究 | 全向接触感知 | 义肢、医疗机器人 | 工业力控 | 可穿戴、大面积覆盖 |

---

### 2.2 详细传感器解析

#### 2.2.1 GelSight 系列 (MIT / GelSight Inc.)
**原理**: 基于**光度立体视觉 (Photometric Stereo)** 的光学触觉传感器。

**核心组件**:
1. **弹性体 (Elastomer)**: 透明硅胶层 (厚度 ~3mm)，表面涂有反光涂层。
2. **LED 阵列**: 3 组不同角度的 RGB LED (通常为 120° 分布)。
3. **CMOS 摄像头**: 640×480 或更高分辨率，帧率 30-60 fps。
4. **算法核心**: 通过 3 张不同光照下的图像重建表面法向量 (Normal Map)。

**技术参数详解**:
- **深度重建精度**: ±10 μm (在 5mm × 5mm 区域内)。
- **法向量误差**: ±5° (与真实表面法向量的角度差)。
- **响应时间**: ~16 ms (60 Hz)，足够实时反馈。
- **弹性体材料**: Smooth-On Ecoflex 00-30 或 Dragon Skin 系列（Shore A 硬度 00-30）。

**数据输出格式**:
```python
{
  "rgb_image": np.ndarray(shape=(480, 640, 3), dtype=np.uint8),
  "depth_map": np.ndarray(shape=(480, 640), dtype=np.float32),  # 单位: mm
  "normal_map": np.ndarray(shape=(480, 640, 3), dtype=np.float32),  # (nx, ny, nz)
  "contact_mask": np.ndarray(shape=(480, 640), dtype=np.bool_)  # 接触区域掩码
}
```

**面试高频**:
- "GelSight 为什么用 3 个 LED？" → "用于光度立体重建。不同角度的光照产生不同的阴影，通过反向求解可以恢复表面法向量。"
- "GelSight 的主要限制？" → "弹性体容易磨损 (寿命 ~1000 次接触)，且对透明/反光物体效果差。"

---

#### 2.2.2 DIGIT (Meta AI)
**设计目标**: 低成本、易复现的 GelSight 替代品。

**技术差异**:
- **开源设计**: 3D 打印外壳 + 市售 USB 摄像头 (e.g., Raspberry Pi Camera V2)。
- **简化照明**: 只用单组 LED（白光），牺牲了法向量重建精度，但保留了纹理感知能力。
- **成本优化**: 通过批量生产弹性体模具，单个成本降至 $300 以下。

**数据处理流程**:
```python
# DIGIT 的典型数据处理流水线
import cv2
import numpy as np

def process_digit_frame(raw_image):
    # 1. 背景减除 (Background Subtraction)
    diff = cv2.absdiff(raw_image, background_ref)

    # 2. 边缘增强 (Edge Enhancement)
    edges = cv2.Canny(diff, threshold1=50, threshold2=150)

    # 3. 归一化到 [0, 1]
    normalized = diff.astype(np.float32) / 255.0

    return {
        "tactile_image": normalized,
        "edges": edges,
        "contact_area": np.sum(edges > 0) * pixel_area  # mm^2
    }
```

**Meta AI 的 VLA 集成案例**:
- 在 **Robot Parkour** 项目中，DIGIT 被用于检测机械腿与地面的接触状态。
- 在 **SSVL (Self-Supervised Visuo-Tactile Learning)** 中，DIGIT 与 RGB 相机的数据通过对比学习对齐。

---

#### 2.2.3 BioTac (SynTouch)
**特殊性**: 唯一的"仿生皮肤"传感器，模仿人类指尖。

**核心技术**:
1. **流体填充 (Liquid-filled Core)**: 内部充满导电液体，通过测量液体压力分布感知接触。
2. **19 个电极阵列**: 分布在传感器表面，测量局部阻抗变化 (Impedance)。
3. **震动传感器 (Vibration Sensor)**: 用于检测滑移 (Slip) 和纹理 (Texture)。
4. **温度传感器**: 测量物体的热传导率 (Thermal Conductivity)。

**输出数据维度**:
```python
biotac_data = {
  "pac": np.ndarray(shape=(19,), dtype=float),  # Pressure at Contact (静态压力)
  "pdc": np.ndarray(shape=(19,), dtype=float),  # DC Pressure (直流压力)
  "tac": np.ndarray(shape=(19,), dtype=float),  # AC Vibration (交流震动)
  "tdc": float,  # 温度 (°C)
  "electrode_impedance": np.ndarray(shape=(19,), dtype=float)  # 电极阻抗
}
```

**为什么在 VLA 中少见？**
- 输出是 19 维向量，不是图像，难以与 ViT/ResNet 等 CV 架构兼容。
- 需要专用的特征工程 (如 PCA 降维、FFT 频谱分析)，不如图像"即插即用"。

---

#### 2.2.4 ReSkin (Meta AI)
**革命性设计**: 基于**磁性**的超薄 (厚度 <2mm) 可穿戴触觉传感器。

**工作原理**:
1. 弹性体内嵌入微米级磁性颗粒 (Magnetic Particles)。
2. 外部 3 轴磁传感器 (Magnetometer, 如 MLX90393) 测量磁场变化。
3. 当弹性体形变时，磁性颗粒重新排列，改变磁场分布。
4. 通过机器学习模型 (MLP) 将磁场向量映射到接触力/剪切力。

**技术参数**:
- **厚度**: 仅 1.5 mm，可以贴在机械手指表面。
- **柔性**: 可以弯曲半径 <10 mm。
- **分辨率**: 单个传感器单元覆盖 2×2 mm，可以制作 10×10 的阵列。
- **寿命**: 约 5000 次接触后磁性颗粒会团聚 (Agglomerate)，需要更换。

**ReSkin 的 VLA 应用挑战**:
- **数据稀疏**: 输出是 3D 磁场向量，不像图像有丰富的空间结构。
- **标定 (Calibration)**: 每个新的 ReSkin 传感器都需要重新标定，因为磁性颗粒的分布略有差异。
- **未来方向**: 结合大规模传感器阵列 (如 100 个 ReSkin 单元)，可以生成"伪图像" (Pseudo-Image)，再喂给 VLA。

---

### 2.3 传感器选择决策树

```
需要触觉传感吗？
├─ 是 → 需要高精度材质识别吗？
│   ├─ 是 → GelSight (精度 ~30 μm, 成本高)
│   └─ 否 → 需要大面积覆盖吗？
│       ├─ 是 → ReSkin 阵列 (超薄, 可弯曲)
│       └─ 否 → DIGIT (性价比最高, VLA 首选)
└─ 否 → 纯视觉 VLA (如 RT-2, OpenVLA)

## 3. 最新模型进展 (2024-2025)

### 3.1 VLA-Touch (2025)
> **论文**: [VLA-Touch: Enhancing Generalist Robot Policies with Dual-Level Tactile Feedback](https://arxiv.org/abs/2502.xxxxx)
> **核心思想**: 双层反馈机制 (Dual-level Feedback)。

- **背景**: 现有的 VLA (如 RT-2) 缺乏触觉通道。直接微调大模型成本太高。
- **架构细节**:
    1.  **High-level Planning (VLM)**:
        - **Tactile Encoder**: 使用 **ResNet-50** 或 **ViT-B** 编码 GelSight 图像。
        - **Tactile-Language Model (TLM)**: 预训练一个 Decoder-only Transformer，将触觉 Embedding 翻译成自然语言描述 (e.g., "I feel a smooth, hard surface")。
        - **Prompting**: 将生成的触觉描述作为 Prompt 喂给 VLM (如 GPT-4o 或 Gemini)，辅助其进行推理。
    2.  **Low-level Control (Diffusion)**:
        - **Fusion**: 使用 **FiLM (Feature-wise Linear Modulation)** 将触觉特征注入到 Diffusion Policy 的 U-Net 中。
        - **Action Refinement**: 触觉信号主要用于修正动作的最后几毫米 (Contact Phase)，确保接触力适中。
- **优势**: 无需重新训练整个 VLA，即插即用。

### 3.2 OmniVTLA (2025)
> **论文**: [OmniVTLA: A Unified Vision-Tactile-Language-Action Model](https://arxiv.org/abs/2503.xxxxx)
> **核心思想**: 统一的视触觉语言动作模型 (Unified Vision-Tactile-Language-Action Model)。

- **架构细节**:
    - **Unified Tokenization**:
        - **Vision**: ViT Patch Embeddings.
        - **Tactile**: 同样使用 ViT 处理触觉图像，将其 Patch 化为 Tactile Tokens。
        - **Language**: 文本 Token。
    - **Semantic Alignment (Contrastive Learning)**:
        - 关键创新在于**语义对齐**。它不仅学习触觉的物理特征，还学习触觉的语义描述 (e.g., "slippery", "rough")。
        - **Loss Function**: $\mathcal{L} = \mathcal{L}_{action} + \lambda \mathcal{L}_{align}$。其中 $\mathcal{L}_{align}$ 是 InfoNCE Loss，拉近触觉 Embedding 与对应的材质描述文本 Embedding 的距离。
- **训练**: 使用大规模的多模态数据集 (包含图像、触觉图、语言指令、动作)。
- **能力**: 能够执行 "Pick up the softest object" (抓起最软的物体) 这种需要跨模态推理的任务。

## 4. 挑战与未来
1.  **数据稀缺**: 相比于图像数据，高质量的触觉-语言对 (Tactile-Language Pairs) 非常少。
2.  **Sim-to-Real**: 触觉仿真非常困难 (涉及复杂的软体形变)，目前主要依赖真机数据收集。
3.  **硬件成本**: 高分辨率触觉传感器 (如 GelSight) 依然昂贵且易损耗。

## 5. 深度解析: ResNet vs ViT for Tactile
在触觉 VLA 中，选择 ResNet 还是 ViT 作为触觉编码器 (Tactile Encoder) 是一个关键的设计决策。这不仅仅是"CNN vs Transformer"的问题，而是关乎**触觉信号的物理特性**如何被编码。

### 5.1 ResNet (CNN): 纹理与几何的专家
ResNet 在处理 GelSight 这类**基于光学 (Optical-based)** 的触觉传感器时表现出色，原因在于其**归纳偏置 (Inductive Bias)** 与触觉图像的特性高度契合。

*   **技术细节**:
    *   **平移不变性 (Translation Invariance)**: 触觉特征（如物体表面的凸起）可能出现在传感器的任何位置。ResNet 的**权重共享 (Weight Sharing)** 卷积核保证了无论凸起在哪里，提取的特征都是一致的。
    *   **局部性 (Locality)**: 触觉感知的核心是**接触 (Contact)**。接触通常发生在局部区域。ResNet 的卷积核 (e.g., 3x3) 强制模型关注局部像素的梯度变化，这对于检测**边缘 (Edges)**、**纹理 (Textures)** 和 **滑移 (Slip)** 至关重要。
    *   **层级特征 (Hierarchical Features)**: ResNet 通过 Pooling 不断下采样，自然地形成了从"微观纹理"到"宏观形状"的特征金字塔。这对于判断物体材质（微观）和抓取稳定性（宏观）都很有用。

### 5.2 ViT (Transformer): 全局接触与多模态统一
ViT 在 OmniVTLA 等最新模型中更受欢迎，主要是为了**多模态对齐**和**全局上下文**。

*   **技术细节**:
    *   **Patchify & Linear Projection**: ViT 将 $224 \times 224$ 的触觉图像切分为 $16 \times 16$ 的 Patches，展平后通过线性层映射为 Embedding。这一步彻底打破了像素的网格结构，使其能与文本 Token 在同一向量空间中交互。
    *   **全局感受野 (Global Receptive Field)**: Self-Attention 允许每一个 Tactile Patch 在第一层就与其他所有 Patch 交互。这对于理解**多点接触 (Multi-point Contact)** 非常关键。例如，当手指捏住物体时，指尖两侧的受力分布是相关的，ResNet 需要堆叠多层才能"看"到这种长距离关联，而 ViT 一眼就能看到。
    *   **位置编码 (Positional Encoding)**: 由于 ViT 没有卷积的归纳偏置，它必须依赖可学习的位置编码来理解"哪里是上，哪里是下"。在触觉中，绝对位置往往对应着机械手的具体部位 (e.g., 指尖 vs 指腹)，这对控制很重要。

### 5.3 核心差异对比表

| 特性 | ResNet (Tactile) | ViT (Tactile) |
| :--- | :--- | :--- |
| **归纳偏置** | 强 (平移不变, 局部性) | 弱 (需数据学习) |
| **擅长特征** | **高频纹理** (Texture), 边缘, 局部形变 | **低频分布** (Force Distribution), 全局接触模式 |
| **数据需求** | 低 (几千张图即可收敛) | 高 (需 MAE 预训练或 ImageNet 迁移) |
| **多模态融合** | 需通过 Pooling 压缩成向量后融合 | **Token 级融合** (可与 Text Token 拼接) |
| **典型应用** | 材质识别, 滑移检测 (Slip Detection) | 复杂操作策略 (Manipulation Policy), 跨模态推理 |

## 6. 面试常见问题
**Q: 触觉图像 (Tactile Image) 和普通 RGB 图像有什么区别?**
A: 触觉图像通常反映的是**几何形状 (Geometry)** 和 **受力分布 (Force Distribution)**，对光照变化不敏感，但对接触极其敏感。处理时通常不需要复杂的颜色增强，但需要关注纹理细节。

**Q: 如何将触觉融入 VLA?**
A: 最简单的方法是将触觉图像视为额外的视觉通道 (Concat)，或者使用 Cross-Attention 将触觉特征注入到 Policy 中。最新的趋势是像 OmniVTLA 一样进行多模态对齐。


---
[← Back to Theory](./README.md)
