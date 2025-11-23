# 真机与部署 (Real-world & Deployment)

本模块关注 VLA 算法在真实物理世界中的落地与应用。

## 目录
1. **[硬件选型与价格 (Hardware & Pricing)](./hardware.md)**
    - **灵巧手 (Dexterous Hands)** [重点]
    - 机械臂 (Arms)
    - 移动底盘与人形机器人
2. **[灵巧手落地实战指南 (Dexterous Hand Guide)](./dexterous_hand_guide.md)** [New]
    - 硬件集成避坑 (通信, 供电)
    - 软件栈 (Retargeting, Teleop)
    - 真实案例 (线缆管理, 散热)
3. **[模型优化与边缘部署 (Optimization)](./optimization.md)**
    - 量化 (Quantization): GPTQ, AWQ
    - 边缘推理: TensorRT-LLM, vLLM
3. **[Sim-to-Real (仿真到真机)](./sim_to_real.md)**
    - Domain Randomization
    - Reality Gap 应对策略

## 学习建议
- **硬件党**: 直接看 [硬件选型](./hardware.md)，了解最新的灵巧手和机器人平台。
- **工程党**: 重点研读 [模型优化](./optimization.md)，掌握如何在 Jetson 上跑大模型。
- **算法党**: [Sim-to-Real](./sim_to_real.md) 是必考题，务必理解 Domain Randomization 的细节。
