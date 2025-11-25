# 系统设计 (System Design)

本模块关注 VLA 系统的宏观架构设计，这是 Tech Lead 和 Staff Engineer 面试的核心考点。

## 目录

1.  **[数据闭环设计 (Data Pipeline Design)](./data_pipeline.md)**
    - 如何构建一个自动化的数据飞轮？
    - Auto-labeling (VLM 标注)
    - Active Learning (主动学习与难例挖掘)
    - Human-in-the-loop (人机回环)

2.  **[云端基础设施 (Cloud Infrastructure)](./cloud_infrastructure.md)**
    - 分布式训练架构 (FSDP, Megatron-LM)
    - 存储系统选型 (S3 vs Lustre)
    - 持续评估 (Continuous Evaluation)
    - 车队管理 (Fleet Management & OTA)

3.  **[评估系统设计 (Evaluation System)](./evaluation.md)**
    - Simulation Benchmark (仿真基准)
    - Real-world Proxy (真机代理指标)
    - A/B Testing & Canary Deployment

## 学习建议
- **关注 Scalability**: 所有的设计都要考虑 "如果机器人数从 10 台变成 1000 台，这个系统还能跑吗？"
- **关注 Automation**: 尽量减少人工介入。最好的系统是机器人自己收集数据、自己训练、自己变强。

---
[← Back to Main README](../README.md)
