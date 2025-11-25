# VLA 云端基础设施 (Cloud Infrastructure)

> **面试场景**: "我们需要训练一个 7B 参数的 VLA 模型，数据量 100TB，请设计训练集群架构。"

## 1. 训练集群架构 (Training Cluster)

### 1.1 计算资源 (Compute)
- **硬件**: NVIDIA H100 / A100 (80GB). 7B 模型全量微调至少需要 A100。
- **互联**: InfiniBand (IB) 或 RoCEv2。VLA 训练涉及大量的视频数据传输，网络带宽通常是瓶颈。

### 1.2 分布式训练策略 (Distributed Training)
单卡放不下，单机跑太慢。
- **FSDP (Fully Sharded Data Parallel)**: PyTorch 原生支持。将模型参数、梯度、优化器状态切分到所有 GPU 上。是目前微调 7B-70B 模型的首选。
- **DeepSpeed / Megatron-LM**: 针对超大模型 (e.g., >100B) 的 3D 并行 (Data + Tensor + Pipeline Parallelism)。对于 7B VLA，通常 FSDP 足够。
- **LoRA**: 如果资源受限，使用 LoRA 可以将显存需求降低 3-4 倍，允许在消费级显卡集群上训练。

### 1.3 存储系统 (Storage)
- **训练数据读取**: 
    - **问题**: 随机读取大量小文件 (图片/视频帧) 会导致 IOPS 爆炸。
    - **方案**: 
        - **WebDataset / TFRecord**: 将小文件打包成大文件 (Tar/Record)，顺序读取 (Sequential Read)。
        - **高性能文件系统**: FSx for Lustre / GPFS。提供极高的吞吐量。
        - **本地缓存**: 训练开始前将数据预加载到计算节点的 NVMe SSD。

## 2. 持续评估 (Continuous Evaluation)
## 2. 持续评估 (Continuous Evaluation)
> **Deep Dive**: 详见 **[评估系统设计 (Evaluation System)](./evaluation.md)**。

模型训好了，怎么知道它变强了？
- **Simulation Benchmark**: 在 Isaac Sim / ManiSkill 中运行自动化测试。
- **Real-world Proxy**: 计算 Action Prediction Loss，但要注意 **Goodhart's Law** (Loss 降低 $\neq$ 成功率提高)。
- **A/B Testing**: 灰度发布到真机车队。

## 3. 车队管理 (Fleet Management)
- **OTA (Over-the-Air) Updates**:
    - 使用 Kubernetes / KubeEdge 管理边缘节点。
    - **A/B Testing**: 先在 5% 的机器人上部署新模型，监控各项指标 (成功率、急停次数)，确认无误后再全量推。
- **Model Registry**:
    - 使用 MLflow / Weights & Biases 记录每一个版本的模型权重、训练数据版本、超参数。确保可追溯 (Reproducibility)。

## 4. 面试 Q&A
**Q: 训练过程中 GPU 利用率低 (Low GPU Utilization)，怎么排查？**
A: 
1. **DataLoader 卡顿**: CPU 处理图片太慢，GPU 在等数据。解决：增加 Worker 数量，使用 NVIDIA DALI 进行 GPU 解码。
2. **通信瓶颈**: 梯度同步慢。解决：检查 IB 网络，使用梯度累积 (Gradient Accumulation) 减少通信频率。

**Q: 如何降低训练成本？**
A: 
1. **Spot Instances**: 使用 AWS Spot 实例 (便宜 70%)，配合 Checkpoint 机制，挂了能从断点续传。
2. **混合精度训练 (Mixed Precision)**: 使用 BF16 (BFloat16)，速度快且显存省一半。

---
[← Back to System Design](./README.md)
