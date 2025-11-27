# 评估体系 (Evaluation Protocols)

在 VLA 领域，"怎么算做好了"是一个极其复杂的问题。与 CV/NLP 不同，机器人没有静态的 Test Set，必须在动态环境中评估。

## 1. 评估指标 (Metrics)

### 1.1. Success Rate (成功率)
- **定义**: 完成任务的次数 / 总尝试次数。
- **标准**: 通常尝试 20-50 次 (Real World) 或 100-1000 次 (Simulation)。
- **局限**: 0/1 二值指标，方差大。如果成功率是 50%，你很难说是模型不好还是运气不好。

### 1.2. Executable Rate (可执行率)
- **定义**: 机器人生成的动作是否违反了运动学约束 (如奇异点、碰撞、超出关节限位)。
- **作用**: 衡量模型的安全性。如果模型总是输出“自杀”动作，即使偶尔成功也不能用。

### 1.3. Interventions (人工干预次数)
- **定义**: 在长程任务中，人类需要接管多少次才能让机器人完成任务。
- **公式**: $\text{Intervention Rate} = \frac{\text{Interventions}}{\text{Time Steps}}$
- **适用**: 自动驾驶或长时间巡逻任务。

---

## 2. 仿真基准 (Simulation Benchmarks)

仿真评估是快速迭代的关键。

| Benchmark | 任务类型 | 特点 | 适用模型 |
| :--- | :--- | :--- | :--- |
| **CALVIN** | 桌面长序列 | 强调**语义理解**和**指令跟随** (e.g., "Push the block then open the drawer")。 | VLA (Language-Conditioned) |
| **ManiSkill** | 泛化抓取 | 强调**物体级泛化** (Part-level)。 | PointCloud/Visual Policy |
| **RLBench** | 多任务 | 基于 PyRep (CoppeliaSim)，任务种类多。 | Perceiver-Actor, RVT |
| **SIMPLER** | Sim-to-Real | 专门用于评估 VLA 模型 (如 RT-1, Octo) 在仿真中的表现是否能**预测**真机表现。 | Generalist VLA |
| **Libero** | 终身学习 | 强调知识迁移和抗遗忘能力。 | Lifelong Learning |

---

## 3. 真机评估 (Real World Evaluation)

真机评估是最终的金标准 (Gold Standard)，但极其昂贵。

### 3.1. 常见设置
- **Seen Objects / Scenes**: 在训练过的场景和物体上测试 (In-distribution)。
- **Unseen Objects**: 换颜色、形状、大小不同的物体 (Visual Generalization)。
- **Unseen Scenes**: 换桌布、光照、背景 (Robustness)。
- **Unseen Instructions**: 换一种说法 (e.g., "Pick up the apple" -> "Grab the red fruit") (Semantic Generalization)。

### 3.2. Checkpoint Selection (模型挑选)
在 CV 中，我们选 Val Loss 最低的模型。但在 VLA 中，**Val Loss 与 Success Rate 的相关性很弱**。
- **现象**: Loss 还在下降，但成功率可能已经崩了 (Overfitting to trajectory noise)。
- **策略**: 
    - 每隔固定 Step (e.g., 5000 steps) 保存 Checkpoint。
    - 在仿真中并行评估所有 Checkpoint。
    - 选仿真成功率最高的上真机。

---

## 4. 面试高频问题

**Q: 为什么 VLA 模型的 Val Loss 很低，但真机完全动不了？**
A: 
1. **分布偏移 (Distribution Shift)**: 训练数据的分布与真机推理时的分布不一致 (Covariate Shift)。
2. **多模态平均**: 如果使用 MSE Loss 训练，模型可能输出了两个正确动作的平均值 (e.g., 左边和右边都有路，模型直直撞向中间的墙)。
3. **死记硬背 (Overfitting)**: 模型记住了背景纹理，而不是物体的几何特征。

**Q: 如何公平地比较两个 VLA 模型？**
A: 必须控制变量：
1. **控制频率**: 都在 10Hz 下运行。
2. **控制数据**: 使用完全相同的数据集训练。
3. **控制评估环境**: 物体初始位置必须一致 (或遵循相同的随机分布)。
4. **统计显著性**: 至少跑 50 次以上，计算置信区间。
