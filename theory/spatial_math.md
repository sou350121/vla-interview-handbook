# 空间智能与坐标系 (Spatial Intelligence & Coordinate Systems)

对于 AI 背景的同学来说，机器人学最令人头大的往往不是深度学习模型，而是**坐标变换 (Coordinate Transformations)**。理解空间关系是训练 VLA 模型的基础。

## 1. 核心坐标系 (Core Frames)

在机器人操作中，我们通常涉及以下几个关键坐标系：

### 1.1. World Frame (世界坐标系)
- **定义**: 全局固定的参考系，通常是机器人底座固定的桌子角，或者房间的某个角落。
- **作用**: 多机器人协作时的统一基准。

### 1.2. Base Frame (基座坐标系)
- **定义**: 机器人底座 (Base Link) 的中心。
- **作用**: **绝大多数单臂机器人的 Action 都是相对于 Base Frame 定义的**。
- **注意**: 如果机器人是移动的 (Mobile Manipulator)，Base Frame 本身是在 World Frame 中移动的。

### 1.3. End-effector Frame (末端坐标系 / TCP)
- **定义**: 机械臂末端执行器 (Gripper) 的中心点 (Tool Center Point, TCP)。
- **作用**: 描述“手”的位置和朝向。
- **Delta Action**: 很多时候模型预测的是 TCP 相对于**当前时刻 TCP** 的移动量 (即在自己的坐标系下往前走 1cm)。

### 1.4. Camera Frame (相机坐标系)
- **定义**: 以相机光心为原点。
- **作用**: 视觉输入 (RGB/Depth) 最初都是在 Camera Frame 下的。需要通过 **Extrinsics (外参矩阵)** 变换到 Base Frame 或 World Frame。

---

## 2. 旋转表示 (Rotation Representations)

位置 (Position) 很好表示 $(x, y, z)$，但旋转 (Rotation) 的表示方式多种多样，各有优劣。

### 2.1. 欧拉角 (Euler Angles)
- **表示**: $(roll, pitch, yaw)$ 或 $(\alpha, \beta, \gamma)$。
- **优点**: 直观，人类易读。
- **缺点**: **万向节死锁 (Gimbal Lock)**。当两个轴重合时，会丢失一个自由度，导致数学奇异性。
- **VLA 适用性**: ❌ **不推荐**作为模型输出，因为不连续。

### 2.2. 旋转矩阵 (Rotation Matrix)
- **表示**: $3 \times 3$ 的正交矩阵 $R$。
- **优点**: 无奇异性，线性变换方便。
- **缺点**: 参数多 (9个)，且必须满足正交约束 ($R^T R = I$)，神经网络很难直接预测出完美的正交矩阵。
- **VLA 适用性**: ❌ 通常只用于中间计算，不作为输出。

### 2.3. 四元数 (Quaternion)
- **表示**: $(w, x, y, z)$ 或 $(x, y, z, w)$。
- **优点**: 
    - 无万向节死锁。
    - 紧凑 (4个参数)。
    - 计算插值 (SLERP) 非常平滑。
- **缺点**: 
    - **双倍覆盖 (Double Cover)**: $q$ 和 $-q$ 表示同一个旋转。这会导致模型训练时的 Loss 震荡 (模型不知道该输出 $q$ 还是 $-q$)。
    - 必须归一化 ($|q|=1$)。
- **VLA 适用性**: ✅ **主流选择之一**。训练时通常会做处理：如果预测的 $\hat{q}$ 与真值 $q$ 的点积小于 0，则将真值翻转为 $-q$。

### 2.4. 6D 旋转表示 (6D Rotation Representation)
- **表示**: 取旋转矩阵的前两列 $(r_1, r_2)$，共 6 个参数。
- **优点**: 
    - **连续性 (Continuity)**: 這是神经网络预测旋转的**最佳表示**。
    - 不需要正交约束，只需要通过 Gram-Schmidt 正交化即可还原为 $3 \times 3$ 矩阵。
- **VLA 适用性**: ✅✅ **SOTA 选择** (如 Diffusion Policy, Pi0)。

---

## 3. 坐标变换实战 (Transformations)

假设我们有一个点 $P$，它在相机坐标系下的坐标是 $P_{cam}$。我们想知道它在机器人基座坐标系下的坐标 $P_{base}$。

我们需要相机的**外参 (Extrinsics)**，通常表示为一个 $4 \times 4$ 的齐次变换矩阵 $T_{cam}^{base}$：

$$
T_{cam}^{base} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}
$$

变换公式：
$$
P_{base} = T_{cam}^{base} \times P_{cam}
$$

> **面试题**: 为什么 VLA 模型通常喜欢预测 **Delta Action** (相对移动) 而不是 **Absolute Action** (绝对坐标)？
> **答**: 
> 1. **泛化性**: Delta Action (如“向前移动 1cm”) 在不同位置、不同机器人尺寸下更通用。
> 2. **坐标系解耦**: 绝对坐标强依赖于 Base Frame 的标定，如果相机外参稍微有点误差，绝对坐标就全错了。Delta Action 更多依赖视觉闭环，对绝对定位精度要求低。
