# 硬件选型与价格参考 (Hardware & Pricing)

本章节汇总了 VLA 算法落地常用的硬件设备，重点关注 **灵巧手 (Dexterous Hands)**，并提供参考价格（人民币）和官方链接。

> [!NOTE]
> 价格仅供参考，实际成交价受汇率、关税、代理商折扣及配置（如是否含触觉传感器）影响较大。

## 1. 灵巧手 (Dexterous Hands) [核心关注]
灵巧手是 VLA 算法（尤其是精细操作）落地的关键执行单元。

| 厂商 | 型号 | 自由度 (DOF) | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inspire Robots**<br>(因时机器人) | **RH56DFX** | 6 | ~7.4万 - 15.5万 | 商业化成熟，集成度高，常用于科研与人形机器人集成。支持力控。 | [官网](https://www.inspire-robots.com/en/product/dexterous-hands) |
| **Unitree**<br>(宇树科技) | **Dex3-1** | 7 | ~5.7万 - 6万 | 三指设计，力控，适合抓取任务。 | [官网](https://www.unitree.com/products/dex3-1) |
| | **G1 Hand** | 6 | ~7.2万 | Unitree G1 人形机器人的配套手，性价比高。 | [官网](https://www.unitree.com/products/g1) |
| **Agibot**<br>(智元机器人) | **OmniHand** | 12 (Active) | ~1.45万 | **高性价比**。适合低成本科研验证，模块化设计。 | [官网](https://www.agibot.com/) |
| **LEAP Hand** | **LEAP Hand** | 16 | ~1.4万 - 2.1万 | **开源项目**。成本低，可自行 3D 打印组装。Sim-to-Real 友好，学术界常用。 | [官网](https://www.leaphand.com/) |
| **Wonik Robotics** | **Allegro Hand** | 16 (4指) | ~11万 - 18万 | 经典的科研用手，资料丰富，但价格较贵。 | [官网](https://www.wonikrobotics.com/allegro-hand) |
| **Shadow Robot** | **Shadow Hand** | 20 | ~85万+ | **高端触觉**。接近人手灵活度，常用于遥操作 (Teleoperation) 和高端触觉研究。 | [官网](https://www.shadowrobot.com/) |

### 选型建议
- **低成本科研 / 学生**: 推荐 **LEAP Hand** (开源自制) 或 **Agibot OmniHand**。
- **通用抓取 / 工业验证**: 推荐 **Unitree Dex3-1** 或 **Inspire RH56DFX**。
- **高端触觉 / 遥操作**: 推荐 **Shadow Hand** (预算充足) 或 **Wonik Allegro Hand**。

## 2. 机械臂 (Robotic Arms)
VLA 算法通常需要机械臂作为载体。

| 厂商 | 型号 | 自由度 | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trossen Robotics** | **WidowX 250** | 6 | ~1.9万 | **入门科研首选**。ALOHA 项目标配，Python API 友好，适合复现 Mobile ALOHA。 | [官网](https://www.trossenrobotics.com/widowx-250-robot-arm.aspx) |
| **Universal Robots** | **UR5e** | 6 | ~20万 - 30万 | **工业级协作**。稳定性极强，精度高，ROS 支持完善。 | [官网](https://www.universal-robots.com/products/ur5-robot/) |
| **Elephant Robotics** | **myCobot 280** | 6 | ~3000 | 桌面级教育机械臂，适合极低预算入门。 | [官网](https://www.elephantrobotics.com/en/mycobot-en/) |

## 3. 移动底盘 (Mobile Bases)
用于实现移动操作 (Mobile Manipulation)。

| 厂商 | 型号 | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **AgileX** (松灵) | **LIMO** | ~2万 - 4万 | 多模态（四轮/履带），适合教育和小型实验。 | [官网](https://www.agilex.ai/) |
| | **Scout Mini** | ~6万 | 越野能力强，适合室外环境。 | [官网](https://www.agilex.ai/) |
| | **Tracer** | ~7万 | 室内 AGV 底盘，适合平整地面。 | [官网](https://www.agilex.ai/) |

## 4. 人形/四足机器人 (Humanoid & Legged)
终极形态的具身智能载体。

| 厂商 | 型号 | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **Unitree** (宇树) | **Go2** | ~2万 (Air) / ~8-16万 (Edu) | 四足狗。Edu 版支持高算力开发。 | [官网](https://www.unitree.com/go2/) |
| | **G1** | ~10万 | **量产人形**。性价比极高，适合作为人形机器人研究平台。 | [官网](https://www.unitree.com/g1/) |
| **Fourier** (傅利叶) | **GR-1** | ~90万 - 100万 | 高端人形，主要用于康复和科研。 | [官网](https://www.fftai.com/) |
| **Astribot** (星尘) | **S1** | ~50万 - 60万 (预计) | 高性能，动作极快，演示效果惊人。 | [官网](https://astribot.com/) |

## 5. 中国头部机器人公司概览

| 公司 | 核心产品 | 领域 | 融资/规模 (Est.) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Unitree (宇树科技)** | Go2, B2, H1, G1 | 四足, 人形 | **B轮+ (独角兽)** | 行业领跑者，产品线最全，出货量大，性价比高。 |
| **UBTech (优必选)** | Walker S, Cruzr | 人形, 服务 | **上市 (HK.9880)** | "人形机器人第一股"，工业场景落地快，规模巨大。 |
| **Agibot (智元机器人)** | 远征 A2, 灵犀 X1 | 人形, 灵巧手 | **独角兽 (>70亿)** | 稚晖君创业项目，迭代速度极快，人才密度高。 |
| **Fourier (傅利叶智能)** | GR-1, 康复机器人 | 康复, 人形 | **D轮 (独角兽)** | 康复机器人起家，力控技术深厚，硬件自研能力强。 |
| **Galbot (银河通用)** | Galbot G1 | 具身智能 | **天使轮 (7亿)** | 专注于大模型与机器人的结合 (General Purpose)，泛化能力强。 |
| **X Square (自变量机器人)** | WALL-OSS | 具身智能 | **A+轮 (20亿)** | WALL-OSS 开发者，融资能力极强 (阿里/美团/联想)，专注于通用具身大模型。 |
| **Robot Era (星动纪元)** | STAR 1, 小星 | 人形 | **A轮 (超3亿)** | 清华交叉信息院孵化，算法强，STAR 1 奔跑速度惊人。 |
| **LimX Dynamics (逐际动力)** | CL-1, P1 | 人形, 四轮足 | **A轮** | 专注于运动控制 (Motion Control)，足式机器人技术硬核。 |
| **Deep Robotics (云深处)** | 绝影 X30, Dr.01 | 四足, 人形 | **B轮** | 工业级四足机器人领军者，电力巡检等场景落地多。 |
| **AgileX (松灵机器人)** | Scout, LIMO | 移动底盘 | **B轮** | ROS 生态支持最好，移动底盘市占率高，适合科研。 |
| **Dreame (追觅科技)** | 扫地机, 人形 | 消费, 人形 | **C轮 (百亿级)** | 消费电子巨头跨界，电机技术强，资金雄厚。 |
| **Xiaomi (小米)** | CyberDog 2, CyberOne | 四足, 人形 | **上市巨头** | 生态链完善，价格屠夫，推动了四足机器人的普及。 |
| **Xpeng (小鹏)** | PX5 (Iron) | 人形 | **上市巨头** | 车企造机器人，共享自动驾驶 AI 算力和制造体系。 |
| **Dataa (达闼机器人)** | Cloud Ginger | 服务人形 | **Pre-IPO (独角兽)** | 云端智能机器人 (Cloud Robot) 概念先行者，服务场景多。 |
| **Kepler (开普勒)** | Forerunner K2 | 人形 | **A轮** | 专注于商业化落地，对标 Tesla Optimus，设计硬朗。 |
| **Stardust (星尘智能)** | Astribot S1 | 人形 | **A轮** | 腾讯/百度背景，S1 展示了极高的操作精度和速度 (天下武功唯快不破)。 |


---

## 6. 国际头部机器人公司概览 (International)

| 公司 | 核心产品 | 领域 | 融资/规模 (Est.) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Tesla (US)** | Optimus (Gen 2) | 人形 | **上市巨头** | 拥有最强的量产制造能力和 FSD 数据闭环，行业风向标。 |
| **Figure AI (US)** | Figure 01/02 | 人形 | **B轮 ($675M)** | OpenAI/Microsoft/NVIDIA 投资，端到端模型能力强，落地 BMW 工厂。 |
| **Boston Dynamics (US)** | Atlas (Electric) | 人形, 四足 | **Hyundai 收购** | 运动控制 (Control) 的天花板，液压转电驱后更适合商业化。 |
| **Agility Robotics (US)** | Digit | 人形 (双足) | **B轮+ ($150M+)** | Amazon 投资，专注物流场景，Digit 已在亚马逊仓库试运行。 |
| **1X Technologies (Norway)** | Eve, Neo | 人形 (轮式/双足) | **B轮 ($100M)** | OpenAI 投资，Eve 是轮式人形，Neo 是双足。强调安全与家庭应用。 |
| **Sanctuary AI (Canada)** | Phoenix | 人形 | **B轮 ($140M+)** | 强调通用智能 (General Purpose)，Phoenix 拥有极强的灵巧手操作能力。 |
| **Apptronik (US)** | Apollo | 人形 | **A轮 ($14.6M+)** | NASA 背景，Apollo 设计紧凑，与 Mercedes-Benz 合作。 |
| **Physical Intelligence (US)** | π0 (Model) | 具身智能 | **Seed ($400M)** | 专注于通用机器人基础模型 (Software-first)，不造本体，赋能所有机器人。 |
| **Skild AI (US)** | Skild Brain | 具身智能 | **A轮 ($300M)** | 也是 Software-first，致力于构建通用的 "机器人大脑"。 |
| **Hugging Face (US/France)** | LeRobot | 开源社区 | **C轮 ($235M)** | 推动具身智能的开源化 (Open Source)，类似 Transformers 库的地位。 |

---
[← Back to Deployment](./README.md)

