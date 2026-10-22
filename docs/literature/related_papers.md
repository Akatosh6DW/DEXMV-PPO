# 相关算法与文献清单（文献综述用）

整理：2026-10-09。所有 arXiv 编号已通过 arXiv API 核对标题与作者。
★ = 建议优先精读；"对应实验"指该文献主要支撑 S02/S03/S04 的哪部分论述。
会议/期刊只在已确认时写出，未确认的只写 arXiv 年份，写综述前再逐条核对正式出处。

## A. 强化学习基础算法

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| A1 ★ | Proximal Policy Optimization Algorithms | Schulman et al., arXiv 2017 | [1707.06347](https://arxiv.org/abs/1707.06347) | 所有在线训练的核心算法（clip 目标，本项目 ε=0.3） | S02–S04 |
| A2 | Trust Region Policy Optimization | Schulman et al., ICML 2015 | [1502.05477](https://arxiv.org/abs/1502.05477) | PPO 的前身；DexMV/DAPG 原版使用 NPG/TRPO 类方法 | 背景 |
| A3 | High-Dimensional Continuous Control Using Generalized Advantage Estimation | Schulman et al., ICLR 2016 | [1506.02438](https://arxiv.org/abs/1506.02438) | GAE 优势估计（λ=0.95, γ=0.99） | S02–S04 |
| A4 | Asymmetric Actor Critic for Image-Based Robot Learning | Pinto et al., RSS 2018 | [1710.06542](https://arxiv.org/abs/1710.06542) | actor 123 维 / critic 140 维特权信息的非对称结构 | S02–S04 |
| A5 | Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation | Freeman et al., NeurIPS 2021 D&B | [2106.13281](https://arxiv.org/abs/2106.13281) | 本项目使用的 PPO 实现 | S02–S04 |
| A6 | MuJoCo: A Physics Engine for Model-Based Control | Todorov, Erez, Tassa, IROS 2012 | [DOI](https://doi.org/10.1109/IROS.2012.6386109) | 仿真物理引擎（5 ms 子步） | S02–S04 |

## B. 灵巧手强化学习与硬件

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| B1 ★ | Learning Complex Dexterous Manipulation with Deep RL and Demonstrations (DAPG) | Rajeswaran et al., RSS 2018 | [1709.10087](https://arxiv.org/abs/1709.10087) | "BC 预训练 + 示范增强策略梯度"的经典范式；BC 后 PPO 路线的直接前身 | S02, S04 |
| B2 | Learning Dexterous In-Hand Manipulation | OpenAI et al., IJRR 2020 | [1808.00177](https://arxiv.org/abs/1808.00177) | 大规模 PPO 训练灵巧手、非对称 actor-critic、物理随机化 | 背景 |
| B3 | ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand | Christoph et al., IROS 2025 | [2504.04259](https://arxiv.org/abs/2504.04259) | 本项目使用的 Orca 手型 | 全部 |

## C. 从人类视频 / 人手动作学习灵巧操作（最直接的相关工作）

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| C1 ★ | DexMV: Imitation Learning for Dexterous Manipulation from Human Videos | Qin et al., ECCV 2022 | [2108.05877](https://arxiv.org/abs/2108.05877) | 视频→3D 手物位姿→重定向→示范→RL；本项目重定向工具来源 | 全部 |
| C2 | DexVIP: Learning Dexterous Grasping with Human Hand Pose Priors from Video | Mandikal & Grauman, CoRL 2021 | [2202.00164](https://arxiv.org/abs/2202.00164) | 人手姿态先验以奖励形式进入 RL（本项目则以动作结构进入） | S02 |
| C3 | Learning Generalizable Dexterous Manipulation from Human Grasp Affordance (ILAD) | Wu et al., CoRL 2022 | [2204.02320](https://arxiv.org/abs/2204.02320) | DexMV 后续，跨物体泛化 | S03 新物体 |
| C4 | VideoDex: Learning Dexterity from Internet Videos | Shaw et al., CoRL 2022 | [2212.04498](https://arxiv.org/abs/2212.04498) | 互联网视频作为先验 | 背景 |
| C5 | Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps (PGDM) | Dasari et al., ICRA 2023 | [2209.11221](https://arxiv.org/abs/2209.11221) | 预抓握初始化 + RL；与"共享起手姿态"设计相关 | S02, S04 |
| C6 | ViViDex: Learning Vision-based Dexterous Manipulation from Human Videos | Chen et al., ICRA 2025 | [2404.15709](https://arxiv.org/abs/2404.15709) | 人类视频参考 + 轨迹引导 RL | S02 |
| C7 | Hand-Object Interaction Pretraining from Videos | Singh et al., arXiv 2024 | [2409.08273](https://arxiv.org/abs/2409.08273) | 视频预训练 + RL 微调（对应 BC 后 PPO 思路） | S04 |
| C8 | Bridging the Human to Robot Dexterity Gap through Object-Oriented Rewards (HuDOR) | Guzey et al., arXiv 2024 | [2410.23289](https://arxiv.org/abs/2410.23289) | 单条人类视频 + 在线 RL 微调 | S04 |
| C9 | Object-Centric Dexterous Manipulation from Human Motion Data | Chen et al., arXiv 2024 | [2411.04005](https://arxiv.org/abs/2411.04005) | 人手运动数据 + RL 的分层方案 | 背景 |
| C10 ★ | DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References | Liu et al., ICLR 2025 | [2502.09614](https://arxiv.org/abs/2502.09614) | 追踪人类参考轨迹的 RL/IL 控制器 | S02 |
| C11 ★ | ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning | Li et al., CVPR 2025 | [2503.21860](https://arxiv.org/abs/2503.21860) | "人手轨迹模仿器 + 残差微调"，与本项目"参考 + 残差"最接近 | S02, S04 |
| C12 ★ | Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration | Lum et al., arXiv 2025 | [2504.12609](https://arxiv.org/abs/2504.12609) | 单条人类演示 → 仿真 RL，与本项目"一段视频"设定最接近 | S02, S04 |
| C13 | DeepMimic: Example-Guided Deep RL of Physics-Based Character Skills | Peng et al., SIGGRAPH 2018 | [1804.02717](https://arxiv.org/abs/1804.02717) | 参考动作 + RL 跟踪的范式源头 | 背景 |
| C14 | SFV: Reinforcement Learning of Physical Skills from Videos | Peng et al., SIGGRAPH Asia 2018 | [1810.03599](https://arxiv.org/abs/1810.03599) | 视频→姿态估计→参考→RL | 背景 |

## D. 跨手型重定向与人手数据

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| D1 | DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System | Handa et al., ICRA 2020 | [1910.03135](https://arxiv.org/abs/1910.03135) | 基于关键点向量的重定向优化 | 方法背景 |
| D2 | AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System | Qin et al., RSS 2023 | [2307.04577](https://arxiv.org/abs/2307.04577) | 通用重定向工具（dex-retargeting） | 方法背景 |
| D3 | Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO) | Romero et al., SIGGRAPH Asia 2017 | [2201.02610](https://arxiv.org/abs/2201.02610) | 人手参数模型 | 方法背景 |
| D4 | Capturing Hands in Action using Discriminative Salient Points and Physics Simulation (HIC) | Tzionas et al., IJCV 2016 | [1506.02178](https://arxiv.org/abs/1506.02178) | 视频来源数据集（HIC20） | 全部 |

## E. 示范与强化学习结合（BC 后 PPO 路线）

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| E1 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) | Ross et al., AISTATS 2011 | [1011.0686](https://arxiv.org/abs/1011.0686) | BC 的复合误差问题（仅 BC 不够的理论原因） | S02, S04 |
| E2 | Leveraging Demonstrations for Deep RL on Robotics Problems with Sparse Rewards (DDPGfD) | Vecerik et al., arXiv 2017 | [1707.08817](https://arxiv.org/abs/1707.08817) | 示范增强 RL 的早期代表 | 背景 |
| E3 | Overcoming Exploration in Reinforcement Learning with Demonstrations | Nair et al., ICRA 2018 | [1709.10089](https://arxiv.org/abs/1709.10089) | 示范缓解探索困难 | S04 |
| E4 | Jump-Start Reinforcement Learning | Uchendu et al., ICML 2023 | [2204.02372](https://arxiv.org/abs/2204.02372) | 用引导策略提升样本效率，对应"有限预算"论点 | S04 |
| E5 | Efficient Online Reinforcement Learning with Offline Data (RLPD) | Ball et al., ICML 2023 | [2302.02948](https://arxiv.org/abs/2302.02948) | 离线数据加速在线 RL | S04 |
| E6 ★ | Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem | Wołczyk et al., ICML 2024 | [2402.02868](https://arxiv.org/abs/2402.02868) | 预训练能力在 RL 微调中被遗忘，解释 S03 中 BC 后 PPO 长训退化 | S03 |

## F. 残差策略学习（参考加残差 PPO 路线）

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| F1 ★ | Residual Policy Learning | Silver et al., arXiv 2018 | [1812.06298](https://arxiv.org/abs/1812.06298) | 在基础控制器上学习修正量的原始提法 | S02–S04 |
| F2 ★ | Residual Reinforcement Learning for Robot Control | Johannink et al., ICRA 2019 | [1812.03201](https://arxiv.org/abs/1812.03201) | 传统控制器 + RL 残差，并行提出 | S02–S04 |
| F3 | Residual Reinforcement Learning from Demonstrations | Alakuijala et al., arXiv 2021 | [2106.08050](https://arxiv.org/abs/2106.08050) | 残差叠加在 BC 策略上，介于本项目两条路线之间 | S04 |
| F4 | From Imitation to Refinement — Residual RL for Precise Assembly (ResiP) | Ankile et al., arXiv 2024 | [2407.16677](https://arxiv.org/abs/2407.16677) | 模仿策略 + 残差 RL 精修 | S04 |
| F5 | Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model | Yuan et al., arXiv 2024 | [2412.13630](https://arxiv.org/abs/2412.13630) | 有界残差 + 受控探索（与本项目 0.08 rad 上限对应） | S02 |

## G. 实验评估与统计方法

| # | 论文 | 作者 / 出处 | 链接 | 与本项目的关系 | 对应实验 |
|---|---|---|---|---|---|
| G1 ★ | Deep Reinforcement Learning at the Edge of the Statistical Precipice | Agarwal et al., NeurIPS 2021 | [2108.13264](https://arxiv.org/abs/2108.13264) | 少量种子下的 RL 评估、bootstrap 区间 | S02–S04 |
| G2 | Deep Reinforcement Learning that Matters | Henderson et al., AAAI 2018 | [1709.06560](https://arxiv.org/abs/1709.06560) | 种子方差与基线调参公平性（S03 受控调参的动机） | S03 |
| G3 | An Introduction to the Bootstrap | Efron & Tibshirani, Chapman & Hall 1993 | — | 交叉 bootstrap 置信区间 | S02–S04 |
| G4 | A Simple Sequentially Rejective Multiple Test Procedure | Holm, Scand. J. Statist. 1979 | — | Holm 多重比较校正 | S02–S04 |
| G5 | Robust Tests for the Equality of Variances | Brown & Forsythe, JASA 1974 | — | 跨种子方差（稳定性）检验 | S02 |

## H. 综述（文献综述结构参考）

| # | 论文 | 作者 / 出处 | 链接 | 用途 |
|---|---|---|---|---|
| H1 | Learning by Watching: A Review of Video-based Learning Approaches for Robot Manipulation | Eze & Crick, IEEE Access 2024 | [2402.07127](https://arxiv.org/abs/2402.07127) | 视频学习方向的分类框架 |
| H2 | Dexterous Manipulation through Imitation Learning: A Survey | An et al., arXiv 2025 | [2504.03515](https://arxiv.org/abs/2504.03515) | 灵巧操作模仿学习的分类框架 |
