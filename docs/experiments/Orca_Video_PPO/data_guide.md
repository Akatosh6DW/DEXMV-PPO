# 数据、权重与复现说明

[总目录](README.md) · [完整报告](report.md) · [完整数据表](all_results.md)

## 1. 数据分三层，不混着看

1. **来源与先验数据**：视频、三维跟踪、物体网格、参考NPZ、D01示范。解释“知识从哪里来”。
2. **训练产物**：每次运行配置、BC初始化、PPO检查点、开发曲线。解释“怎样学”。
3. **测试结果**：固定初态的逐回合指标、汇总、统计与视频。解释“最后表现如何”。

一次训练的100次测试不等于100次训练；一次视频的40次扰动回放也不等于40个来源视频。

## 2. 当前模型和数据接口

| 产物 | 原始位置 |
|---|---|
| 来源视频 | [HIC20 input_rgb.mp4](../../../artifacts/cube20/hic20/videos/20/input_rgb.mp4) |
| DexMV中间拟合 | [adroit_cube.npz](../../../artifacts/cube20/hic20/dexmv_retarget/adroit_cube.npz) |
| 手物坐标对应 | [O01来源目录](../../../artifacts/orca_collision_refit/O01_hic53_source_object/) |
| 当前物理模型 | [O20 cube.xml](../../../artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml) |
| 校正前参考 | [O20 reference.npz](../../../artifacts/orca_collision_refit/O20_hic53_elliptic/reference.npz) |
| 正式使用的校正参考 | [E01 candidate_reference.npz](../../../artifacts/orca_cube_acquisition/E01_flex50_refinement/candidate_reference.npz) |
| BC示范 | [D01 trajectories.npz](../../../artifacts/orca_cube_acquisition/D01_quality_demos/trajectories.npz) |
| 示范来源/筛选记录 | [D01清单副本](data/D01_dataset_manifest.json) |
| E01校正记录 | [E01报告副本](data/E01_refinement_report.json) |

这些文件名比泛指`demo_dataset.npz`更准确；正式链路实际读的是上述版本化产物。

### 参考NPZ

- `ctrl`：形状`(T,17)`，关节目标，单位rad。
- `time`：形状`(T,)`，秒，从0开始严格递增；加载时插值到50ms控制时刻。
- `initial_ctrl`：共同起手关节目标，不等于轨迹全部内容。
- `object_position/object_quaternion`：仿真物体位置和朝向。
- `object_extents_m/spatial_scale/object_to_world_rotation`：尺寸、空间比例、坐标变换；当前53mm任务强制比例1。
- E01额外保存`refinement_delta`和`initial_opening`，记录仿真校正与打开起手程度。

### D01示范NPZ

- `obs`：`(N,T,123)`，当时的Actor观测。
- `action`：`(N,T,17)`，归一化的完整目标动作，不是残差动作。
- `active`：`(N,T)`，标记有效回合时间步。
- 清单检查模型、参考、文件哈希及物理后端；按轨迹前75%训练、后25%验证。

BC学的是完整动作，因此训练器为`guided/bc`建立`pure`动作环境。不能把这份动作标签直接当成残差标签。

## 3. 本目录的CSV怎么用

以下成功率列保存0到1，例如0.851是85.10%；以`_pp`结尾的差值是百分点。单位为m、mm、s、N或rad的字段不能混用。

| 文件 | 行数 | 每一行代表什么 |
|---|---:|---|
| [aggregate.csv](data/aggregate.csv) | 60 | 一个阶段、条件、方法的汇总 |
| [per_seed.csv](data/per_seed.csv) | 429 | 一份训练策略在一个条件下的结果；确定性参考单列 |
| [effects.csv](data/effects.csv) | 4 | 同轮两种融合各自减纯PPO的效应量；p为未校正值 |
| [cross_round_descriptive.csv](data/cross_round_descriptive.csv) | 5 | 历史跨轮均值变化，只有描述含义 |
| [learning_curves.csv](data/learning_curves.csv) | 240 | 开发集、某种子、某检查点的原始成功率 |
| [sample_efficiency.csv](data/sample_efficiency.csv) | 120 | 某种子首次达到50%/80%的保存检查点，未达到留空 |
| [threshold_sensitivity.csv](data/threshold_sensitivity.csv) | 240 | 某条件/方法在2/3/4/5mm门槛下的描述结果 |
| [push_conditional.csv](data/push_conditional.csv) | 5 | S02施力前已有效的回合，施力后仍有效的比例及分母 |
| [feedback_subset.csv](data/feedback_subset.csv) | 9 | 已保存初态子集的输出响应，不是全测试总体 |
| [S03_historical_conditions.csv](data/S03_historical_conditions.csv) | 见文件 | 五个已完成历史条件的短/长预算结果，不是S04新测试 |

`aggregate.csv`主要列：`round/condition/method`标识来源；`policies/episodes`是策略数与回合数；`raw/clean/full_clean`是原始、撤托后清洁、全程穿透额外合格的成功率；`ci_low/ci_high`是95%区间；`hold_mean_sec/force_mean_n`是平均最长保持与平均回合指部峰值力；`seed_sd_pp`是种子成功率标准差。

`penetration_p95_mm`在aggregate中基于该组全部回合，在per_seed中基于单个种子的回合；二者并非必然相同。`force_clean_success_mean_n`只统计清洁成功回合，配合`clean_success_count`读；没有成功回合时为空，不填0伪装成低力。

`effects.csv`中的未校正p不可直接替代正式校正p。S02保留Holm-5主比较；S04两项Holm-2结果见总报告及原统计报告。额外补充比较不追认成原定主假设。

`S03_selected_recipes.json`中的`confirmation_not_started: true`是选择配方时的历史状态，不是当前是否训练完成的状态。该文件用于记录当时选出的参数，不随之后执行进度回写。

## 4. 原始逐回合数据在哪里

- [S02 episodes.csv](../../../artifacts/orca_benchmark_suite/S02/report/episodes.csv)：77600条。
- [S04 episodes.csv](../../../artifacts/orca_benchmark_suite/S04/report/episodes.csv)：4100条。
- [S02冻结协议](../../../artifacts/orca_benchmark_suite/S02/protocol.json)与[S04冻结协议](../../../artifacts/orca_benchmark_suite/S04/protocol.json)：任务、种子、文件指纹。
- [S03完整性说明](../S03/README.md)：中止后的已完成与未完成范围。

逐回合记录的身份由阶段、条件、方法、训练种子、初态种子共同决定。分析脚本检查重复、配对顺序和成功布尔逻辑。CSV中的`False`必须按布尔文本解析，不能用`bool("False")`，否则它会被错误当成真。

本目录复制的是轻量表格和图，不重复拷贝大体积轨迹、视频与权重。原始文件保留原路径，避免出现两个互相矛盾的“原始数据”。

## 5. 最终权重不是只有一个文件

S04为十个训练种子各保存了对应模型，没有把测试最好的一个冒充全部实验。举例，种子21001的三个163840步检查点为：

- [纯PPO](../../../artifacts/orca_benchmark_suite/S03/confirmation/ppo_tuned/s21001/step_000163840/params.pkl)
- [BC后PPO](../../../artifacts/orca_benchmark_suite/S03/confirmation/guided_tuned/s21001/step_000163840/params.pkl)
- [残差PPO](../../../artifacts/orca_benchmark_suite/S03/confirmation/residual_tuned/s21001/step_000163840/params.pkl)

它们是原有Brax/JAX参数文件，不是已导出到硬件的通用控制程序。加载需匹配网络结构、观测归一化方式、关节顺序、动作语义及任务配置。残差部署还必须携带参考轨迹与合成/限幅逻辑，不能只带Actor参数。

PKL仅加载本项目可信本地权重，不随意加载外来来源不明的文件。

## 6. 如何看效果

- [S04五方法固定案例视频](../../../artifacts/orca_benchmark_suite/S04/videos/index.md)：每种都有STL与碰撞体视图，手心视角。
- [S02六条件及消融视频](../../../artifacts/orca_benchmark_suite/S02/videos/index.md)。
- [本报告图表目录](figures/)：11组，每组PNG和PDF。

视频展示固定预定种子与初态，失败不换成成功；它是直观检查，不取代全部回合统计。STL检查外形，碰撞体检查真正参与接触的代理形状，二者重放同一轨迹。

## 7. 如何重建资料，不误启动训练

在主工程根目录：

```bash
conda activate aero_rl
# 从已有记录重算统计和图，不重新跑81700回合。
python scripts/analyze_orca_s02_s04.py
# 汇集本目录的数据、图与中文注释源码副本。
python scripts/package_orca_experiment_reading.py
# 检查注释及打包逻辑。
python -m unittest discover -s scripts -p 'test_orca_experiment_reading.py'
```

分析需NumPy、Matplotlib和Noto CJK字体；当前环境具备。打包只用Python标准库。人工编写的总报告不由分析脚本重写；数据变化后需要同步审核文字结论。

真要开始新训练，应另建版本化作业和输出目录，参考[S03复现说明](../S03/reproduction.md)。不要直接重跑旧`prepare`覆盖已冻结目录；该流程有意拒绝覆盖。DexMV准备环境叫`dexmv`，PPO主环境叫`aero_rl`；当前现成参考的训练/评估不需要同时激活两者。

## 8. 可追溯性

[manifest.json](manifest.json)记录打包输入、生成文件和注释副本哈希。数据目录还保留[统计分析原清单](data/analysis_manifest.json)。原始源码、协议和训练结果不因写注释而修改。

文档是整个研究的统一入口，实验来源仍分别标注。合并资料不意味着合并不兼容的训练种子、测试集或任务版本。
