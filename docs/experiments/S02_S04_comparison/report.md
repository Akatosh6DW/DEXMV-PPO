# S02与S04联合数据分析报告

生成日期：2026-10-09。所有图表由Python从现有数据重新计算，没有启动训练、修改评分或覆盖原实验结果。

## 1. 核心结论

两轮都支持有限在线预算下视频相关方法的平均成功率高于纯PPO。S04在新的匹配初态上的提升更明显，且两项原定主检验通过校正。
但S04并不是在相同种子、相同初态下只改一个参数重做S02，跨轮涨幅不能直接解释成某个超参数的因果收益。
S02主要补充多条件和消融证据，S04主要补充调优后配方在有限预算下的表现。两者应互补使用，不混成同一次实验。

| 方法 | S02原始条件清洁率 | S04新初态清洁率 | 跨轮观察差值 |
|---|---:|---:|---:|
| 纯PPO | 29.30% | 31.30% | +2.00个百分点 |
| BC后PPO | 68.90% | 84.00% | +15.10个百分点 |
| 参考+残差PPO | 69.80% | 85.10% | +15.30个百分点 |
| 仅BC | 51.55% | 63.50% | +11.95个百分点 |
| 零修正参考 | 68.00% | 66.00% | -2.00个百分点 |

## 2. 数据与可比性

| 项目 | S02 | S04 |
|---|---|---|
| 模型、物体、来源视频 | Orca、53mm/35g方块、HIC20 | 相同 |
| 主比较在线预算 | 163840步 | 同样163840步 |
| 权重来源 | 原配方独立训练 | S03调优后训练的对应早期检查点 |
| 训练种子 | 1001..1010 | 21001..21010 |
| 测试初态 | 40000..40199，每策略200个 | 81000..81099，每策略100个 |
| 本轮数据范围 | 六条件、消融，77600回合 | 原始条件五方法，4100回合 |
| 开发曲线 | 8个开发初态，40960步间隔 | 16个开发初态，81920步间隔 |
| 原定主检验校正 | B0五项Holm-5 | 两项Holm-2 |

合计81700条不同实验回合。本次逐行核验评分代数、回合身份及已有汇总；这不是重新仿真81700回合。
零参考是固定控制器，仅一组初态回放，不能假装成十个独立训练种子。BC与PPO的同种子可在同轮内配对，S02与S04不能按末尾编号强行配对。
配方变化：纯PPO保留原学习率/探索配方；BC后PPO学习率3e-5变为7.5e-6；残差探索标准差0.3变为0.7，并去掉前期接近/接触辅助奖励。
网络结构、控制接口及成功定义未因此变成新算法。两种融合的语义仍分别是先模仿再PPO，以及每步参考加PPO修正。

## 3. 主结果与种子稳定性

![主结果](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/01_nominal.png)

误差线是训练种子与共同测试初态两个维度的交叉bootstrap 95%区间。S02每策略200回合和S04每策略100回合不是同一组样本。
![种子稳定性](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/02_seed_stability.png)

每点为一份训练策略的成功率。连线只表示同轮同种子，不能把不同方法之间的每条灰线当作训练过程。
S04的BC后PPO相对PPO为9/10种子正差，残差为10/10种子正差；平均提升不能保证每次训练都更好。

| 轮次 | 方法 | 种子间标准差（百分点） | 最低种子成功率 | 最高种子成功率 |
|---|---|---:|---:|---:|
| S02 | 纯PPO | 37.40 | 0.00% | 96.00% |
| S02 | BC后PPO | 30.64 | 7.50% | 100.00% |
| S02 | 参考+残差PPO | 3.56 | 64.50% | 76.50% |
| S04 | 纯PPO | 19.89 | 1.00% | 62.00% |
| S04 | BC后PPO | 23.79 | 22.00% | 100.00% |
| S04 | 参考+残差PPO | 5.86 | 75.00% | 94.00% |

两轮内残差方法的种子波动均小于另外两种方法，说明本任务下训练结果更集中；这不是额外的方差显著性检验。残差自身的跨轮标准差从3.56增至5.86个百分点，不能说S04所有指标都改善。

| 同轮比较 | 均值差（百分点） | 95%差值区间 | 未校正配对p | 正/负/平种子 |
|---|---:|---|---:|---|
| S02 guided-ppo | +39.60 | [+10.95, +68.40] | 0.041016 | 7/3/0 |
| S02 residual-ppo | +40.50 | [+17.65, +62.10] | 0.017578 | 8/2/0 |
| S04 guided-ppo | +52.70 | [+29.10, +72.50] | 0.005859 | 9/1/0 |
| S04 residual-ppo | +53.80 | [+38.80, +68.20] | 0.001953 | 10/0/0 |

![差值](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/03_within_round_effects.png)

本图补充统一的效应量展示，不另造新的主检验。S02残差减PPO原定Holm-5 p=0.052734，未达0.05；S04两项原定Holm-2 p分别0.005859和0.003906。
两轮检验家族不同，不能把p值变小全归因于算法变强。S02 BC后PPO减PPO在这里是补充比较，不追认为S02原定主假设。
S04残差85.10%与BC后PPO84.00%很接近，本分析不据1.10个百分点断言残差优于BC路线。

## 4. 学习过程与预算

![学习曲线](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/04_learning_curves.png)

重新从S03训练日志提取S04的原始成功率，避免把S02原始率与S04清洁率混画。粗线是均值，细线是每个训练种子。
BC在横轴零点前已经进行了4000次监督更新；残差在零点前已经获得校正参考。因此零点不是所有方法都没有准备成本。
开发集不同且样本少，曲线主要用于观察早期能力、稳定性和退化，不作同一测试集的严格时间到达比较。
阈值表只记录第一次在保存检查点达到阈值的步数；未达到者记为右删失，CSV保留空值，不删除或填成0。

| 轮次 | 方法 | 达到50%的种子数 | 达到80%的种子数 |
|---|---|---:|---:|
| S02 | 纯PPO | 4/10 | 2/10 |
| S02 | BC后PPO | 10/10 | 9/10 |
| S02 | 参考+残差PPO | 10/10 | 6/10 |
| S04 | 纯PPO | 1/10 | 0/10 |
| S04 | BC后PPO | 10/10 | 9/10 |
| S04 | 参考+残差PPO | 10/10 | 10/10 |

## 5. S02多条件与消融

![六条件](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/05_s02_robustness.png)

这张图只属于S02；S04没有重测六个条件。半修正分支只有B0结果，其余格明确标未测。
视频相关方法在位置或姿态变化下不总保持原有优势；质量加倍明显伤害BC后PPO。低摩擦下零参考本身接近饱和，不能把全部高分都解释成PPO反馈的功劳。
![消融](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/06_s02_ablation.png)

S02原始条件：未校正参考加残差22.20%、线性参考加残差40.40%、完整残差69.80%，提示参考轨迹及仿真校正在这套实现中重要。
但完整残差相对校正参考零修正只从68.00%到69.80%，原检验不支持显著增益；执行时修正减半为65.40%。不能只比较最差参考就断言PPO修正贡献很大。
S04对应对照为仅BC63.50%→BC后PPO84.00%，零参考66.00%→残差85.10%。这两个差值是平均增益的描述，不在此新增显著性胜出声明。

## 6. 抓住、接触质量与失效

![失败分解](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/07_failure_modes.png)

失败分解是互斥的三类：清洁成功、原始成功但接触质量未过门槛、原始任务失败。不是靠总奖励判成功。
S04主三组原始率与清洁率几乎相同，因此相对PPO的大幅提升不是仅由3mm门槛造成。S02 BC后PPO撤托后清洁率68.90%，全程穿透合格率63.95%，前期接触仍需单独审查。
![接触质量](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/08_contact_and_hold.png)

每个箱图中的点是一份策略的回合汇总，不把200个相关回合当成200次独立训练。指部峰值力是活跃回合中记录的物理子步峰值，再对回合平均；不是电机额定力矩或同一时刻全部关节载荷。
图中力含成功和失败回合，较低力可能与较早失败有关，不能仅据低力推断更安全。逐种子CSV另列清洁成功条件下的力及成功分母，分母为0时留空。
穿透图使用每个策略回合最大穿透的P95，而官方成功要求每回合不超过门槛；P95低于3mm不代表全部回合合格。
![阈值](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/09_threshold_sensitivity.png)

2/3/4/5mm只作敏感性分析，保持原始成功和禁接触条件不变；正式评分仍为3mm，不选择最有利阈值替代。全部条件的阈值表均导出，图只展示原始条件。

## 7. 优势的边界和反馈机制

![长短预算](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/10_budget_boundary.png)

这是S04引用的S03历史数据，显式独立于S04新复测。短预算时五个条件有4/5两种融合均高于PPO，质量加倍是反例；长预算时只有1/5满足两者都高于PPO。
所以应主张匹配先验能减少在线探索需求，而不是所有预算、所有物理条件下都更强。BC后PPO长训退化提示需要分析遗忘和更新稳定性，不能通过挑中间最好点掩盖。
![反馈诊断](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/11_feedback_limits.png)

S02外力仅0.15N持续0.2秒。左图以施力前有效回合为分母；右图来自每策略前4个保存初态的配对诊断，分母明确显示，不推广成全测试集统计。
可以观察到输出随扰动改变，但这不足以证明闭环修正比固定偏置更有性能优势。S04未完成闭环与固定回放机制实验，不补造这项结论。

## 8. 如何组织实验结论

主结论：在已有匹配视频先验、固定Orca方块任务和163840步在线预算下，BC后PPO与参考加残差PPO均比纯PPO取得更高平均成功率；S04的两项原定主检验支持其提升。
机制证据：BC/零参考对照显示先验提供基础能力，进一步PPO学习在S04中带来平均增益；S02消融显示参考质量重要。
边界：跨轮变好不能只归因于调参；有限在线步数相同不代表BC、视频处理与开发搜索总成本相同；一条视频不代表跨视频泛化；没有真机或手指自碰撞安全证明。
后续若要定位改进原因，可在相同训练种子与初态下分别改变学习率、探索幅度、辅助奖励，建立单变量复现实验。本次不增加训练，先把已有证据说清楚。

## 9. 图表、数据与复现

全部图表提供PNG和PDF，CSV保留未四舍五入的数据。脚本只读取原结果，所有输入文件SHA256记录在manifest.json中。
```bash
conda activate aero_rl
python scripts/analyze_orca_s02_s04.py
python -m unittest discover -s scripts -p 'test_orca_s02_s04_analysis.py'
```

- [完整输入/输出清单](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/manifest.json)
- [逐条件汇总](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/tables/aggregate.csv)
- [逐种子诊断](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/tables/per_seed.csv)
- [同轮效应量](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/tables/effects.csv)
- [学习过程与首次达标](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/tables/sample_efficiency.csv)
- [穿透阈值敏感性](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/tables/threshold_sensitivity.csv)
- [S02原报告](../S02/README.md)
- [S04原报告](../S04/README.md)

### 图注索引

- **01_nominal**：均值及训练种子/共同初态交叉bootstrap 95%区间。跨轮只作描述，不是配对调参效应。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/01_nominal.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/01_nominal.pdf)
- **02_seed_stability**：线只连接同一轮内相同编号的训练种子；不跨S02/S04连线。黑横线为均值。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/02_seed_stability.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/02_seed_stability.pdf)
- **03_within_round_effects**：差值95%区间；p为十种子配对精确检验未校正值。正式结论沿用S02 Holm-5与S04 Holm-2，不另选校正口径。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/03_within_round_effects.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/03_within_round_effects.pdf)
- **04_learning_curves**：统一使用原始成功率，细线为十个种子。两轮开发初态和检查点间隔不同，不当成同一测试集学习速度的严格检验。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/04_learning_curves.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/04_learning_curves.pdf)
- **05_s02_robustness**：半修正仅在B0评估，其余格标未测而非0分。其他参考消融均保留。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/05_s02_robustness.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/05_s02_robustness.pdf)
- **06_s02_ablation**：不是一个变量的连续加法链。线性/未校正分支分别训练；修正减半是执行时干预。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/06_s02_ablation.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/06_s02_ablation.pdf)
- **07_failure_modes**：三个互斥类别总和为100%。原始任务失败不进一步臆分成碰撞/姿态/力量等未经标注的原因。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/07_failure_modes.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/07_failure_modes.pdf)
- **08_contact_and_hold**：每点为一个训练种子的回合汇总；力含失败回合，不能以低力直接证明安全。CSV另列成功条件下力及其分母。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/08_contact_and_hold.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/08_contact_and_hold.pdf)
- **09_threshold_sensitivity**：描述性敏感性补充，原始成功及禁接触规则保持不变；官方3mm评分未修改。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/09_threshold_sensitivity.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/09_threshold_sensitivity.pdf)
- **10_budget_boundary**：明确属于S04引用的S03历史补充，不是S04新做的五条件测试；长短点取同一次训练，非独立实验。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/10_budget_boundary.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/10_budget_boundary.pdf)
- **11_feedback_limits**：0.15N推力偏弱。右图仅前4个保存初态的有效配对，不是完整总体；有状态依赖输出不等于已证明反馈的因果性能增益。 [PNG](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/11_feedback_limits.png) / [PDF](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/11_feedback_limits.pdf)
