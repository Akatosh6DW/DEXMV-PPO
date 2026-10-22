# S02 方块抓握实验资料总目录

本轮实验的说明文档统一放在本文件夹。建议先看结果，再查实验细节与原始数据。
已完成50项训练、388批评估、77600个测试回合、60段视频；六组均为同一方块任务的条件变化。

## 核心文档

| 文档 | 内容 |
|---|---|
| [阶段实验报告（中期）](S02_experiment_report.md) | 论文格式：核心亮点、方法、结果与消融、稳定性与鲁棒性分析、下一阶段计划 |
| [最终结果与消融分析](orca_benchmark_suite_results.md) | 六条件成绩、消融贡献、支持与不支持的结论、后续研究方向 |
| [实验详细信息与数据指南](orca_benchmark_data_guide.md) | 视频来源、重定向、PPO接口、训练参数、奖励、指标、术语与复算命令 |
| [实验协议](orca_benchmark_suite_protocol.md) | 比较方法、训练/测试种子、预算、评分规则和统计计划 |
| [执行与验证记录](orca_benchmark_suite_execution.md) | 测试、训练、中断恢复、阶段结果、分析补充与最终验收 |

## 统计与原始数据

自动生成的资料保留在原artifacts目录，避免影响复算脚本；以下入口可直接打开。

| 资料 | 内容 |
|---|---|
| [完整统计报告](../../../artifacts/orca_benchmark_suite/S02/report/results.md) | 所有方法成绩、置信区间与校正后的比较 |
| [逐回合CSV](../../../artifacts/orca_benchmark_suite/S02/report/episodes.csv) | 77600个回合的详细指标 |
| [逐种子CSV](../../../artifacts/orca_benchmark_suite/S02/report/per_seed.csv) | 每个策略在测试集上的表现 |
| [汇总CSV](../../../artifacts/orca_benchmark_suite/S02/report/aggregate.csv) | 方法和条件的汇总数据 |
| [成对比较CSV](../../../artifacts/orca_benchmark_suite/S02/report/paired.csv) | 残差相对各对照的差值与统计检验 |
| [学习曲线CSV](../../../artifacts/orca_benchmark_suite/S02/report/learning_curves.csv) | 固定开发检查点的训练过程 |
| [接触质量补充](../../../artifacts/orca_benchmark_suite/S02/report/contact_quality_supplement.md) | 整段穿透验收与主指标的区别 |
| [反馈响应诊断](../../../artifacts/orca_benchmark_suite/S02/report/feedback_diagnostic/results.md) | 外力前后是否产生状态依赖修正 |
| [物理参数和执行器表](../../../artifacts/orca_benchmark_suite/S02/benchmark_card.md) | 手、物体、关节与执行器的实际参数 |
| [最终审计](../../../artifacts/orca_benchmark_suite/S02/audit.json) | 作业、输入、轨迹和视频校验结果 |

## 图表与视频

- [六条件成功率图](../../../artifacts/orca_benchmark_suite/S02/report/figures/clean_success.png)
- [开发集学习曲线](../../../artifacts/orca_benchmark_suite/S02/report/figures/learning_curves.png)
- [标准条件成功/失败分解图](../../../artifacts/orca_benchmark_suite/S02/report/figures/nominal_outcomes.png)
- [60段固定案例视频索引](../../../artifacts/orca_benchmark_suite/S02/videos/index.md)：手心视角，STL与碰撞体两版，失败案例也保留。
- [原始视频片段说明](../../../artifacts/orca_benchmark_suite/S02/source_video/README.md)

## 版本与路径

本目录只集中本轮S02资料，历史实验不混入其结果。旧M01见[历史实验说明](../../orca_cube_acquisition_explained.md)。
原docs下的四份同名文件仅作跳转，正文以本目录为准。
所有运行命令仍在DEXMV-PPO主目录执行；文档中的代码路径以主目录为基准。
权重、数据、视频和冻结的协议JSON不移动、不重复复制，实验结果未改变。
