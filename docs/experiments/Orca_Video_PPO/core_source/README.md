# 核心源码中文注释版

[阅读源码讲解](../source_guide.md) / [阅读完整实验报告](../report.md)

这是完整文件的只读教学快照，不是新的训练入口。原文件未修改；快照仅增加中文注释，AST和非注释词法序列均核验一致。

S04冻结列为否，只表示不在S04协议的冻结源码清单内，例如更早的数据准备脚本；不代表已核实为历史运行时的精确版本。

| 顺序与作用 | 注释版 | 原始源码 | 本次注释处数 | 与S04冻结版本一致 |
|---|---|---|---:|---|
| 01 视频跟踪到DexMV中间手型 | [retarget_hic_cube_to_dexmv.py](retarget_hic_cube_to_dexmv.py) | [原文件](../../../../scripts/retarget_hic_cube_to_dexmv.py) | 4 | 不在清单 |
| 02 绑定同源物体及坐标 | [prepare_hic_orca_object.py](prepare_hic_orca_object.py) | [原文件](../../../../scripts/prepare_hic_orca_object.py) | 2 | 不在清单 |
| 03 将物体相对轨迹拟合到Orca | [probe_hic53_orca.py](probe_hic53_orca.py) | [原文件](../../../../scripts/probe_hic53_orca.py) | 4 | 不在清单 |
| 04 E01仿真校正参考 | [probe_orca_cube_acquisition.py](probe_orca_cube_acquisition.py) | [原文件](../../../../scripts/probe_orca_cube_acquisition.py) | 4 | 不在清单 |
| 05 参考回放生成D01示范 | [train_orca_acquisition.py](train_orca_acquisition.py) | [原文件](../../../../scripts/train_orca_acquisition.py) | 3 | 是 |
| 06 BC预训练及共享初始化 | [train_orca_matched.py](train_orca_matched.py) | [原文件](../../../../scripts/train_orca_matched.py) | 5 | 是 |
| 07 残差合成公式 | [orca_residual_controls.py](orca_residual_controls.py) | [原文件](../../../../scripts/orca_residual_controls.py) | 3 | 是 |
| 08 把每步参考与策略接入环境 | [orca_residual_env.py](orca_residual_env.py) | [原文件](../../../../scripts/orca_residual_env.py) | 4 | 是 |
| 09 基础控制与有效抓握判据 | [orca_benchmark_env.py](orca_benchmark_env.py) | [原文件](../../../../scripts/orca_benchmark_env.py) | 3 | 是 |
| 10 当前基础奖励覆盖层 | [orca_refit_env.py](orca_refit_env.py) | [原文件](../../../../scripts/orca_refit_env.py) | 2 | 是 |
| 11 前3秒接近奖励 | [orca_acquisition_env.py](orca_acquisition_env.py) | [原文件](../../../../scripts/orca_acquisition_env.py) | 1 | 是 |
| 12 手物与参考契约 | [orca_hic53_env.py](orca_hic53_env.py) | [原文件](../../../../scripts/orca_hic53_env.py) | 2 | 是 |
| 13 观测与模拟触觉 | [orca_matched_env.py](orca_matched_env.py) | [原文件](../../../../scripts/orca_matched_env.py) | 2 | 是 |
| 14 JAX策略与原生MuJoCo物理 | [orca_native_physics.py](orca_native_physics.py) | [原文件](../../../../scripts/orca_native_physics.py) | 2 | 是 |
| 15 扰动条件与5ms接触审查 | [orca_suite_env.py](orca_suite_env.py) | [原文件](../../../../scripts/orca_suite_env.py) | 4 | 是 |
| 16 S02训练与正式评分 | [run_orca_suite_job.py](run_orca_suite_job.py) | [原文件](../../../../scripts/run_orca_suite_job.py) | 3 | 是 |
| 17 S04所用权重的训练来源 | [run_orca_s03_job.py](run_orca_s03_job.py) | [原文件](../../../../scripts/run_orca_s03_job.py) | 4 | 是 |
| 18 S04正式回放评估入口 | [run_orca_s03_eval.py](run_orca_s03_eval.py) | [原文件](../../../../scripts/run_orca_s03_eval.py) | 2 | 是 |
| 19 新初态评估矩阵与统计 | [run_orca_s04.py](run_orca_s04.py) | [原文件](../../../../scripts/run_orca_s04.py) | 3 | 是 |
| 20 统计方法 | [report_orca_suite.py](report_orca_suite.py) | [原文件](../../../../scripts/report_orca_suite.py) | 3 | 是 |
