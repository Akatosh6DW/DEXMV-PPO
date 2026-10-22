# S04 视频先验友好的有限交互基准

2026-10-09。状态：完成。41项新初态评估、4100回合、两项主比较、图表、十段视频和最终核验均已完成。

本轮只回答：**已有匹配视频示范、在线训练预算较少时，BC后PPO和参考加残差PPO是否优于纯PPO？**
不再继续S03未完成的大规模压力矩阵，不做开瓶盖，不重训已存在的模型。
这是根据已观察结果明确设计的针对性benchmark，不包装为任意场景、任意预算的普遍优势。

## 资料入口

- [先读：结果、两种方法的提升与失效边界](results_explained.md)
- [基准设计、方法与术语](benchmark.md)
- [执行记录与验收清单](execution.md)
- [新初态复测结果](../../../artifacts/orca_benchmark_suite/S04/report/results.md)
- [五个已有条件及长预算边界](../../../artifacts/orca_benchmark_suite/S04/report/historical_conditions.md)
- [新初态对照图](../../../artifacts/orca_benchmark_suite/S04/report/comparison.png)
- [开发集学习曲线](../../../artifacts/orca_benchmark_suite/S04/report/learning_curves.png)
- [手心视角视频](../../../artifacts/orca_benchmark_suite/S04/videos/index.md)
- [原始逐回合数据](../../../artifacts/orca_benchmark_suite/S04/report/episodes.csv)
- [逐训练种子结果](../../../artifacts/orca_benchmark_suite/S04/report/per_seed.csv)
- [最终交付核验](../../../artifacts/orca_benchmark_suite/S04/delivery_audit.json)
- 数据与协议：`artifacts/orca_benchmark_suite/S04/`，与S03并列。

旧S03的全部已完成成绩及不利结果原样保留；S04完成不代表旧S03全部压力矩阵完成。

## 运行

在主工程根目录、`conda activate aero_rl`后执行。
本轮队列已正常结束；查看服务记录：`systemctl --user status orca-s04-focused --no-pager`。
日志：`journalctl --user -u orca-s04-focused -n 20 --no-pager`。

```bash
python -m unittest discover -s scripts -p 'test_orca_s04.py'
python scripts/run_orca_s04.py report
python scripts/run_orca_s04.py render
python scripts/audit_orca_s04_delivery.py
```

后三条命令用于复算报告、检查视频和最终核验；缺少数据会报错，不将缺失填成失败。
不要在现有服务仍运行时启动另一条相同队列。`run`会自动接续报告和视频。
