# S03 强PPO、开发成本与闭环残差验证

2026-10-09。状态：根据用户新目标停止扩展评测，保留部分结果，未完成全矩阵。S02保持不变。

新重点见并列的[S04 视频先验友好的有限交互基准](../S04/README.md)。
停止时40项确认训练完整，正式评估477/738项完整；不将S03包装为全部完成。

本轮落实四项工作：受控调试得到的PPO对照、等开发预算、闭环与固定修正回放对照、
未参与训练的新物体迁移。所有结果和失败均保留，不以残差获胜作为完成条件。

- [实验协议与交付清单](protocol.md)
- [方法、对接方式与术语说明](methods_explained.md)
- [准备成本与计算预算口径](preparation_cost.md)
- [执行与决策日志](execution.md)
- [正式测试中途统计（未完成）](../../../artifacts/orca_benchmark_suite/S03/report/results.md)
- [运行、复算与查看方法](reproduction.md)
- [开发阶段逐轮决策](../../../artifacts/orca_benchmark_suite/S03/development_report.md)
- [已完成训练的中途审计](../../../artifacts/orca_benchmark_suite/S03/training_audit.json)
- [新物体几何检查](../../../artifacts/orca_benchmark_suite/S03/object_geometry/gates.json)
- [长方体手心预览](../../../artifacts/orca_benchmark_suite/S03/object_geometry/cuboid/stl.png)
- [球体手心预览](../../../artifacts/orca_benchmark_suite/S03/object_geometry/sphere/stl.png)
- 数据根目录：`artifacts/orca_benchmark_suite/S03/`，与S02并列。
- 历史基准：[S02总目录](../S02/README.md)。

## 计划边界

主任务仍为Orca固定腕座抓握53mm方块，不做开瓶盖，不修改历史结果。
新物体选用长方体和球体，检验从方块训练策略向新形状的迁移；不声称视频里出现了这些物体，
也不声称这是新增视频级泛化。现有HIC21在旧夹具下的重定向误差很大，不能直接当成功示范。
本轮选择上一轮建议中的“新物体”路线，而不是把同一视频切片当成独立视频。

完整执行包含开发、冻结、确认训练、独立测试、消融、视频、统计与验收；任何阶段完成
都不等于整个S03完成。开发时只看开发集，不读取最终测试成绩来选参数。
