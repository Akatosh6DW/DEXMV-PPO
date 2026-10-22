# S02/S04 联合统计与图表

[阅读完整报告](report.md)。分析已有的81,700条回合记录，不启动新训练，不改动原评分和原实验结果。

## 内容

- 11组中文图表，各提供PNG与PDF：主成功率、种子稳定性、同轮提升、学习曲线、多条件表现、参考消融、失败分解、接触与保持、阈值敏感性、长短预算、反馈诊断。
- 9份CSV：逐条件汇总、逐种子统计、同轮效应量、跨轮描述差值、开发曲线、首次达标步数、阈值敏感性、条件抗扰统计、保存子集反馈统计。
- 原始输入与生成输出的SHA256校验清单。

S02和S04使用不同训练种子、测试初态和部分训练配方；跨轮差值仅作描述。同轮检验保留原定多重比较校正。长短预算图来自明确标注的S03历史结果，不是S04新做的多条件实验。

## 文件

- [完整分析报告与逐图解释](report.md)
- [Python分析与绘图脚本](../../../scripts/analyze_orca_s02_s04.py)
- [分析单元测试](../../../scripts/test_orca_s02_s04_analysis.py)
- [图表目录](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/figures/)
- [CSV目录](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/tables/)
- [校验清单](../../../artifacts/orca_benchmark_suite/S02_S04_comparison/manifest.json)
- [S02原始文档入口](../S02/README.md)
- [S04原始文档入口](../S04/README.md)

## 复现

```bash
conda activate aero_rl
python scripts/analyze_orca_s02_s04.py
python -m unittest discover -s scripts -p 'test_orca_s02_s04_analysis.py'
```

脚本需NumPy、Matplotlib和系统Noto CJK字体，使用无窗口绘图后端；报告和图表会重新生成。当前环境已具备这些依赖。
