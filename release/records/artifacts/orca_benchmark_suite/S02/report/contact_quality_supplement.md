# 接触质量范围补充

状态：完成。
S02预定主指标只对撤托后的穿透/禁接触做清洁验收，最大穿透列则覆盖整段活跃回合。
所以某组最大穿透超过3mm、但原始和主清洁成功率相同，并不一定是统计错误。
下面补充整段回合手物穿透也不超过3mm的结果；不更换主指标，不增加显著性检验。
M01使用20Hz保存状态的整段活跃穿透，和S02的采样频率、时间范围都不同，不能直接横比。

| 条件 | 方法 | 回合 | 预定撤托后清洁成功 | 补充整段穿透合格成功 | 因撤托前深穿透被补充指标排除的成功数 |
|---|---|---|---|---|---|
| B0_nominal | bc | 2000 | 51.55% | 51.55% | 0 |
| B0_nominal | guided | 2000 | 68.90% | 63.95% | 99 |
| B0_nominal | half_residual | 2000 | 65.40% | 65.40% | 0 |
| B0_nominal | linear | 2000 | 40.40% | 40.40% | 0 |
| B0_nominal | ppo | 2000 | 29.30% | 29.30% | 0 |
| B0_nominal | residual | 2000 | 69.80% | 69.80% | 0 |
| B0_nominal | unrefined | 2000 | 22.20% | 22.20% | 0 |
| B0_nominal | zero_linear | 200 | 45.00% | 45.00% | 0 |
| B0_nominal | zero_unrefined | 200 | 14.50% | 14.50% | 0 |
| B0_nominal | zero_video | 200 | 68.00% | 68.00% | 0 |
| B1_position | bc | 2000 | 23.20% | 23.15% | 1 |
| B1_position | guided | 2000 | 59.80% | 53.90% | 118 |
| B1_position | linear | 2000 | 38.75% | 38.30% | 9 |
| B1_position | ppo | 2000 | 21.00% | 21.00% | 0 |
| B1_position | residual | 2000 | 37.10% | 28.70% | 168 |
| B1_position | unrefined | 2000 | 9.50% | 8.70% | 16 |
| B1_position | zero_linear | 200 | 35.50% | 35.00% | 1 |
| B1_position | zero_unrefined | 200 | 6.00% | 5.50% | 1 |
| B1_position | zero_video | 200 | 28.50% | 23.00% | 11 |
| B2_yaw | bc | 2000 | 43.10% | 43.10% | 0 |
| B2_yaw | guided | 2000 | 73.45% | 67.80% | 113 |
| B2_yaw | linear | 2000 | 41.70% | 41.65% | 1 |
| B2_yaw | ppo | 2000 | 35.60% | 35.60% | 0 |
| B2_yaw | residual | 2000 | 60.50% | 60.50% | 0 |
| B2_yaw | unrefined | 2000 | 18.85% | 18.85% | 0 |
| B2_yaw | zero_linear | 200 | 42.00% | 42.00% | 0 |
| B2_yaw | zero_unrefined | 200 | 14.50% | 14.50% | 0 |
| B2_yaw | zero_video | 200 | 56.50% | 56.50% | 0 |
| B3_friction | bc | 2000 | 78.80% | 78.80% | 0 |
| B3_friction | guided | 2000 | 97.55% | 97.55% | 0 |
| B3_friction | linear | 2000 | 99.60% | 99.60% | 0 |
| B3_friction | ppo | 2000 | 19.25% | 19.25% | 0 |
| B3_friction | residual | 2000 | 96.70% | 96.70% | 0 |
| B3_friction | unrefined | 2000 | 52.30% | 52.30% | 0 |
| B3_friction | zero_linear | 200 | 99.00% | 99.00% | 0 |
| B3_friction | zero_unrefined | 200 | 49.50% | 49.50% | 0 |
| B3_friction | zero_video | 200 | 96.00% | 96.00% | 0 |
| B4_mass | bc | 2000 | 19.90% | 19.90% | 0 |
| B4_mass | guided | 2000 | 2.65% | 2.65% | 0 |
| B4_mass | linear | 2000 | 15.15% | 15.15% | 0 |
| B4_mass | ppo | 2000 | 22.90% | 22.90% | 0 |
| B4_mass | residual | 2000 | 25.05% | 25.05% | 0 |
| B4_mass | unrefined | 2000 | 7.75% | 7.75% | 0 |
| B4_mass | zero_linear | 200 | 22.00% | 22.00% | 0 |
| B4_mass | zero_unrefined | 200 | 3.50% | 3.50% | 0 |
| B4_mass | zero_video | 200 | 27.50% | 27.50% | 0 |
| B5_push | bc | 2000 | 51.20% | 51.20% | 0 |
| B5_push | guided | 2000 | 68.85% | 63.90% | 99 |
| B5_push | linear | 2000 | 40.55% | 40.55% | 0 |
| B5_push | ppo | 2000 | 29.15% | 29.15% | 0 |
| B5_push | residual | 2000 | 69.50% | 69.50% | 0 |
| B5_push | unrefined | 2000 | 21.85% | 21.85% | 0 |
| B5_push | zero_linear | 200 | 44.00% | 44.00% | 0 |
| B5_push | zero_unrefined | 200 | 14.50% | 14.50% | 0 |
| B5_push | zero_video | 200 | 67.50% | 67.50% | 0 |

整段穿透验收仍不是手部自碰撞或真机安全认证；撤托前托板接触属于任务允许的支撑。
