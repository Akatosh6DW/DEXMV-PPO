# S05 开发迭代记录

日期：2026-10-12。以下 P 编号是几何/参考开发，不是 PPO 训练编号。

## D01：实际接触诊断

运行 `python scripts/audit_s05_contacts.py`，数据位于 `artifacts/orca_benchmark_suite/S05/diagnosis`。

方块 S02/S04 六个展示案例采样到的食指末节接触，侧面比例均为 100%。圆柱纯 PPO/残差同样如此。侧面定义为 `abs(ny)>cos(45度)`。样本为展示集，不代表总体成功率。圆柱 BC+PPO 还涉及近节接触。

下一步类型：环境/参考重定向。增加朝向约束，不改历史 PPO 奖励文件。

## P01：指腹位置与朝向、方块保持原竖直轴

命令：`python scripts/retarget_s05_pads.py --output artifacts/orca_benchmark_suite/S05/P01_pad_ik --starts 12`

模型为 O20，来源为 HIC20 跟踪点，无 checkpoint。只允许七个夹持关节、物体平移与偏航优化。最优两指合计位置误差约 19.25 mm，法线误差约 0.41。四个 8 秒探测均有旧标准的 5.05 秒保持，但新接触形态仍不合格，不能算修复。

失败原因：只让方块绕竖直轴转，难以同时满足源示范接触位置和机械手指腹朝向。

下一步类型：环境/参考几何。允许完整三维刚体对齐。

## P02：允许完整物体旋转

命令：`python scripts/retarget_s05_pads.py --output artifacts/orca_benchmark_suite/S05/P02_full_rotation --starts 16 --full-rotation`

最优位置误差约 0.624 mm，法线误差约 0.219，穿透约 0.641 mm。但四个动力学探测均在 2.25～2.30 秒结束，未完成撤托抓持。

结论：静态可达不等于能从支撑上抓起来。倾斜的方块在合拢前改变了姿态。

下一步类型：环境几何。先检查其他可平放的物体轴，再尝试整体安装朝向校准。

## P03/P04：其他平放轴

命令分别在 `retarget_s05_pads.py` 后增加 `--up-axis x`、`--up-axis negative-y`，各 12 个起点，输出 P03/P04 目录。

两组最优位置误差约 14.39/26.70 mm，四个探测均无撤托稳定保持。没有解决接触位置与朝向兼容问题，不进入训练。

下一步类型：环境几何。保留 P02 的相对姿势，仅对手和物体做整体刚体安装校准。

## P05：整体安装校准

命令：`python scripts/align_s05_mount.py --input artifacts/orca_benchmark_suite/S05/P02_full_rotation/reference.npz --output artifacts/orca_benchmark_suite/S05/P05_mount_alignment`

碰撞形状、尺寸、惯性、摩擦不变。方块平放，手的固定基座随刚体变换重新安装。四个 8 秒探测不掉落，但都低于原保持高度，末节指腹有效保持为 0。最后承力为拇指较低区域与食指末端。

失败原因之一：参考 3 秒才完成闭合，而托盘从 2 秒开始下降。

下一步类型：环境时序。只把参考闭合时长改到原实验的 1.5 秒。

## P06：撤托前完成参考闭合

命令：`python scripts/align_s05_mount.py --input artifacts/orca_benchmark_suite/S05/P02_full_rotation/reference.npz --output artifacts/orca_benchmark_suite/S05/P06_close_before_release --duration 1.5`

四个旧标准保持时间为 2.55/2.90/2.60/2.50 秒；新指腹保持仍为 0。改善了时序，但仍滑向拇指较低接触区。

下一步类型：参考几何/控制校正。在有限关节修正范围内寻找能稳定承力的参考，而不是直接用不合格参考开始 BC。

## P07：有界参考校正

命令：`python scripts/refine_s05_reference.py --input artifacts/orca_benchmark_suite/S05/P06_close_before_release --output artifacts/orca_benchmark_suite/S05/P07_bounded_refinement --iterations 6`

校正腕和七个夹持关节，每维限制 ±0.12 rad，随闭合阶段逐渐加入，不修改初态。开发种子 5011；验证种子 5020～5027。优化的是接触合格保持时间，包含穿透惩罚。此过程是仿真参考优化，不是 PPO。

开发搜索 281 次评估，按 `pad_hold_sec` 的 first/last/max 为 0/0/3.0 秒，最高值首次见评估 3。注意这些是不同候选，不是一个策略连续训练的曲线。按综合目标选出的候选在 8 个 40 秒验证中，指腹有效保持约 2.9～3.0 秒，约 20 秒后掉落。尚不合格。

下一步类型：参考几何/控制校正。只扩大离线参考校正限幅到 ±0.24 rad；PPO 的在线残差仍保持 ±0.08 rad。

## P08：扩大离线参考校正限幅

命令：`python scripts/refine_s05_reference.py --input artifacts/orca_benchmark_suite/S05/P06_close_before_release --output artifacts/orca_benchmark_suite/S05/P08_refinement_bound24 --bound .24 --iterations 10`

开发候选 first/last/max 为 0/0/5.1 秒，最高值首次见评估 82（100 ms 稀疏重建时间包含采样端点，不能据此声称精确连续时长）。8 个长探测实际保持约 7.3～8.4 秒，仍在 30.7～35.35 秒掉落。失败原因：只用 8 秒目标搜索，候选超过短窗口后无法区分谁更稳定。

下一步类型：环境/评估时序。保持限幅和物理参数不变，将开发搜索窗口延长至 40 秒。

## P09：40 秒开发窗口

命令：`python scripts/refine_s05_reference.py --input artifacts/orca_benchmark_suite/S05/P06_close_before_release --output artifacts/orca_benchmark_suite/S05/P09_long_refinement --bound .24 --iterations 12 --probe-seconds 40`

开发候选 first/last/max 指腹保持为 0/1.1/10.6 秒，最大值在评估 513。综合目标同时考虑两指指腹接触比例，因此最终选择的不是单一最长保持候选。8 个 40 秒验证均未掉至终止高度，但合格保持仅约 9.7～10.5 秒；随后主要问题仍是物体缓慢下滑超过 1 cm 保持高度阈值，部分回合晚期出现非合格接触。

下一步类型：PPO 训练超参数。以 P09 为参考，开始小残差训练，检查状态反馈能否抑制慢滑。不能把当前离线参考称为已完成的 PPO 策略。

## V01：新环境完整子步 smoke

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P08_refinement_bound24 --output artifacts/orca_benchmark_suite/S05/V01_substep_smoke --method reference --steps 0 --eval-count 4`

训练环境和审计入口已编译通过，初始观察维度保持 actor 123 / critic 140。4 回合、40 秒、每 5 ms 审计结果：平均新保持 8.0875 秒，30 秒成功率 0，最大穿透 1.0224 mm，禁止接触率 0。完整审计与参考短板一致，没有把侧面重新当作指腹成功。

## T01：小探索残差 PPO 短训

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P09_long_refinement --output artifacts/orca_benchmark_suite/S05/T01_residual_probe --method residual --steps 40960 --evals 3 --std .1 --learning-rate .0001`

从零残差策略开始，不读历史权重。训练种子 5101，开发评估初态 51000～51003。固定新奖励 `15*合格保持 -50*掉落 +2*几何接近势`，控制间隔 50 ms。参考来自 P09；每维在线残差限幅 0.08 rad。

`metrics.csv` 平均新保持 first/last/max 为 10.1250/10.1000/10.1750 秒，best_step=20480；30 秒成功率始终 0。最大穿透 1.397 mm、禁止接触率 0，末次修正 RMS 仅 0.00132 rad。不是成功修复，接触形态基本维持，但防滑未改善。

下一轮类型：PPO 超参数。只将探索标准差从 0.1 改为 0.3，重新从同一个零残差初始状态训练相同 40960 步，检验探索范围不足假设。

## T02：扩大残差探索

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P09_long_refinement --output artifacts/orca_benchmark_suite/S05/T02_residual_std03 --method residual --steps 40960 --evals 3 --std .3 --learning-rate .0001`

同 T01 的场景、参考、训练种子、评估种子和奖励。完成后新保持 first/last/max=10.1250/10.1750/10.2500 秒，best_step=20480；30 秒成功率仍全部为 0。修正 RMS 增至 0.00789 rad，但全程指腹配对比例由约 93.9% 降至 79.9%；禁止接触率仍 0，最大穿透 1.41 mm。扩大探索没有解决主瓶颈，不据此直接延长训练。

下一步类型：环境/参考几何。继续 P12 的上下朝向与握持力验证，并检查让原本冻结的腕关节参与 IK，是否能避免改变安装朝向。下一轮 PPO 使用经过这些验证的参考，保留失败的 T01/T02 记录。

## P10/P11：检查源物体的上下方向

跟踪数据中，人手腕部位于物体局部 Y=-114 mm，指端约 Y=-14～-18 mm；原 RGB 图像中腕和接触点在物体上方。物体 +Y 在源相机中主要指向图像下方。因此增加负 Y 朝上的安装校准候选，而不是继续把源负 Y 接触点放在物体下半部。

P10 命令：`python scripts/align_s05_mount.py --input artifacts/orca_benchmark_suite/S05/P02_full_rotation/reference.npz --output artifacts/orca_benchmark_suite/S05/P10_negative_y_mount --duration 1.5 --up-axis negative-y --seconds 40`

仅改变 P06 的整体安装上下方向。四个未校正回放在约 7.2 秒失败，新指腹保持只有 0.2～0.4 秒。说明对齐上下方向本身不保证动力学抓持。

P11 命令：`python scripts/refine_s05_reference.py --input artifacts/orca_benchmark_suite/S05/P10_negative_y_mount --output artifacts/orca_benchmark_suite/S05/P11_source_up_refinement --bound .24 --iterations 12 --probe-seconds 40`

用相同 0.24 rad 离线校正限幅与 40 秒开发窗口，8 个验证的新指腹保持达到约 13.4～14.3 秒，但约 23.4 秒仍掉落。最大穿透约 1.47 mm；不能算最终成功。该方向更符合源图像的上下关系，后续继续保留其姿态定义。

下一步类型：参考几何/控制校正。扩大离线校正限幅到 0.4 rad，检查静态参考可否形成足够稳定的力闭合。保持真实碰撞和 3 mm 穿透限制，不扩大在线 PPO 残差。

## P12：较强参考预紧力

命令：`python scripts/refine_s05_reference.py --input artifacts/orca_benchmark_suite/S05/P10_negative_y_mount --output artifacts/orca_benchmark_suite/S05/P12_refinement_bound40 --bound .4 --iterations 12 --probe-seconds 40`

八个 40 秒验证的指腹保持约 27.9～30.3 秒，但回合末不再满足高度标准，原严格成功判据仍不通过。最大穿透约 2.24 mm，峰值接触力约 32.6 N；预紧力提高是代价，不能只报时长改善。

下一步类型：PPO 训练。把这个临界可行参考交给与 T01 相同的小探索残差 PPO，检测闭环能否维持末段状态。另行继续原安装场景的腕关节 IK 检查，以减少任务设置改动。

## T03：源上下方向校准后的参考＋残差 PPO

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P12_refinement_bound40 --output artifacts/orca_benchmark_suite/S05/T03_source_up_residual --method residual --steps 40960 --evals 3 --std .1 --learning-rate .0001`

从零残差初始化；同 T01 的 PPO 设置。场景安装和参考已经改变，不能作为仅更换 PPO 算法的公平对照。完成：新保持 first/last/max=27.7625/27.9250/28.2875 秒，best_step=20480。30 秒成功率始终 0；禁止接触率 0，最大穿透 2.312 mm，峰值力约 37.2 N。相比参考只有很小改善，末段下滑未解决。

下一步类型：reward。仅增加权重 5 的高度保持反馈，在两指接触方式合格且撤托时计算 `exp(-max(初始高度-当前高度,0)/5mm)`。不更改保持判据，也不给侧面接触提供此项奖励。

## T04：早期防滑反馈

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P12_refinement_bound40 --output artifacts/orca_benchmark_suite/S05/T04_height_feedback --method residual --steps 40960 --evals 3 --std .1 --learning-rate .0001 --height-weight 5`

重新从相同零残差参数初始化，其他设置同 T03，单独检验稠密高度反馈的作用。

T04 完成：first/last/max=27.7625/28.0375/28.3500 秒，best_step=20480；30 秒成功率仍为 0。没有实质性突破；末次最大穿透增至 2.835 mm，不继续提高奖励权重。下一步类型：环境/接触求解器。检查摩擦软约束引起的慢滑，避免让 PPO 用过大夹紧力补偿数值漂移。

## P13/P14：保留原安装、解冻腕关节

P13 命令：`python scripts/retarget_s05_pads.py --output artifacts/orca_benchmark_suite/S05/P13_free_wrist --starts 16 --up-axis negative-y --free-wrist --duration 1.5`

解冻腕关节后最优位置误差降至约 2.11 mm，法线误差约 0.288，但物体位置优化触及 Y=-40 mm 的人为搜索边界；四个动力学探测仍无合格保持。

下一步类型：环境几何搜索范围。P14 仅将 Y 下界放宽至 -100 mm，保持原 hand mount、物体尺寸和关节物理范围不变，检验是否是人为工作区限制妨碍对齐。

P14 命令：`python scripts/retarget_s05_pads.py --output artifacts/orca_benchmark_suite/S05/P14_free_wrist_workspace --starts 16 --up-axis negative-y --free-wrist --duration 1.5 --y-min=-.1`

P14 得到位置误差约 0.847 mm 的候选，四个短探测能维持旧标准抓持。P15 对原样参考进行 40 秒验证（`refine_s05_reference.py --baseline-only --probe-seconds 40`）：八个均通过旧保持标准，但新指腹保持只有 0～0.4 秒。接触重建发现食指近节承担约 0.43 N，而指尖仅约 0.058 N，因此必须拒绝这个“看起来能抓住”的候选。两个初态穿透略超过 3 mm，也不满足完整 reset gate。

下一步类型：参考几何。IK 增加非目标指节到物体的 2 mm 净空要求，避免把近节隐性支撑作为成功来源。

## P16/P17：非指腹区域净空

P16 命令：`python scripts/retarget_s05_pads.py --output artifacts/orca_benchmark_suite/S05/P16_nonpad_clearance --starts 16 --up-axis negative-y --free-wrist --duration 1.5 --y-min=-.1 --nonpad-clearance .002`

净空仅用于运动学碰撞距离查询，进入动力学前恢复原碰撞 margin/gap。保存的 XML 与原 O20 完全一致。四个 reset 均无穿透，但直接参考动力学保持为 0，不能跳过物理验证。

下一步类型：参考控制校正。P17 在同一原安装场景，对 P16 做 ±0.24 rad 有界动力学校正，仍以新指腹指标作为主要目标。

P17 命令：`python scripts/refine_s05_reference.py --input artifacts/orca_benchmark_suite/S05/P16_nonpad_clearance --output artifacts/orca_benchmark_suite/S05/P17_original_mount_refinement --bound .24 --iterations 12 --probe-seconds 40`

P17 八个验证的新指腹保持 18.4～19.0 秒；最大穿透约 1.77 mm，峰值接触力 23.18 N。原安装不变、初态无穿透，后段仍缓慢下滑并发生非指腹接触。

## P18/P19：区分数值慢滑与策略失误

实际 MuJoCo 配置为 elliptic、Newton、`impratio=1`、`noslip_iterations=0`。P12 的撤托后峰值承力仍约 31.7 N，却持续慢滑，不适合继续靠提高预紧力处理。

MuJoCo 官方说明：软接触的摩擦维度是阻尼约束，可能在摩擦锥内仍产生慢滑；提高 `impratio` 或使用 NoSlip 后处理可减轻。NoSlip 也可能带来额外不稳定性，不能当作无代价开关。[官方求解器与慢滑说明](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)

P18 命令：`python scripts/probe_s05_solver.py --input artifacts/orca_benchmark_suite/S05/P17_original_mount_refinement --output artifacts/orca_benchmark_suite/S05/P18_solver_ablation`

同一参考、同一 4 个初态，NoSlip=0 时平均新保持 18.8 秒；1 次迭代变差到约 2 秒；5/10 次可保持完整 37 秒，但合拢期最大穿透升至 3.42/3.90 mm，因此没有直接采纳。恢复“初始姿态”的控制不保证真正完全张开，部分回合仍由非目标指节托住；不能据此断言求解器粘住物体。

下一步类型：环境求解器。测试官方优先推荐的摩擦/法向阻抗比 `impratio`，保留 NoSlip=0。增加“取消手物接触、保留重力和地面”的独立负对照，确认没有任何隐形固定物体的机制。这个干预只用于负对照，不用于成功案例。

P19 命令：`python scripts/probe_s05_solver.py --input artifacts/orca_benchmark_suite/S05/P17_original_mount_refinement --output artifacts/orca_benchmark_suite/S05/P19_impratio_ablation --parameter impratio`

重要：更改接触求解器改变了仿真设置。任何改善必须先归因于该对照，不能归功于 PPO，也不能直接拿新设置成功率与旧 S02/S04 比算法优劣。

P19 完成：`impratio=1` 新保持均值 18.8 秒，`impratio=2` 为 36.35 秒，最大穿透 1.73 mm；但四个回合在最后时刻均失去高度标准，因此严格成功率仍为 0。5/10 在当前 5 ms 设置出现明显数值不稳定，直接淘汰，不作为成功候选。所有取消手物接触的负对照都在撤托后约 2.3 秒掉落。

下一步类型：PPO 训练。采用唯一稳定且明显缓解漂移的 `impratio=2`，保持原 hand mount 与完整碰撞，不用 NoSlip；从零残差训练，验证能否主动维持最后一段高度。P17→P19 的保持提升属于仿真接触设置改善，后续单独计算 PPO 的增益。

## T05：原安装、校准接触求解器后的残差训练

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P19_impratio_ablation/impratio_2 --output artifacts/orca_benchmark_suite/S05/T05_original_mount_residual --method residual --steps 40960 --evals 3 --std .1 --learning-rate .0001 --height-weight 5`

此候选保留原手的安装、STL、碰撞几何、惯性、摩擦系数与关节范围；改动为允许腕参与的视频重定向、物体初始工作区、非指腹净空、有界参考校正，以及显式记录的 `impratio=2`。

T05 完成：合格保持 first/last/max=36.0250/36.2875/36.2875 秒，best_step=40960；严格成功率 first/last/max=25%/0%/25%，best_step=0。末次指腹对接触比例 100%，最大穿透 1.707 mm，无禁止接触，残差 RMS 0.002142 rad。平均保持略增不等于完整成功；最后高度仍失效，不能选择零步参考冒充重新训练的模型。

## T06：提高学习率，检验修正幅度不足

下一轮类型：PPO 超参数。只将学习率从 1e-4 提高到 3e-4，仍从相同零残差参数初始化，训练 40960 步。T05 价值损失 11.89→7.44、确定性修正很小，因此先做受控探测，不直接重复长训。

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P19_impratio_ablation/impratio_2 --output artifacts/orca_benchmark_suite/S05/T06_learning_rate --method residual --steps 40960 --evals 3 --std .1 --learning-rate .0003 --height-weight 5`

T06 完成：保持 first/last/max=36.0250/36.3875/36.6000 秒，best_step=20480；成功率 25%/0%/25%，成功 best_step=0（20480 也为25%）。指腹对接触比例仍100%，末次最大穿透1.709mm、无禁止接触、残差RMS0.001511rad。提高学习率没有解决末段慢滑，不继续提高。

## P20：稳定区间内的求解器小范围对照

下一轮类型：环境求解器。P19 的2安全但末段高度失败，5数值不稳定；仅测试2.2、2.5、3，中间值不改物理摩擦系数、手部安装或碰撞。仍做取消手物接触的负对照；改善只能归因于数值设置，不归因于PPO。

命令：`python scripts/probe_s05_solver.py --input artifacts/orca_benchmark_suite/S05/P17_original_mount_refinement --output artifacts/orca_benchmark_suite/S05/P20_impratio_narrow --parameter impratio --values 2.2 2.5 3`

Smoke：S05六项单元测试通过；solver脚本通过编译。P20为原生物理探测，不是训练。

P20 完成：2.2/2.5/3 的四个开发初态均通过原生40秒回合；稀疏新保持均37.1秒（采样分辨率100ms，不当作最终子步结果）。各组最大穿透分别1.746/1.769/1.768mm，峰值力22.55/22.45/22.72N，未见前次5/10的不稳定；取消手物接触均2.35秒终止，恢复初始张开控制均最终掉落。选择满足要求的最小改动2.2，再做严格子步审计和真正的PPO训练。

## T07：稳定参考上的指腹残差训练

下一轮类型：环境配置。相对T05只改变impratio=2→2.2；学习率恢复T05的1e-4。新参考仍来自HIC20的P16/P17流程，没有改成手写抓姿。

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P20_impratio_narrow/impratio_2.2 --output artifacts/orca_benchmark_suite/S05/T07_pad_residual --method residual --steps 40960 --evals 3 --std .1 --learning-rate .0001 --height-weight 5`

T07 完成：保持 first/last/max=37.05/37.05/37.05秒，成功率100%/100%/100%，首次并列best_step=0。最后40960步是实际训练模型，不使用零步作为最终策略。末次指腹对比例100%，最大穿透1.705mm，无禁止接触，残差RMS0.002156rad。中期保存轨迹重建的两指指腹接触比例均100%，食指平均法线约[-0.987,0.098,0.129]，旧残差展示约[0.261,-0.939,0.223]。执行器控制与“参考+.08*action，再限幅/限速”的复算最大差1.79e-7rad，证明实际残差执行而不是仅训练时示范。

## T08：延长训练验证保持性

下一轮类型：PPO训练预算。短训练成功后，仅将总交互数延长至163840（开发评估5次），其余同T07；仍从零残差初始化，不复用旧抓法权重，不声称恢复优化器。测试较长训练会否破坏指腹抓法。正式独立测试在选择训练步之后进行，不能用独立测试挑权重。

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P20_impratio_narrow/impratio_2.2 --output artifacts/orca_benchmark_suite/S05/T08_pad_residual_long --method residual --steps 163840 --evals 5 --std .1 --learning-rate .0001 --height-weight 5`

## D01/T09：全新指腹示范和BC+PPO分支

D01命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P20_impratio_narrow/impratio_2.2 --output artifacts/orca_benchmark_suite/S05/D01_pad_demonstrations --method collect --steps 0 --eval-count 24 --eval-start 53000 --height-weight 5`

24个采集初态均通过完整子步30秒标准，最长连续保持37.05秒，无禁止接触，最大穿透1.700mm。数据为参考回放采集，不是PPO策略产物；按回合18/6划分BC训练/验证，未使用52000起的独立测试初态。

下一轮类型：训练初始化。额外验证BC+PPO分支，使用新示范做4000次BC更新，然后40960在线交互；这不是与T08等预算的算法优劣比较。旧侧面示范、旧权重均未加载。

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P20_impratio_narrow/impratio_2.2 --output artifacts/orca_benchmark_suite/S05/T09_pad_guided --method guided --dataset artifacts/orca_benchmark_suite/S05/D01_pad_demonstrations/trajectories.npz --bc-updates 4000 --steps 40960 --evals 3 --std .1 --learning-rate .0001 --height-weight 5`

T09 完成：保持 first/last/max=9.2625/37.05/37.05秒，成功率25%/100%/100%，best_step=40960。末次指腹比例100%，最大穿透1.747mm，无禁止接触。这里零步已经执行4000次BC，不是未经学习；随后40960步才是PPO。开发初态上BC→BC+PPO有真实提高，但不能替代独立测试，也不是训练种子统计。

## T10：BC+PPO同预算长训练

下一轮类型：训练预算。T09短训练成功后，仅延长在线交互到163840，开发评估5次，BC数据/4000更新/随机种子不变。与T08在线步数相同，但BC额外使用示范与离线计算，不能称总成本相同。

命令：`python scripts/run_s05_job.py --input artifacts/orca_benchmark_suite/S05/P20_impratio_narrow/impratio_2.2 --output artifacts/orca_benchmark_suite/S05/T10_pad_guided_long --method guided --dataset artifacts/orca_benchmark_suite/S05/D01_pad_demonstrations/trajectories.npz --bc-updates 4000 --steps 163840 --evals 5 --std .1 --learning-rate .0001 --height-weight 5`

T08长训练完成：成功率first/last/max=100%/100%/100%，保持37.05/37.05/37.05秒；首次并列best_step=0，按预先约定排除零步并取并列最晚训练步，选择163840。所有开发评估均保持指腹对比例100%；末次最大穿透1.711mm、无禁止接触、峰值23.39N、残差RMS0.003014rad。没有证据表明残差提高了这个已很容易成功的参考的成功率；证据支持长训练后没有退回侧面抓法。

下一步类型：独立评估，不再修改该残差模型。等待同预算BC+PPO训练完成后按预先约定统一冻结选择，执行32个配对新初态；不在测试初态上调参。

T10长训练完成：成功率first/last/max=25%/100%/100%，首次best_step=81920；保持9.2625/37.05/37.05秒。按预先约定并列取最晚训练步，选择163840。末次指腹对比例100%、最大穿透2.186mm、无禁止接触、峰值32.02N。相对残差分支夹紧力更大，虽然合格，不能称作更节能。T10与T09的评估周期不同，Brax训练epoch/RNG安排亦不同，因此不能假设相同步数的参数必须一致。

## E01：冻结权重后的独立验证

两分支开发完毕，不再依据独立初态修改参数。残差与BC+PPO均选163840步，BC-only选T10零步（已做4000次BC）。`evaluate_s05_selected.py`在测试开始前保存选择和SHA256；每种方法32个同种子初态52000～52031，保存全部轨迹并断言初始qpos完全一致。

命令：`python scripts/evaluate_s05_selected.py --residual artifacts/orca_benchmark_suite/S05/T08_pad_residual_long --guided artifacts/orca_benchmark_suite/S05/T10_pad_guided_long --output artifacts/orca_benchmark_suite/S05/E01_independent`

E01完成：4种方法各32回合，初始qpos逐元素完全相同。参考32/32、残差32/32、仅BC8/32、BC+PPO32/32。两个PPO分支连续合格保持均37.05秒、指腹对比例均100%、禁止接触均0；最大穿透分别1.714mm/2.208mm，峰值力23.51N/32.18N。仅BC平均合格保持9.2625秒。32/32的Wilson95%区间为89.28%～100%，8/32为13.25%～42.11%；仅反映初态采样，不包含训练种子差异。

结论：本任务的接触方式修复及两个真实训练分支通过独立初态验证；BC+PPO优于同一BC初始化，但残差相对已校正参考没有成功率增益。不能以此声称新S05比旧S02/S04算法更优。

残差全部32轨迹复算控制最大误差1.79e-7rad；两指各11872条稀疏接触重建均属指腹，食指平均法线[-0.9872,0.0998,0.1244]。完整运行时子步判据仍是成功依据。几何不变性检查16项全部通过，唯独显式校准的impratio=2.2与原值1不同。

下一步：按已验证结果导出手心短片、保留失败截点、完成目录与报告，不继续在该测试集调参。

## Smoke 测试

最终验证：`python -m unittest discover -s scripts -p 'test_s05*.py'`共13项通过，涵盖旧侧面抓法拒绝、新训练指腹接受、回调审计不串回合、模型选择规则、置信区间与失败视频截点。正式128条完整轨迹的保持计数/成功布尔值全部重新计算一致；训练与评估的核心环境/接触源码哈希一致，所选权重哈希一致。

视频导出：`python scripts/export_s05_showcase.py --evaluation artifacts/orca_benchmark_suite/S05/E01_independent --output deliverables/S05_指腹抓取修复_2026-10-12`。共6段手心短片，成功两策略各外观/碰撞两版，仅BC失败两版，全部ffmpeg完整解码通过。失败片到15.35秒终止状态，含初帧的文件时长15.40秒；成功片7.05秒。目录和校验记录与视频同文件夹。

`python scripts/test_s05_pad_contacts.py`：4 项通过，包括拒绝末节侧面、近节、背面接触。

宿主机 `aero_rl` 的 JAX 已确认识别 `CudaDevice(id=0)`。不需要修改 NVIDIA 驱动。沙箱不能直接访问 GPU 时，训练使用获准的宿主机执行。
