"""Write a plain-language report from audited S08 data, not assumed rankings."""

import csv
import json

from run_s08_study import OUT, DOC, SEEDS
from run_s07_study import BENCHMARKS

LABELS=dict(ppo="纯PPO",bc="仅BC",guided="BC+PPO",reference="几何参考回放",residual="参考+残差PPO")
CONDITIONS=dict(zip(BENCHMARKS,("正常，位置±2 mm","位置±6 mm","世界Z轴偏航±10度","摩擦乘0.6","质量/惯性乘2","水平推力0.15 N")))


def read(name):
    with (OUT/"report"/name).open() as f:
        return list(csv.DictReader(f))


def main():
    check=json.loads((OUT/"report/verification.json").read_text())
    assert check["status"]=="verified" and check["jobs"]==86 and check["total_test_episodes"]==3424
    aggregate,policies,pairs=read("aggregate.csv"),read("policy_summary.csv"),read("paired_changes.csv")
    training,residuals=read("training_diagnosis.csv"),read("residual_stats.csv")
    failures=read("nominal_failure_modes.csv")
    sensitivity=read("height_threshold_sensitivity.csv")
    selection=json.loads((OUT/"selection.json").read_text())["selected"]
    demos=json.loads((OUT/"demonstrations/dataset_manifest.json").read_text())
    def agg(bench,method):
        return next(x for x in aggregate if x["benchmark"]==bench and x["method"]==method)
    lines=["# S08：摆位修复后的纵向包覆完整方法比较","",
        "## 先说这次做的是什么","",
        "原先圆柱与张开的拇指重叠；P03只把圆柱初始中心抬高12 mm，就出现有效包覆。本轮在这个修好的场景上重新采集示范、重新训练三方法，而不是给旧模型换个视频标题。",
        "","已完成9次正式训练、86项测试/消融、3424个独立评估回合。这里的‘独立’指没有参加选模的初态；不同策略共用相同初态进行配对，不能把它们当互相独立的训练重复。",
        "","**参考来源必须说清楚：本轮S08使用几何求解与仿真摆位筛选得到的参考，不是重新从人类RGB视频提取。** S05来自HIC处理后的跟踪点；S06/S07来自DexMV处理后的Adroit罐体示范。S08检验的是参考引导与在线学习的结合方式，不能拿它证明新的视频提取链路已经完成。",
        "","## 一次尝试如何发生","",
        "1. 圆柱由托板放在手前，手张开。圆柱直径53 mm、高53 mm、质量35 g。",
        "2. 每50 ms给17个关节一次目标角度；期间MuJoCo以5 ms步长计算10次接触和运动。",
        "3. 第2秒托板开始撤下，第3秒以后要求手独立维持包覆。",
        "4. 最多观察40秒；掉落55 mm以上提前终止。连续合格30秒且末尾仍合格，才算成功。",
        "","这里不是机械臂在桌面上寻找物体再抬起；‘抬高12 mm’是初始摆位修复，不是新学到的抬升动作。固定基座但腕关节可控。",
        "","源码中的`cube`、`cube_z`是沿用旧接口的物体名称；本轮编译模型的实际几何类型已检查为cylinder（圆柱），不是文件名叫cube就仍在抓方块。",
        "","## 三方法究竟哪里不同","",
        "| 方法 | 训练时 | 每一步执行时 |","|---|---|---|",
        "| 纯PPO | 从在线尝试与奖励学习 | 网络决定全部关节目标 |",
        "| 仅BC | 用示范状态/动作做监督学习 | BC网络决定全部关节目标 |",
        "| BC+PPO | 先BC，再用PPO继续改同一个策略 | 微调后的网络决定全部关节目标，不再强制加回参考 |",
        "| 参考回放 | 没有网络训练 | 按时间表执行目标角度，有底层关节跟踪，无高层状态纠偏 |",
        "| 残差PPO | PPO只学参考附近的小修正 | 参考角度＋修正，再限幅限速 |",
        "","残差公式：`q_target(t)=clip_joint(b(t)+0.08*clip(pi_theta(obs_t),-1,1))`，再限制相对上一目标每步最多变化0.1 rad。0.08 rad约4.58度，是单关节请求上限，不是每步都会改那么多。修正不在时间上不断累加。",
        "","BC+PPO公式：先让`pi_theta(s_demo)`接近`a_demo`，然后PPO更新theta；执行时直接`a_t=pi_theta(s_t)`。所以BC是教会一个初始习惯，残差是执行时每一步都保留参考。",
        "","这里BC学习的是起手、闭合到保持的整段过程，不是仅仅摆一个初始姿势。它和残差的区别在于执行时有没有独立的参考项强制相加，而不是有没有学习过后续动作。",
        "","网络看到关节、物体相对状态、触觉、时间等；残差额外看到参考目标。不是执行时重新识别摄像头视频。模拟触觉不是实物测得的传感器。",
        "","## 数据、奖励与训练量","",
        f"示范采集32次，筛出{len(demos['seeds'])}条严格成功轨迹，前{len(demos['seeds'])*3//4}条用于BC训练、其余用于BC验证；每条最多800个状态动作记录。不是{len(demos['seeds'])}段独立视频。失败采集也保留。",
        "","BC每次抽128个起手阶段样本和128个后续保持样本，避免长时间保持淹没短起手。Adam学习率1e-3，训练20000次；损失比较网络的动作均值logit与示范归一化动作经arctanh变换后的标签，不是直接以毫米或弧度衡量。验证误差小仍不保证闭环抓持成功。",
        "","策略和价值网络都是两层128单元的MLP，actor输入123维，critic输入140维，动作17维。完整动作策略初始均值对应张开的初始关节目标，残差初始均值为零修正。",
        "","每种在线方法3个训练种子8501/8502/8503，每次163840步，合计1474560在线交互步；BC另外每种子20000次监督梯度更新。相同在线预算不等于相同总计算成本。",
        "","运行环境是`conda activate aero_rl`。神经网络与PPO更新走JAX/GPU，物理推进使用CPU原生MuJoCo回调并保留双精度积分状态；因此GPU利用率不高不代表训练没运行，也不是把物理仿真换成另一套未经比较的引擎。",
        "","163840步是32个并行环境累计的交互：某个环境收到动作、推进50 ms、返回状态和奖励，计1步。不是163840回合。每个采样更新大轮收集32×20×4=2560步，共64个大轮；每批4个小批重复更新4次，共1024次小批更新。掉落会提前重置，不能用步数除以800冒充精确回合数。",
        "","奖励固定为`r=0.05*(15*有效包覆 -50*终止 +2*接触覆盖 +5*高度保持)`，三个方法相同。没有每隔一段时间让大模型读日志后改奖励。",
        "","接触覆盖表示掌面、拇指及其余指的曲面载荷覆盖程度，不是模仿角度误差；它需要接触才能提供反馈，不等价于S07的接近/朝向塑形。这个设计可能让从零探索较难，不能把弱纯PPO结果解释为PPO算法本身不能完成任务。高度保持是以5 mm为尺度的指数反馈，鼓励少下滑。",
        "","有效包覆要求掌面、拇指及至少两根其它手指的合格曲面接触均承力>0.02 N；撤托后禁止非手支撑，穿透<=3 mm，高度下降<15 mm，初态穿透<=1 mm。接触力/穿透每5 ms检查，控制步取最差子步；高度和持续时间每50 ms检查。计数口径可显示37.05秒，物理时间约37秒。",
        "","## 正常条件的完整结果","",
        "每个策略实例使用相同64个新初态。多种子方法的均值/标准差按3个独立训练实例计算，参考只有1条。",
        "","前期P03的26/32与本轮参考回放来自不同测试初态及随机数实现；P03为NumPy开发回放，S08为JAX初态和float32策略接口、双精度物理积分。不能把两次参考成功率之差当成PPO收益。本轮方法结论只来自同一S08接口下的配对比较。",
        "","| 方法 | 各训练种子通过数 | 平均成功率 | 种子间标准差 | 平均最长保持 |","|---|---|---:|---:|---:|"]
    for method,label in LABELS.items():
        a=agg("B0_nominal",method)
        group=[x for x in policies if x["benchmark"]=="B0_nominal" and x["method"]==method]
        counts="；".join(f"{x['train_seed']}: {x['successes']}/{x['count']}" if method!="reference" else f"{x['successes']}/{x['count']}" for x in group)
        sd="不适用" if method=="reference" else f"{100*float(a['success_sd']):.2f}个百分点"
        lines.append(f"| {label} | {counts} | {100*float(a['success_mean']):.2f}% | {sd} | {float(a['mean_hold']):.2f}s |")
    delta_res=100*(float(agg("B0_nominal","residual")["success_mean"])-float(agg("B0_nominal","reference")["success_mean"]))
    delta_bc=100*(float(agg("B0_nominal","guided")["success_mean"])-float(agg("B0_nominal","bc")["success_mean"]))
    lines += ["",f"残差相对参考均值差：**{delta_res:+.2f}个百分点**；BC+PPO相对仅BC：**{delta_bc:+.2f}个百分点**。这些是本次固定设置下的结果；正差说明本批有收益，负差说明有退化，不能改写成‘一定更强’。",
        "","![正常与扰动](../../../artifacts/orca_benchmark_suite/S08/report/success_and_robustness.png)",
        "","## 训练途中是否退化，选了哪个进度","",
        "开发初态85000～85007与示范86000系列、最终测试87000系列分开。每20480步检查，保存0至163840步共9个点。只从正步数检查点按成功率、保持时间、较晚步数排序选择，BC的0步单列。所有权重在独立测试前冻结。",
        "","| 方法 | 种子 | 开始成功率 | 最后成功率 | 历史最高 | 选中步数 |","|---|---:|---:|---:|---:|---:|"]
    for r in training:
        lines.append(f"| {LABELS[r['method']]} | {r['train_seed']} | {100*float(r['first']):.1f}% | {100*float(r['last']):.1f}% | {100*float(r['max']):.1f}% | {float(r['selected_step']):.0f} |")
    lines += ["", "### 更新稳定性不能省略", "",
        "KL衡量更新前后动作概率分布的变化，不是关节转了多少度。下面给出每次训练记录的最大值与最后值；极大的KL意味着部分更新偏离旧策略很远。数值有限、训练跑完，不等于优化稳定。",
        "", "| 方法 | 种子 | 最大KL | 最后KL |", "|---|---:|---:|---:|"]
    for r in training:
        lines.append(f"| {LABELS[r['method']]} | {r['train_seed']} | {float(r['max_kl']):.6g} | {float(r['last_kl']):.6g} |")
    lines += ["", "本轮部分训练出现很大的KL，属于必须披露的优化稳定性问题。PPO的裁剪目标并不保证每次更新后的KL都很小；不能据此把纯PPO的失败全部归因于没有先验，也不能把BC+PPO的波动全部归因于模仿机制。这里只记录现象，不把尚未验证的原因写成定论。",
        "", "为保持本轮预定比较条件，没有中途改变学习率或奖励。后续可另开受控实验，比较更小学习率、较少更新次数或KL提前停止，同时让各方法获得相同调参机会。"]
    lines += ["","![开发学习曲线](../../../artifacts/orca_benchmark_suite/S08/report/learning_curves.png)",
        "","奖励曲线高不等于最终成功率高；只有8个开发初态，曲线一次跳12.5个百分点，不能把这种离散变化解读为精密趋势。训练KL等原始指标保存在`ppo_metrics.jsonl`和`training_diagnosis.csv`。",
        "","同样初始标准差0.1不代表同样的物理探索幅度：完整动作通过关节范围映射到角度，残差动作通过0.08 rad的小范围映射。因而残差的随机试探通常更小、更容易保留抓法，但也限制恢复能力。这是当前架构的实际机制，同时也是因果归因的混杂因素；还没有做物理动作噪声完全匹配的正式重训。",
        "","## 换条件后是否还能抓","",
        "正常每策略64初态，其它各32，全部是冻结权重评估，不额外训练。",
        "","| 条件 | 纯PPO | 仅BC | BC+PPO | 参考 | 残差PPO |","|---|---:|---:|---:|---:|---:|"]
    for bench in BENCHMARKS:
        values=[f"{100*float(agg(bench,m)['success_mean']):.2f}%" for m in LABELS]
        lines.append("| "+CONDITIONS[bench]+" | "+" | ".join(values)+" |")
    lines += ["","撤托后曾超过3 mm穿透上限的回合数如下。这些回合即使没有掉落，也不能计成功；数字与其它失败原因可能重叠。",
        "","| 条件 | 纯PPO | 仅BC | BC+PPO | 参考 | 残差PPO |","|---|---:|---:|---:|---:|---:|"]
    for bench in BENCHMARKS:
        values=[f"{agg(bench,m)['excess_penetration_count']}/{agg(bench,m)['episodes']}" for m in LABELS]
        lines.append("| "+CONDITIONS[bench]+" | "+" | ".join(values)+" |")
    lines += ["","各条件参考回放的初始穿透超1 mm数量（其它方法共享相同初态）："]
    for bench in BENCHMARKS:
        a=agg(bench,"reference")
        lines.append(f"- {CONDITIONS[bench]}：{a['invalid_initial_count']}/{a['episodes']}。")
    lines += ["","若扩大摆位范围产生不合法初态，按预定规则记失败但单列数量；不能把这部分全归为策略抗扰动不足。没有删掉这些初态再偷偷改变成功率分母。",
        "","计入body与geom两层旋转后，实际圆柱轴沿世界Z；B2主要是轴对称物体自转与四元数观测变化的检查，不是大角度倾斜泛化。详见[实际几何轴核对](姿态坐标核对.md)。摩擦、质量测试另行改变接触动力学，正常成功率不等于泛化保证。",
        "","## 残差救回与破坏的回合","",
        "同一个初态，参考失败、残差成功叫救回；参考成功、残差失败叫破坏。参考是三组比较共用的，不是3条独立示范。",
        "","| 训练种子 | 救回 | 破坏 | 同成同败 | 净变化 |","|---|---:|---:|---:|---:|"]
    for r in pairs:
        if r["benchmark"]=="B0_nominal" and r["right"]=="reference":
            lines.append(f"| {r['train_seed']} | {r['wins']} | {r['losses']} | {r['ties']} | {float(r['net_percentage_points']):+.2f}个百分点 |")
    lines += ["","![同初态救回与破坏](../../../artifacts/orca_benchmark_suite/S08/report/paired_outcomes.png)",
        "","## 失败具体发生在哪里","",
        "以下按所有正常测试回合计数，不只抽成功视频。末尾缺少承力和高度损失可能同时发生，列之间有重叠，不能相加当失败总数。‘曾超穿透’统计撤托后的全程，不能被后面的稳定片段抵消。",
        "","| 方法 | 回合数 | 掉落终止 | 末尾高度损失超限 | 缺掌面载荷 | 缺拇指载荷 | 其它合格手指不足 | 曾超穿透 | 连续保持不足30秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method,label in LABELS.items():
        group=[r for r in failures if r["method"]==method]
        counts=[sum(key in r["reasons"].split("|") for r in group) for key in
            ("drop","height_loss","no_palm_load","no_thumb_load","too_few_other_fingers",
             "excess_penetration","insufficient_continuous_hold")]
        lines.append("| "+label+" | "+str(len(group))+" | "+" | ".join(map(str,counts))+" |")
    lines += ["","![完整高度与承力实例](../../../artifacts/orca_benchmark_suite/S08/report/wrap_contact_examples.png)",
        "","图中固定取8501残差策略的首个成功和首个失败，是实例而不是总体均值。灰色竖线是3秒撤托完成；高度下降达到15 mm就不再合格，即使还没落地。底部三条载荷曲线分别是掌、拇指、其它四指中第二大的合格载荷，第三条超过0.02 N表示至少两根其它手指过门槛。载荷取每个50 ms区间的最小值，不是峰值。",
        "","因此本任务评价的是持续、限下滑的掌面包覆，不只是‘40秒内没有落地’。如果另研究允许大幅下滑的抓持，应单独定义新指标，不能在看到结果后替换本轮成功标准。"]
    lines += ["","### 高度门槛敏感性（补充诊断，不替换主指标）","",
        "观察到部分回合靠近15 mm门槛后，额外用相同保存轨迹重算10/15/20 mm，其他接触、穿透、30秒连续保持条件不变。这是事后透明诊断，不是三套新训练或三项预注册结果，也不据此重选权重。",
        "","| 方法 | 10 mm | 15 mm，正式标准 | 20 mm | 40秒未触发55 mm掉落终止 |","|---|---:|---:|---:|---:|"]
    for method,label in LABELS.items():
        values=[]
        for threshold in (10,15,20):
            group=[r for r in sensitivity if r["method"]==method and int(r["height_threshold_mm"])==threshold]
            values.append(f"{100*sum(int(r['successes']) for r in group)/sum(int(r['count']) for r in group):.2f}%")
        values.append(f"{100*sum(int(r['no_drop_termination_count']) for r in group)/sum(int(r['count']) for r in group):.2f}%")
        lines.append("| "+label+" | "+" | ".join(values)+" |")
    lines += ["","最后一列仅说明未触发下落终止，不能替代合格包覆：它没有要求合格接触和小幅下滑。若方法间差异随高度门槛变化明显，应解释为对指标敏感，不能夸大成机械手真实抓取能力的稳健排序。原始逐策略数据见`height_threshold_sensitivity.csv`。"]
    lines += ["","## 消融：关掉修正或改参考","",
        "| 条件 | 通过数 | 成功率 |","|---|---:|---:|"]
    for r in policies:
        if r["benchmark"].startswith("A_"):
            lines.append(f"| {r['run']} | {r['successes']}/{r['count']} | {100*float(r['success_rate']):.2f}% |")
    lines += ["","gain0与gain0.5是在执行时关掉或减半修正，不是重新训练。`linear`是快速限速闭合，P03本来就是1.5秒线性闭合；`uncorrected`是P01拇指校正前的几何动作，但初始摆位仍一致。这不是视频有无的因果实验。",
        "",f"零增益与参考的保存状态/控制/保持/活跃标记是否逐数组完全一致：**{check['zero_gain_exact']}**。详细最大差在`zero_gain_replay_checks.json`，不只比较两个成功率是否相等。",
        "","![执行消融](../../../artifacts/orca_benchmark_suite/S08/report/ablation.png)",
        "","## 动作偏离和力的诊断","",
        "偏离参考不必然是坏事，保持得像也不必然成功。下图统计相对几何参考的目标角偏差，不是人类姿态重建误差。只看合格保持帧会产生幸存者偏差，因此同时保留全部活跃帧版本，先按回合统计再汇总。",
        "","![参考跟随误差](../../../artifacts/orca_benchmark_suite/S08/report/reference_following.png)",
        "","| 残差训练种子 | 请求RMS(rad) | 最大请求(rad) | 接近上限比例 |","|---|---:|---:|---:|"]
    for r in residuals:
        if r["benchmark"]=="B0_nominal":
            lines.append(f"| {r['run']} | {float(r['request_rms_rad']):.6f} | {float(r['request_max_rad']):.6f} | {100*float(r['saturation_fraction']):.3f}% |")
    lines += ["","残差表的RMS按所有已保存活跃帧计算，长回合权重更大；它不是逐回合等权RMS。上方参考偏离图则先按回合计算，二者统计口径不同。没有合格保持帧的方法在条件图中标为N/A，不能当作误差为0。"]
    lines += ["","| 方法（正常） | 所有回合最大撤托穿透 | 全程峰值载荷 |","|---|---:|---:|"]
    for m in LABELS:
        a=agg("B0_nominal",m)
        lines.append(f"| {LABELS[m]} | {float(a['max_depth_mm']):.3f} mm | {float(a['max_force_n']):.2f} N |")
    lines += ["","极值包括失败，不是典型成功的平均力。训练接口记录手指分组载荷峰值与单点峰值的较大值；不能与P03仅单点峰值直接比较。MuJoCo数值有限不等于接触物理合理，更不等于硬件安全。",
        "","## 与S05、S06、S07放在一起怎么讲","",
        "- S05：修复方块末节指腹判定，单训练种子。回答‘原先侧面接触的问题能否修复’，不是3种子完整benchmark。",
        "- S06：圆柱三指参考与物理设置开发、少量独立测试，为S07准备条件。不能混入S07当独立训练重复。",
        "- S07：固定圆柱三指任务的完整多种子对比，暴露BC+PPO退化、残差收益和鲁棒性缺口。",
        "- S08：改为掌面参与的纵向包覆；先修初态穿透，再按同样预算完整比较。任务标准不同，不能用S08百分比减S07百分比就说算法提升。",
        "","主线是‘正确几何/接触定义 → 可行参考 → 学习方式比较 → 独立鲁棒性验证’，不是不断换容易场景直到胜过纯PPO。S08没有原视频贡献归因，纯PPO探索反馈和预算也尚未调到极限。",
        "","## 文件、视频和审计范围","",
        f"全部3424回合有原始指标和初态；保存的{check['recomputed_saved_episodes']}条完整轨迹已独立重算成功标准。正常与消融保存全部，扰动每任务仅前4条，不能声称离线复算了所有3424条。控制重组最大误差{check['control_max_error_rad']:.3g} rad。",
        "","- `artifacts/orca_benchmark_suite/S08/report/episodes.csv`：每个方法、种子、扰动、初态的结果。",
        "- `aggregate.csv`、`policy_summary.csv`：方法均值、种子离散及逐权重Wilson区间。",
        "- `paired_changes.csv`：同初态救回/破坏；`training_diagnosis.csv`：训练开始/最后/最高/选模及KL。",
        "- `reference_following.csv`、`residual_stats.csv`：参考偏离与修正量；`verification.json`：完成度与审计。",
        "- `nominal_failure_modes.csv`：正常条件逐回合的掉落、超高度、缺少掌面/拇指/其它指承力等失败标记；一条失败可有多个标记，不能相加当失败总数。",
        "- [主要策略手心视频目录](../../../deliverables/S08_纵向包覆完整对比_2026-10-12/目录.md)。",
        "- [实验约定](包覆方法对比约定.md)、[完整执行清单](任务执行清单.md)、[迭代日志](iteration_log.md)。",
        "- [资料包数据阅读说明](资料包数据阅读说明.md)、[核心源码导读](源码导读.md)。",
        "","## 术语速查","",
        "| 术语 | 人话解释 |","|---|---|",
        "| BC | 看标准状态/动作答案，练习模仿 |","| PPO | 反复试动作，根据奖励改策略网络 |",
        "| actor / critic | 策略网络负责动作，价值网络估计未来累计奖励、辅助训练 |",
        "| MLP | 多层全连接神经网络，这里两层隐藏层各128个单元 |",
        "| logit / arctanh | 动作经tanh限到[-1,1]之前的内部数值，以及把标签反变换回该数值范围的函数 |",
        "| Adam | 根据梯度调整网络参数的优化算法，BC阶段使用它 |",
        "| 域随机化 | 在训练中随机改变质量、摩擦等条件，练习适应变化 |",
        "| 残差 | 在每一步基础动作上相加的小修正 |","| rollout | 一段连续采集的状态、动作、奖励 |",
        "| seed | 控制随机过程的编号，训练seed和测试初态seed不是同一种重复 |",
        "| checkpoint | 某个训练进度保存的网络参数，本汇报包不携带权重 |",
        "| KL | 更新前后动作概率分布变化的度量，不是动作角度 |",
        "| RMS | 一组数的均方根，表示典型误差/修正幅度 |",
        "| 独立测试 | 没参与训练调参或选权重的新初态验证 |",
        "| 消融 | 控制其它因素，只关闭/替换一个组成部分看影响 |",
        "","## 结论边界与下一步","",
        f"本轮实测残差对固定参考的净均值差为{delta_res:+.2f}个百分点，BC继续PPO对仅BC为{delta_bc:+.2f}个百分点。应结合上面的逐种子、配对得失和扰动表解读，而不是只取最好的一段视频。",
        "","用人话归纳：参考本身已经很强，所以正常条件下加一个小修正不保证更好。BC+PPO在位置范围扩大时表现最好，但换成低摩擦后反而退步，说明它学到的改动不一定适应另一种接触条件。残差在质量加倍时略高于参考，在正常、位置和低摩擦条件下却没有超过参考；不能只摘质量那一列证明全面优势。",
        "","还有一个重要限制：把允许下滑从15 mm收紧到10 mm，BC+PPO成功率从96.35%降到31.25%，而参考仍为96.88%。这说明‘最终拿住’与‘基本不下滑’不是同一种能力。该敏感性是事后诊断，正式成绩仍按预定15 mm报告；没有用修改门槛来制造某个方法获胜。",
        "","当前只有3个训练种子，逐权重Wilson区间描述的是该权重面对测试初态的抽样不确定性，不能当作算法总体差异的置信区间。本文没有宣称小幅均值差达到统计显著，也没有证明充分调优后的纯PPO仍然失败。",
        "","下一轮若研究改进，可以独立比较更丰富的几何接近奖励、BC保留约束、探索物理幅度匹配、摩擦/质量域随机化。但要新建开发与测试划分，不把这次独立结果拿来调完再报成原实验。若要证明视频先验的贡献，还必须做匹配包覆抓法的视频来源、与同预算几何参考对照，并如实计入离线校正成本。",
        "","本轮完成的是仿真受控实验，没有实机成功证明，没有任意物体/位置保证。"]
    (DOC/"完整对比结果.md").write_text("\n".join(lines)+"\n")
    print(DOC/"完整对比结果.md")


if __name__=="__main__":
    main()
