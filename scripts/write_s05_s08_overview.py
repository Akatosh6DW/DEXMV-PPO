"""Build a cross-study reporting guide without treating different tasks as one leaderboard."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_s07_study import write_csv, COLORS
from run_s08_study import ROOT, OUT, DOC

METHODS=("ppo","bc","guided","reference","residual")
LABELS=("纯PPO","仅BC","BC+PPO","参考回放","残差PPO")


def main():
    rows=[]
    for study in ("S05","S06"):
        summary=json.loads((OUT.parent/study/"E01_independent/summary.json").read_text())
        for method in METHODS:
            key="bc_only" if method=="bc" else method
            r=summary["methods"].get(key)
            rows.append(dict(study=study,method=method,evaluated=r is not None,
                success_mean=(r.get("rate",r.get("success_rate")) if r else None),
                success_sd=None,policies=(1 if r else 0),
                successes=(r["successes"] if r else None),episodes=(r.get("total",r.get("count")) if r else 0)))
    for study in ("S07","S08"):
        with (OUT.parent/study/"report/aggregate.csv").open() as f:
            for r in csv.DictReader(f):
                if r["benchmark"]=="B0_nominal":
                    rows.append(dict(study=study,method=r["method"],evaluated=True,
                        success_mean=float(r["success_mean"]),success_sd=float(r["success_sd"]),
                        policies=int(r["policies"]),successes=int(r["successes"]),episodes=int(r["episodes"])))
    write_csv(OUT/"report/historical_overview.csv",rows)
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout="constrained")
    titles=("S05 cube pads (1 seed)","S06 cylinder pads (1 seed)","S07 cylinder pads (3 seeds)","S08 palm wrap (3 seeds)")
    for study,ax,title in zip(("S05","S06","S07","S08"),axes,titles):
        for j,method in enumerate(METHODS):
            r=next(r for r in rows if r["study"]==study and r["method"]==method)
            if r["evaluated"]:
                value=100*r["success_mean"]
                ax.bar(j,value,color=COLORS[method])
                ax.text(j,value+2,f"{value:.1f}",ha="center",fontsize=8)
            else:
                ax.text(j,4,"N/A",ha="center",fontsize=8)
        ax.set(title=title,ylim=(0,110),xticks=range(5),xticklabels=["PPO","BC","BC+PPO","Ref","Residual"])
        ax.tick_params(axis="x",labelrotation=30)
    axes[0].set_ylabel("Task-specific success (%)")
    fig.suptitle("Different task criteria: compare methods within each panel, not scores across panels",fontsize=11)
    fig.savefig(OUT/"report/historical_overview.png",dpi=160);plt.close(fig)
    lines=["# S05至S08：怎么把整个研究讲清楚","",
        "## 一句话主线","",
        "先让仿真中的手、物体和合格接触真正对上，再问：参考只用来教初始策略，与执行时每一步都保留参考，哪一种更容易学、哪一种更稳？",
        "","## 四个实验各自回答的问题","",
        "| 实验 | 问题 | 参考来源 | 已有证据强度 |","|---|---|---|---|",
        "| S05方块 | 原来食指侧面也能算成功，修正为末节指腹后怎样？ | HIC处理后跟踪点、重定向与仿真校正 | 单训练种子，32个新初态/方法 |",
        "| S06圆柱 | 三指围住圆柱、接触朝向更接近源示范是否可行？ | DexMV处理后Adroit罐体序列96，182～271帧 | 几何/求解器开发与单种子16初态/方法 |",
        "| S07圆柱复验 | 固定S06选定场景，几种方法是否稳健地不同？ | 同一源示范和校正参考 | 三训练种子、六条件、消融；3424回合 |",
        "| S08纵向包覆 | 修复初始穿透后，掌面参与的抓法上几种方法表现怎样？ | 手工几何包覆参考，不是新RGB视频提取 | 重新示范采集、三训练种子、六条件和消融；3424回合 |",
        "","## 正常条件数据总览","",
        "下面是资料检索表，不是四个任务共用的排行榜。空缺表示没做，不是0%。多种子列是每训练种子成功率的均值。",
        "","| 实验 | 纯PPO | 仅BC | BC+PPO | 参考回放 | 残差PPO |","|---|---:|---:|---:|---:|---:|"]
    for study in ("S05","S06","S07","S08"):
        values=[]
        for method in METHODS:
            r=next(x for x in rows if x["study"]==study and x["method"]==method)
            values.append(f"{100*r['success_mean']:.2f}%" if r["evaluated"] else "本轮未测")
        lines.append("| "+study+" | "+" | ".join(values)+" |")
    lines += ["","![分任务数据，不跨任务排名](../../../artifacts/orca_benchmark_suite/S08/report/historical_overview.png)",
        "","## 报告时把三类贡献拆开","",
        "1. 几何与任务修复：碰撞代理、指腹法向规则、初始穿透排除、工作区与求解器校准。这决定实验是否在做预期任务，不能算成PPO收益。",
        "2. 先验构建：源示范选择、跨手型映射、物体相对布局和离线校正。参考好坏本身能主导成功率，必须有参考回放对照。",
        "3. 在线学习：在同一场景、奖励、在线步数、新初态下比较纯PPO、BC+PPO、残差；用仅BC及零残差消融定位收益和退化。",
        "","S05的求解器impratio曾改变，S06也有物理设置开发；这些影响要单列。S07冻结后才比较。S08初始抬高12 mm和允许掌面接触改变了任务，不能把新任务较高成功率说成旧算法提高了同样多百分点。",
        "","## 视频在链路中的位置","",
        "原目标是‘视频决定大体动作，PPO决定微调’。真正对应的是执行端显式相加的残差结构，不是BC预训练本身。已有S05/S06/S07利用处理后的人类/Adroit示范与离线校正，执行读取NPZ参考，不是持续看RGB。S08包覆方法比较只有几何参考，补齐视频来源仍是另一项任务，不能用本轮数据掩盖。",
        "","## 汇报时建议展示哪些图","",
        "- 正常成功率：点显示训练种子，柱显示均值；不要只报最好种子。",
        "- 学习曲线：同样在线交互步下是否更早成功、最后是否退化。",
        "- 扰动表：位置、摩擦、质量、外力改变后是否有效；S08计入body与geom两层旋转后实际轴沿世界Z，自转对称性检查不代表任意三维旋转泛化。",
        "- 配对救回/破坏：同初态参考与残差哪一个成功，不只比总百分比。",
        "- 消融：零增益回到参考，半增益检验小修正是否总是有利；校正/快速闭合分开。",
        "- 接触和残差：穿透、承力、修正幅度，不只看外观视频。",
        "","## 数据文件是什么","",
        "CSV是方便表格软件读的逐行统计；JSON是配置、选模、每回合结果与校验；JSONL是连续训练日志；NPZ是数组压缩包，里面可有参考关节角、物体位置、每时刻状态和动作，不是PPO权重；PKL通常是网络权重，本次资料包排除。",
        "","每个回合的800个时刻高度相关，不是800个独立测试样本；同一权重测64个初态，也不是64次独立训练。BC的许多状态动作记录可能来自同一条参考回放，不是许多独立视频。",
        "","## 可以主张和仍欠缺的内容","",
        "可以主张：实现了可解释的显式残差接口，保留了完整对照、任务接触审计、失败和训练退化数据，并能区分参考质量与学习贡献。不能仅凭这套实验宣称残差概念本身新颖、任何视频都可迁移、比所有调优PPO都强或已经能上硬件。",
        "","仍需：与匹配预算的非视频参考比较、多视频和跨物体划分、更强纯PPO奖励/探索基线、训练随机化与实机力/摩擦/延迟标定。若继续调整奖励，应对相关方法给予对等开发机会，并新建测试集。",
        "","本文件的表格由原始汇总自动生成，逐策略分母、种子和原始指标见`historical_overview.csv`与各实验目录，不手工挑好看的数。"]
    (DOC/"历次实验汇报主线.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
