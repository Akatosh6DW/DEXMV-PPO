"""Read-only joint analysis of S02/S04, with explicit S03 historical context."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from report_orca_suite import bootstrap_interval, sign_flip_pvalue, write_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite"
OUT = BASE / "S02_S04_comparison"
DOC = ROOT / "docs/experiments/S02_S04_comparison"
MAIN = ("ppo", "guided", "residual", "bc", "zero")
NAMES = {"ppo": "纯PPO", "guided": "BC后PPO", "residual": "参考+残差PPO", "bc": "仅BC",
         "zero": "零修正参考", "linear": "线性参考+残差", "unrefined": "未校正参考+残差",
         "zero_linear": "线性参考/零修正", "zero_unrefined": "未校正/零修正", "half_residual": "残差修正减半"}
CONDITIONS = ("B0_nominal", "B1_position", "B2_yaw", "B3_friction", "B4_mass", "B5_push")
SHORT = ("原始", "位置±6mm", "旋转±10度", "摩擦×0.6", "质量×2", "弱外力")
COLORS = dict(zip(MAIN, ("#526b80", "#cf8925", "#148875", "#ae8775", "#787d81")))
HASHES = {}
FIGURES = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    HASHES[str(path.relative_to(ROOT))] = digest(path)
    return path


def csv_rows(path):
    with source(path).open() as stream:
        return list(csv.DictReader(stream))


def json_data(path):
    return json.loads(source(path).read_text())


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Expected explicit CSV boolean: " + value)
    return value == "True"


def score(raw, forbidden, depth, threshold=.003):
    return raw and not forbidden and depth <= threshold


def load_episodes(round_name):
    records = []
    for r in csv_rows(BASE / round_name / "report/episodes.csv"):
        item = dict(round=round_name, condition=r.get("benchmark", "B0_nominal"),
                    method="zero" if r["method"] == "zero_video" else r["method"],
                    train_seed=int(r["train_seed"]), case=int(r["seed"]))
        for key in ("success30", "success15", "clean_success30", "substep_forbidden",
                    "valid_before_push", "valid_one_second_after_push"):
            item[key] = boolean(r[key])
        for key in ("longest_hold_sec", "substep_force_peak_n", "max_penetration_m",
                    "post_release_penetration_m", "residual_rms_rad"):
            item[key] = float(r[key])
            if not np.isfinite(item[key]):
                raise ValueError("Nonfinite episode metric")
        assert item["clean_success30"] == score(item["success30"], item["substep_forbidden"], item["post_release_penetration_m"])
        item["full_clean"] = item["clean_success30"] and item["max_penetration_m"] <= .003
        if "full_episode_clean_success30" in r:
            assert item["full_clean"] == boolean(r["full_episode_clean_success30"])
        records.append(item)
    assert len(records) == {"S02": 77600, "S04": 4100}[round_name]
    identities = [(r["condition"], r["method"], r["train_seed"], r["case"]) for r in records]
    assert len(set(identities)) == len(identities)
    return records


def matrix(rows, field="clean_success30"):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["train_seed"]].append(row)
    cases, values = None, []
    for seed in sorted(by_seed):
        data = sorted(by_seed[seed], key=lambda r: r["case"])
        ids = [r["case"] for r in data]
        if len(ids) != len(set(ids)) or (cases is not None and cases != ids):
            raise ValueError("Unpaired or duplicate test starts")
        cases = ids
        values.append([r[field] for r in data])
    return np.asarray(values, float), sorted(by_seed), cases


def outcomes(rows):
    clean = sum(r["clean_success30"] for r in rows)
    dirty = sum(r["success30"] and not r["clean_success30"] for r in rows)
    failed = sum(not r["success30"] for r in rows)
    assert clean + dirty + failed == len(rows)
    return clean, dirty, failed


def setup_plotting():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese font required: " + str(font))
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})


def save(fig, name, caption):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=180)
    plt.close(fig)
    FIGURES.append(dict(name=name, caption=caption))


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    rows = load_episodes("S02") + load_episodes("S04")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["round"], row["condition"], row["method"])].append(row)
    aggregate, per_seed, sensitivity = [], [], []
    matrices = {}
    for key, data in sorted(groups.items()):
        a, seeds, cases = matrix(data)
        untrained = key[2].startswith("zero")
        assert len(seeds) == (1 if untrained else 10)
        assert cases == list(range(40000, 40200) if key[0] == "S02" else range(81000, 81100))
        first_seed = 1001 if key[0] == "S02" else 21001
        assert seeds == ([first_seed] if untrained else list(range(first_seed, first_seed+10)))
        matrices[key] = a
        lo, hi = bootstrap_interval(a)
        c, d, f = outcomes(data)
        summary = dict(round=key[0], condition=key[1], method=key[2], policies=len(seeds), episodes=len(data),
                       clean=a.mean(), raw=np.mean([r["success30"] for r in data]), full_clean=np.mean([r["full_clean"] for r in data]),
                       ci_low=lo, ci_high=hi, clean_count=c, dirty_success_count=d, failure_count=f,
                       hold_mean_sec=np.mean([r["longest_hold_sec"] for r in data]),
                       force_mean_n=np.mean([r["substep_force_peak_n"] for r in data]),
                       penetration_p95_mm=np.quantile([r["post_release_penetration_m"] for r in data], .95)*1000,
                       seed_sd_pp=float(a.mean(axis=1).std(ddof=1)*100) if len(seeds)>1 else None,
                       seed_min=float(a.mean(axis=1).min()), seed_max=float(a.mean(axis=1).max()))
        aggregate.append(summary)
        for seed in seeds:
            samples = [r for r in data if r["train_seed"] == seed]
            successes = [r for r in samples if r["clean_success30"]]
            per_seed.append(dict(round=key[0], condition=key[1], method=key[2], train_seed=seed,
                clean=np.mean([r["clean_success30"] for r in samples]), raw=np.mean([r["success30"] for r in samples]),
                hold_mean_sec=np.mean([r["longest_hold_sec"] for r in samples]),
                force_mean_n=np.mean([r["substep_force_peak_n"] for r in samples]),
                force_clean_success_mean_n=np.mean([r["substep_force_peak_n"] for r in successes]) if successes else None,
                clean_success_count=len(successes), episodes=len(samples),
                penetration_p95_mm=np.quantile([r["post_release_penetration_m"] for r in samples], .95)*1000,
                residual_rms_rad=np.mean([r["residual_rms_rad"] for r in samples])))
        for mm in (2, 3, 4, 5):
            sensitivity.append(dict(round=key[0], condition=key[1], method=key[2], threshold_mm=mm,
                clean=np.mean([score(r["success30"], r["substep_forbidden"], r["post_release_penetration_m"], mm/1000) for r in data])))
    lookup = {(r["round"], r["condition"], r["method"]): r for r in aggregate}
    for run in ("S02", "S04"):
        original = csv_rows(BASE / run / "report/aggregate.csv")
        for r in original:
            method = "zero" if r["method"] == "zero_video" else r["method"]
            current = lookup[(run, r.get("benchmark", "B0_nominal"), method)]
            np.testing.assert_allclose(current["clean"], float(r["clean_success30" if run == "S02" else "clean"]), atol=1e-12)
            np.testing.assert_allclose(current["raw"], float(r["raw_success30" if run == "S02" else "raw"]), atol=1e-12)

    effects = []
    for run in ("S02", "S04"):
        for method in ("guided", "residual"):
            delta = matrices[(run, "B0_nominal", method)] - matrices[(run, "B0_nominal", "ppo")]
            lo, hi = bootstrap_interval(delta)
            values = delta.mean(axis=1)
            effects.append(dict(round=run, comparison=method+"-ppo", difference_pp=delta.mean()*100,
                ci_low_pp=lo*100, ci_high_pp=hi*100, unadjusted_p=sign_flip_pvalue(values),
                positive_seeds=int((values>0).sum()), negative_seeds=int((values<0).sum()), ties=int((values==0).sum())))
    original_tests = {run: csv_rows(BASE / run / "report/paired.csv") for run in ("S02", "S04")}
    for r in effects:
        for old in original_tests[r["round"]]:
            if old["comparison"] == r["comparison"] and old.get("benchmark", "B0_nominal") == "B0_nominal":
                np.testing.assert_allclose(r["difference_pp"], float(old["difference_pp"]))
                np.testing.assert_allclose(r["unadjusted_p"], float(old.get("seed_conditional_p", old.get("p"))))
    cross = []
    for method in MAIN:
        before, after = [lookup[(run, "B0_nominal", method)] for run in ("S02", "S04")]
        cross.append(dict(method=method, S02_clean=before["clean"], S04_clean=after["clean"],
                          observed_difference_pp=(after["clean"]-before["clean"])*100,
                          inference="Descriptive only: different training seeds, test starts and recipes"))

    x = np.arange(len(MAIN))
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for offset, run, color in ((-.19, "S02", "#67829b"), (.19, "S04", "#158f79")):
        data = [lookup[(run, "B0_nominal", m)] for m in MAIN]
        y = np.array([r["clean"]*100 for r in data])
        ax.bar(x+offset, y, .36, label=run, color=color,
               yerr=[[max(0, (r["clean"]-r["ci_low"])*100) for r in data],
                     [max(0, (r["ci_high"]-r["clean"])*100) for r in data]], capsize=3)
        for xx, yy, r in zip(x+offset, y, data):
            ax.text(xx, r["ci_high"]*100+1.5, f"{yy:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [NAMES[m] for m in MAIN])
    ax.set(ylim=(0, 108), ylabel="清洁成功率（%）", title="相同在线预算：163840步；原始方块条件")
    ax.legend(loc="upper left")
    save(fig, "01_nominal", "均值及训练种子/共同初态交叉bootstrap 95%区间。跨轮只作描述，不是配对调参效应。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, run in zip(axes, ("S02", "S04")):
        ys = np.array([matrices[(run, "B0_nominal", m)].mean(axis=1)*100 for m in MAIN[:4]])
        for i in range(10):
            ax.plot(np.arange(4), ys[:, i], color="#aaaaaa", alpha=.45, linewidth=.8)
        for i, method in enumerate(MAIN[:4]):
            ax.scatter(np.full(10, i), ys[i], color=COLORS[method], s=30, zorder=3)
            ax.plot([i-.12, i+.12], [ys[i].mean()]*2, color="black", linewidth=2)
        ax.axhline(lookup[(run, "B0_nominal", "zero")]["clean"]*100, linestyle="--", color="#888888", label="零参考（非十次训练）")
        ax.set_xticks(np.arange(4), [NAMES[m] for m in MAIN[:4]], rotation=12)
        ax.set(title=run+"：每点一个训练种子", ylim=(-3, 103), ylabel="清洁成功率（%）")
        ax.legend(fontsize=8, loc="lower right")
    save(fig, "02_seed_stability", "线只连接同一轮内相同编号的训练种子；不跨S02/S04连线。黑横线为均值。")

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i, row in enumerate(effects):
        ax.errorbar(row["difference_pp"], i,
                    xerr=[[row["difference_pp"]-row["ci_low_pp"]], [row["ci_high_pp"]-row["difference_pp"]]],
                    fmt="o", color=COLORS[row["comparison"].split("-")[0]], capsize=4)
        ax.text(row["ci_high_pp"]+1, i, f"p(未校正)={row['unadjusted_p']:.4f}", va="center", fontsize=9)
    ax.set_yticks(range(4), [r["round"]+" "+NAMES[r["comparison"].split("-")[0]]+" − PPO" for r in effects])
    ax.axvline(0, color="#777777", linestyle="--")
    ax.set(xlim=(-5, 103), xlabel="同轮清洁率差值（百分点）", title="同轮融合方法相对纯PPO的提升：均值与95%区间")
    ax.invert_yaxis()
    save(fig, "03_within_round_effects", "差值95%区间；p为十种子配对精确检验未校正值。正式结论沿用S02 Holm-5与S04 Holm-2，不另选校正口径。")

    curves = []
    for r in csv_rows(BASE / "S02/report/learning_curves.csv"):
        if r["method"] in MAIN[:3]:
            curves.append(dict(round="S02", method=r["method"], seed=int(r["seed"]), step=int(r["step"]), raw=float(r["success30"])))
    for method in MAIN[:3]:
        group = dict(ppo="ppo_tuned", guided="guided_tuned", residual="residual_tuned")[method]
        for seed in range(21001, 21011):
            for r in csv_rows(BASE / "S03/confirmation" / group / f"s{seed}/metrics.csv"):
                if int(r["step"]) <= 163840:
                    curves.append(dict(round="S04", method=method, seed=seed, step=int(r["step"]), raw=float(r["success30"])))
    efficiency = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, run in zip(axes, ("S02", "S04")):
        for method in MAIN[:3]:
            data = [r for r in curves if r["round"] == run and r["method"] == method]
            steps = sorted({r["step"] for r in data})
            seeds = sorted({r["seed"] for r in data})
            values = np.array([[next(r["raw"] for r in data if r["seed"]==s and r["step"]==t) for t in steps] for s in seeds])
            for y in values:
                ax.plot(steps, y*100, color=COLORS[method], alpha=.17, linewidth=.7)
            ax.plot(steps, values.mean(axis=0)*100, "o-", color=COLORS[method], label=NAMES[method], markersize=4)
            for seed, values_one in zip(seeds, values):
                for threshold in (.5, .8):
                    hits = [step for step, rate in zip(steps, values_one) if rate >= threshold]
                    efficiency.append(dict(round=run, method=method, seed=seed, threshold=threshold,
                        reached=bool(hits), first_checkpoint=hits[0] if hits else None, budget=163840,
                        first=values_one[0], last=values_one[-1], maximum=values_one.max(), best_step=steps[int(values_one.argmax())]))
        ax.set(title=run+("：8个开发初态" if run == "S02" else "：16个开发初态（来自S03日志）"),
               xlabel="在线控制步（不含先验准备）", ylabel="开发集原始成功率（%）", ylim=(-3, 103))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save(fig, "04_learning_curves", "统一使用原始成功率，细线为十个种子。两轮开发初态和检查点间隔不同，不当成同一测试集学习速度的严格检验。")

    methods = MAIN + ("linear", "unrefined", "zero_linear", "zero_unrefined", "half_residual")
    heat = np.full((len(methods), 6), np.nan)
    for i, method in enumerate(methods):
        for j, condition in enumerate(CONDITIONS):
            if ("S02", condition, method) in lookup:
                heat[i, j] = lookup[("S02", condition, method)]["clean"]*100
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#ededed")
    im = ax.imshow(heat, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(range(6), SHORT)
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    for i in range(len(methods)):
        for j in range(6):
            v = heat[i,j]
            ax.text(j, i, "未测" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", color="white" if v>65 else "black")
    ax.set_title("S02六条件与参考消融：清洁成功率（%）")
    fig.colorbar(im, ax=ax, shrink=.8, label="%")
    save(fig, "05_s02_robustness", "半修正仅在B0评估，其余格标未测而非0分。其他参考消融均保留。")

    ablations = ("zero_unrefined", "unrefined", "zero_linear", "linear", "zero", "half_residual", "residual")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for i, m in enumerate(ablations):
        r = lookup[("S02", "B0_nominal", m)]
        ax.barh(i, r["clean"]*100, color="#148875" if m=="residual" else "#8299a6",
                xerr=[[max(0,(r["clean"]-r["ci_low"])*100)], [max(0,(r["ci_high"]-r["clean"])*100)]], capsize=3)
        ax.text(r["ci_high"]*100+1, i, f"{r['clean']:.2%}", va="center")
    ax.set_yticks(range(len(ablations)), [NAMES[m] for m in ablations])
    ax.set(xlim=(0, 100), xlabel="清洁成功率（%）", title="S02消融：参考轨迹、仿真校正与残差修正")
    ax.invert_yaxis()
    save(fig, "06_s02_ablation", "不是一个变量的连续加法链。线性/未校正分支分别训练；修正减半是执行时干预。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, run in zip(axes, ("S02", "S04")):
        counts = np.array([outcomes(groups[(run, "B0_nominal", m)]) for m in MAIN], float)
        proportions = counts / counts.sum(axis=1, keepdims=True)*100
        bottom = np.zeros(5)
        for k, (name, color) in enumerate(zip(("清洁成功", "抓住但接触不合格", "原始任务失败"), ("#168b78", "#dba548", "#c7cbce"))):
            ax.bar(x, proportions[:, k], bottom=bottom, color=color, label=name)
            bottom += proportions[:, k]
        ax.set_xticks(x, [NAMES[m] for m in MAIN], rotation=20, ha="right")
        ax.set(title=run+" 原始条件失败分解", ylim=(0, 102), ylabel="回合比例（%）")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save(fig, "07_failure_modes", "三个互斥类别总和为100%。原始任务失败不进一步臆分成碰撞/姿态/力量等未经标注的原因。")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained")
    fields = ("hold_mean_sec", "force_mean_n", "penetration_p95_mm")
    for ax, field, title in zip(axes, fields, ("每策略平均最长保持（秒）", "每策略平均指部峰值力（N）", "每策略撤托后穿透P95（mm）")):
        for j, run in enumerate(("S02", "S04")):
            data = [[r[field] for r in per_seed if r["round"]==run and r["condition"]=="B0_nominal" and r["method"]==m] for m in MAIN[:3]]
            positions = np.arange(3)*3+j*.9
            box = ax.boxplot(data, positions=positions, widths=.65, patch_artist=True, manage_ticks=False, showfliers=False)
            color = ("#67829b", "#158f79")[j]
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(.5)
            for pos, values in zip(positions, data):
                ax.scatter(pos+np.linspace(-.15,.15,len(values)), values, s=10, color=color)
            ax.plot([], [], color=color, linewidth=6, label=run)
        ax.set_xticks(np.arange(3)*3+.45, [NAMES[m] for m in MAIN[:3]], rotation=15)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
    axes[2].axhline(3, color="#ae4d36", linestyle="--", linewidth=1)
    axes[2].set_ylim(0, max(3.3, axes[2].get_ylim()[1]))
    save(fig, "08_contact_and_hold", "每点为一个训练种子的回合汇总；力含失败回合，不能以低力直接证明安全。CSV另列成功条件下力及其分母。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, run in zip(axes, ("S02", "S04")):
        for method in MAIN:
            data = [r for r in sensitivity if r["round"]==run and r["condition"]=="B0_nominal" and r["method"]==method]
            ax.plot([r["threshold_mm"] for r in data], [r["clean"]*100 for r in data], "o-", color=COLORS[method], label=NAMES[method])
        ax.axvline(3, color="#777777", linestyle="--")
        ax.set(title=run+" 原始条件", xlabel="允许的撤托后穿透上限（mm）", ylabel="任务与接触均合格比例（%）", ylim=(0, 102), xticks=[2,3,4,5])
        ax.legend(fontsize=8, loc="lower right")
    save(fig, "09_threshold_sensitivity", "描述性敏感性补充，原始成功及禁接触规则保持不变；官方3mm评分未修改。")

    historical = csv_rows(BASE / "S04/report/historical_conditions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    for ax, budget in zip(axes, (163840, 655360)):
        for i, method in enumerate(MAIN[:3]):
            values = [float(next(r["clean"] for r in historical if r["condition"]==c and r["method"]==method and int(r["budget"])==budget))*100 for c in CONDITIONS[:5]]
            ax.bar(np.arange(5)+(i-1)*.25, values, width=.24, color=COLORS[method], label=NAMES[method])
        ax.set_xticks(range(5), SHORT[:5], rotation=15)
        ax.set(title=f"{budget}步：S03旧测试数据", ylabel="清洁成功率（%）", ylim=(0, 105))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save(fig, "10_budget_boundary", "明确属于S04引用的S03历史补充，不是S04新做的五条件测试；长短点取同一次训练，非独立实验。")

    feedback = json_data(BASE / "S02/report/feedback_diagnostic/summary.json")
    feedback_rows, push_rows = feedback["groups"], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i, method in enumerate(MAIN):
        data = groups[("S02", "B5_push", method)]
        eligible = [r for r in data if r["valid_before_push"]]
        rate = np.mean([r["valid_one_second_after_push"] for r in eligible]) if eligible else None
        push_rows.append(dict(method=method, episodes=len(data), pre_valid=len(eligible), conditional_post_valid=rate))
        axes[0].bar(i, rate*100, color=COLORS[method])
        axes[0].text(i, rate*100+1, f"n={len(eligible)}", ha="center", fontsize=8)
        original_name = "zero_video" if method=="zero" else method
        item = next(r for r in feedback_rows if r["method"]==original_name)
        response = item["mean_target_response_rms_rad"]*1000
        axes[1].bar(i, response, color=COLORS[method])
        axes[1].text(i, response+.015, f"{response:.4g}\nn={item['interpretable_pairs']}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, [NAMES[m] for m in MAIN], rotation=20, ha="right")
    axes[0].set(title="S02：施力前有效回合的后续有效率", ylim=(0, 110), ylabel="11.2秒仍有效（%）")
    axes[1].set(title="仅保存子集：目标动作对外力的响应", ylabel="目标角度响应RMS（毫弧度）", ylim=(0, 1.2))
    save(fig, "11_feedback_limits", "0.15N推力偏弱。右图仅前4个保存初态的有效配对，不是完整总体；有状态依赖输出不等于已证明反馈的因果性能增益。")

    tables = dict(aggregate=aggregate, per_seed=per_seed, effects=effects, cross_round_descriptive=cross,
                  learning_curves=curves, sample_efficiency=efficiency, threshold_sensitivity=sensitivity,
                  push_conditional=push_rows, feedback_subset=feedback_rows)
    for name, data in tables.items():
        write_csv(OUT / "tables" / (name+".csv"), data)
    write_report(lookup, effects, cross, efficiency, FIGURES)
    for path, expected in HASHES.items():
        assert digest(ROOT / path)==expected, "Input modified during analysis: "+path
    outputs = [DOC / "report.md"] + sorted((OUT / "tables").glob("*.csv"))
    for figure in FIGURES:
        png = OUT / "figures" / (figure["name"]+".png")
        pdf = png.with_suffix(".pdf")
        pixels = plt.imread(png)
        assert pixels.shape[0] >= 600 and pixels.shape[1] >= 600
        assert float(pixels[..., :3].std()) > .01, "Blank figure: " + str(png)
        assert pdf.read_bytes().startswith(b"%PDF-"), "Invalid PDF: " + str(pdf)
        outputs.extend((png, pdf))
    assert len(FIGURES) == 11 and len(tables) == 9
    manifest = dict(complete=True, S02_episodes=77600, S04_episodes=4100, unique_combined_episodes=81700,
                    figures=FIGURES, tables={k:len(v) for k,v in tables.items()}, input_sha256=HASHES,
                    output_sha256={str(p.relative_to(ROOT)): digest(p) for p in outputs},
                    nonblank_png_and_pdf_headers_verified=True,
                    script_sha256=digest(Path(__file__)), cross_round_inference="descriptive only",
                    existing_aggregate_scores_recomputed=True, score_rules_recomputed=True,
                    original_hypothesis_families_preserved=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("input_sha256", "output_sha256", "figures")}, ensure_ascii=False))


def write_report(lookup, effects, cross, efficiency, figures):
    prefix = "../../../artifacts/orca_benchmark_suite/S02_S04_comparison"
    lines = ["# S02与S04联合数据分析报告", "", "生成日期：2026-10-09。所有图表由Python从现有数据重新计算，没有启动训练、修改评分或覆盖原实验结果。", "",
        "## 1. 核心结论", "",
        "两轮都支持有限在线预算下视频相关方法的平均成功率高于纯PPO。S04在新的匹配初态上的提升更明显，且两项原定主检验通过校正。",
        "但S04并不是在相同种子、相同初态下只改一个参数重做S02，跨轮涨幅不能直接解释成某个超参数的因果收益。",
        "S02主要补充多条件和消融证据，S04主要补充调优后配方在有限预算下的表现。两者应互补使用，不混成同一次实验。", "",
        "| 方法 | S02原始条件清洁率 | S04新初态清洁率 | 跨轮观察差值 |", "|---|---:|---:|---:|"]
    for r in cross:
        lines.append(f"| {NAMES[r['method']]} | {r['S02_clean']:.2%} | {r['S04_clean']:.2%} | {r['observed_difference_pp']:+.2f}个百分点 |")
    lines += ["", "## 2. 数据与可比性", "",
        "| 项目 | S02 | S04 |", "|---|---|---|",
        "| 模型、物体、来源视频 | Orca、53mm/35g方块、HIC20 | 相同 |",
        "| 主比较在线预算 | 163840步 | 同样163840步 |",
        "| 权重来源 | 原配方独立训练 | S03调优后训练的对应早期检查点 |",
        "| 训练种子 | 1001..1010 | 21001..21010 |",
        "| 测试初态 | 40000..40199，每策略200个 | 81000..81099，每策略100个 |",
        "| 本轮数据范围 | 六条件、消融，77600回合 | 原始条件五方法，4100回合 |",
        "| 开发曲线 | 8个开发初态，40960步间隔 | 16个开发初态，81920步间隔 |",
        "| 原定主检验校正 | B0五项Holm-5 | 两项Holm-2 |", "",
        "合计81700条不同实验回合。本次逐行核验评分代数、回合身份及已有汇总；这不是重新仿真81700回合。",
        "零参考是固定控制器，仅一组初态回放，不能假装成十个独立训练种子。BC与PPO的同种子可在同轮内配对，S02与S04不能按末尾编号强行配对。",
        "配方变化：纯PPO保留原学习率/探索配方；BC后PPO学习率3e-5变为7.5e-6；残差探索标准差0.3变为0.7，并去掉前期接近/接触辅助奖励。",
        "网络结构、控制接口及成功定义未因此变成新算法。两种融合的语义仍分别是先模仿再PPO，以及每步参考加PPO修正。", "",
        "## 3. 主结果与种子稳定性", "",
        f"![主结果]({prefix}/figures/01_nominal.png)", "",
        "误差线是训练种子与共同测试初态两个维度的交叉bootstrap 95%区间。S02每策略200回合和S04每策略100回合不是同一组样本。",
        f"![种子稳定性]({prefix}/figures/02_seed_stability.png)", "",
        "每点为一份训练策略的成功率。连线只表示同轮同种子，不能把不同方法之间的每条灰线当作训练过程。",
        "S04的BC后PPO相对PPO为9/10种子正差，残差为10/10种子正差；平均提升不能保证每次训练都更好。", "",
        "| 轮次 | 方法 | 种子间标准差（百分点） | 最低种子成功率 | 最高种子成功率 |",
        "|---|---|---:|---:|---:|"]
    for run in ("S02", "S04"):
        for method in MAIN[:3]:
            r = lookup[(run, "B0_nominal", method)]
            lines.append(f"| {run} | {NAMES[method]} | {r['seed_sd_pp']:.2f} | {r['seed_min']:.2%} | {r['seed_max']:.2%} |")
    lines += ["", "两轮内残差方法的种子波动均小于另外两种方法，说明本任务下训练结果更集中；这不是额外的方差显著性检验。残差自身的跨轮标准差从3.56增至5.86个百分点，不能说S04所有指标都改善。", "",
        "| 同轮比较 | 均值差（百分点） | 95%差值区间 | 未校正配对p | 正/负/平种子 |",
        "|---|---:|---|---:|---|"]
    for r in effects:
        lines.append(f"| {r['round']} {r['comparison']} | {r['difference_pp']:+.2f} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | {r['unadjusted_p']:.6f} | {r['positive_seeds']}/{r['negative_seeds']}/{r['ties']} |")
    lines += ["", f"![差值]({prefix}/figures/03_within_round_effects.png)", "",
        "本图补充统一的效应量展示，不另造新的主检验。S02残差减PPO原定Holm-5 p=0.052734，未达0.05；S04两项原定Holm-2 p分别0.005859和0.003906。",
        "两轮检验家族不同，不能把p值变小全归因于算法变强。S02 BC后PPO减PPO在这里是补充比较，不追认为S02原定主假设。",
        "S04残差85.10%与BC后PPO84.00%很接近，本分析不据1.10个百分点断言残差优于BC路线。", "",
        "## 4. 学习过程与预算", "", f"![学习曲线]({prefix}/figures/04_learning_curves.png)", "",
        "重新从S03训练日志提取S04的原始成功率，避免把S02原始率与S04清洁率混画。粗线是均值，细线是每个训练种子。",
        "BC在横轴零点前已经进行了4000次监督更新；残差在零点前已经获得校正参考。因此零点不是所有方法都没有准备成本。",
        "开发集不同且样本少，曲线主要用于观察早期能力、稳定性和退化，不作同一测试集的严格时间到达比较。",
        "阈值表只记录第一次在保存检查点达到阈值的步数；未达到者记为右删失，CSV保留空值，不删除或填成0。", "",
        "| 轮次 | 方法 | 达到50%的种子数 | 达到80%的种子数 |", "|---|---|---:|---:|"]
    for run in ("S02", "S04"):
        for method in MAIN[:3]:
            counts = [sum(r["reached"] for r in efficiency if r["round"]==run and r["method"]==method and r["threshold"]==t) for t in (.5,.8)]
            lines.append(f"| {run} | {NAMES[method]} | {counts[0]}/10 | {counts[1]}/10 |")
    lines += ["", "## 5. S02多条件与消融", "", f"![六条件]({prefix}/figures/05_s02_robustness.png)", "",
        "这张图只属于S02；S04没有重测六个条件。半修正分支只有B0结果，其余格明确标未测。",
        "视频相关方法在位置或姿态变化下不总保持原有优势；质量加倍明显伤害BC后PPO。低摩擦下零参考本身接近饱和，不能把全部高分都解释成PPO反馈的功劳。",
        f"![消融]({prefix}/figures/06_s02_ablation.png)", "",
        "S02原始条件：未校正参考加残差22.20%、线性参考加残差40.40%、完整残差69.80%，提示参考轨迹及仿真校正在这套实现中重要。",
        "但完整残差相对校正参考零修正只从68.00%到69.80%，原检验不支持显著增益；执行时修正减半为65.40%。不能只比较最差参考就断言PPO修正贡献很大。",
        "S04对应对照为仅BC63.50%→BC后PPO84.00%，零参考66.00%→残差85.10%。这两个差值是平均增益的描述，不在此新增显著性胜出声明。", "",
        "## 6. 抓住、接触质量与失效", "", f"![失败分解]({prefix}/figures/07_failure_modes.png)", "",
        "失败分解是互斥的三类：清洁成功、原始成功但接触质量未过门槛、原始任务失败。不是靠总奖励判成功。",
        "S04主三组原始率与清洁率几乎相同，因此相对PPO的大幅提升不是仅由3mm门槛造成。S02 BC后PPO撤托后清洁率68.90%，全程穿透合格率63.95%，前期接触仍需单独审查。",
        f"![接触质量]({prefix}/figures/08_contact_and_hold.png)", "",
        "每个箱图中的点是一份策略的回合汇总，不把200个相关回合当成200次独立训练。指部峰值力是活跃回合中记录的物理子步峰值，再对回合平均；不是电机额定力矩或同一时刻全部关节载荷。",
        "图中力含成功和失败回合，较低力可能与较早失败有关，不能仅据低力推断更安全。逐种子CSV另列清洁成功条件下的力及成功分母，分母为0时留空。",
        "穿透图使用每个策略回合最大穿透的P95，而官方成功要求每回合不超过门槛；P95低于3mm不代表全部回合合格。",
        f"![阈值]({prefix}/figures/09_threshold_sensitivity.png)", "",
        "2/3/4/5mm只作敏感性分析，保持原始成功和禁接触条件不变；正式评分仍为3mm，不选择最有利阈值替代。全部条件的阈值表均导出，图只展示原始条件。", "",
        "## 7. 优势的边界和反馈机制", "", f"![长短预算]({prefix}/figures/10_budget_boundary.png)", "",
        "这是S04引用的S03历史数据，显式独立于S04新复测。短预算时五个条件有4/5两种融合均高于PPO，质量加倍是反例；长预算时只有1/5满足两者都高于PPO。",
        "所以应主张匹配先验能减少在线探索需求，而不是所有预算、所有物理条件下都更强。BC后PPO长训退化提示需要分析遗忘和更新稳定性，不能通过挑中间最好点掩盖。",
        f"![反馈诊断]({prefix}/figures/11_feedback_limits.png)", "",
        "S02外力仅0.15N持续0.2秒。左图以施力前有效回合为分母；右图来自每策略前4个保存初态的配对诊断，分母明确显示，不推广成全测试集统计。",
        "可以观察到输出随扰动改变，但这不足以证明闭环修正比固定偏置更有性能优势。S04未完成闭环与固定回放机制实验，不补造这项结论。", "",
        "## 8. 如何组织实验结论", "",
        "主结论：在已有匹配视频先验、固定Orca方块任务和163840步在线预算下，BC后PPO与参考加残差PPO均比纯PPO取得更高平均成功率；S04的两项原定主检验支持其提升。",
        "机制证据：BC/零参考对照显示先验提供基础能力，进一步PPO学习在S04中带来平均增益；S02消融显示参考质量重要。",
        "边界：跨轮变好不能只归因于调参；有限在线步数相同不代表BC、视频处理与开发搜索总成本相同；一条视频不代表跨视频泛化；没有真机或手指自碰撞安全证明。",
        "后续若要定位改进原因，可在相同训练种子与初态下分别改变学习率、探索幅度、辅助奖励，建立单变量复现实验。本次不增加训练，先把已有证据说清楚。", "",
        "## 9. 图表、数据与复现", "",
        "全部图表提供PNG和PDF，CSV保留未四舍五入的数据。脚本只读取原结果，所有输入文件SHA256记录在manifest.json中。",
        "```bash", "conda activate aero_rl", "python scripts/analyze_orca_s02_s04.py",
        "python -m unittest discover -s scripts -p 'test_orca_s02_s04_analysis.py'", "```", "",
        f"- [完整输入/输出清单]({prefix}/manifest.json)", f"- [逐条件汇总]({prefix}/tables/aggregate.csv)",
        f"- [逐种子诊断]({prefix}/tables/per_seed.csv)", f"- [同轮效应量]({prefix}/tables/effects.csv)",
        f"- [学习过程与首次达标]({prefix}/tables/sample_efficiency.csv)", f"- [穿透阈值敏感性]({prefix}/tables/threshold_sensitivity.csv)",
        "- [S02原报告](../S02/README.md)", "- [S04原报告](../S04/README.md)", "",
        "### 图注索引", ""]
    for figure in figures:
        lines.append(f"- **{figure['name']}**：{figure['caption']} [PNG]({prefix}/figures/{figure['name']}.png) / [PDF]({prefix}/figures/{figure['name']}.pdf)")
    (DOC / "report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
