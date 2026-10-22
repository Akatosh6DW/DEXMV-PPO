"""Analyze the declared eight-case post-hoc phase intervention, not a benchmark."""

import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import BASE,transitions
from retarget_s09_reference import digest


def main():
    read = lambda path:json.loads(path.read_text())
    runs = dict(zero=BASE/"E85_learned_base_residual/step_000000000",
        full=BASE/"E85_learned_base_residual/step_000040960",
        before_release=BASE/"D32_residual_before_release/step_000000000",
        after_release=BASE/"D33_residual_after_release/step_000000000")
    groups,rows,evidence = {},[],{}
    common_qpos = None
    for name,folder in runs.items():
        eps = [r for r in read(folder/"episodes.json") if 130016<=r["seed"]<130024]
        assert [r["seed"] for r in eps] == list(range(130016,130024))
        with np.load(folder/"initial_states.npz",allow_pickle=False) as z:
            selected = (z["seeds"]>=130016)&(z["seeds"]<130024)
            if common_qpos is None:
                common_qpos = z["qpos"][selected].copy()
            np.testing.assert_array_equal(common_qpos,z["qpos"][selected])
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified"
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        if name.endswith("release"):
            config = read(folder.parent/"run_config.json")
            probe = read(folder.parent/"phase_probe.json")
            assert config["steps"] == 0 and probe["optimizer_updates"] == 0 and probe["phase"] == name
            assert config["checkpoint_sha256"] == digest(runs["full"]/"params.pkl")
            assert audit["phase_intervention_sha256"] == digest(folder.parent/"phase_probe.json")
        groups[name] = eps
        rows.extend(dict(group=name,**r) for r in eps)
        evidence[name] = dict(episodes_sha256=digest(folder/"episodes.json"),audit_sha256=digest(folder/"native_replay_audit.json"))
    aggregate = [dict(group=name,n=len(eps),successes=sum(r["success30"] for r in eps),
        mean_hold_sec=float(np.mean([r["hold_sec"] for r in eps])),
        max_released_depth_mm=max(r["max_released_penetration_m"] for r in eps)*1000,
        **transitions(groups["zero"],eps)) for name,eps in groups.items()]
    result = dict(status="completed_posthoc_diagnostic",aggregate=aggregate,evidence=evidence,
        selection="Consumed development seeds130016..130023, selected to include E85 gain/loss cases",
        optimizer_updates=0,source_sha256=digest(Path(__file__)),
        scope="Same checkpoints with deterministic temporal gates. Not a newly trained model or held-out generalization estimate.")
    output = BASE/"report/residual_phase_probe"
    output.mkdir(parents=True,exist_ok=True)
    (output/"statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    write_csv(output/"episodes.csv",rows)
    write_csv(output/"aggregate.csv",aggregate)
    lines = ["# E85修正阶段诊断", "", "这是事后选取的八例开发诊断，没有训练更新，不能报告为总体或保留测试成功率。", "",
        "| 控制 | 成功/8 | 平均合格秒 | 全案例释放后最大穿透mm | 相对零修正新增 | 丢失 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['group']} | {r['successes']} | {r['mean_hold_sec']:.3f} | {r['max_released_depth_mm']:.3f} | {r['gained']} | {r['lost']} |")
    lines += ["", "| seed | 零修正 | 全阶段 | 只起手 | 只后阶段 |", "|---|---|---|---|---|"]
    for i in range(8):
        cells = ["成功" if eps[i]["success30"] else "失败" for eps in groups.values()]
        lines.append(f"| {130016+i} | "+" | ".join(cells)+" |")
    lines += ["", "130018的新成功和130021/130023的丢失都随起手修正出现。关闭起手恢复旧成功，却同时移除了新增成功，不能把关门控当作训练修复。",
        "只后阶段修正中130022释放后穿透4.092mm，仍判失败；更长平均抓持不等于接触合格。",
        "阶段边界按控制时刻step60（3秒），不是物理子步内切换。保存预览通过动作/门控/物理审计，不声称全部未保存轨迹也逐帧重放。",
        "不同执行图仍可能有微小浮点差异；该干预支持定位此检查点的阶段性影响，不证明唯一优化失败机制。"]
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
