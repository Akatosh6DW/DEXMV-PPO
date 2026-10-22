"""Combine audited phase/shared critic reports without selecting checkpoints."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np

import analyze_s09_learned_base_residual as report
from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from retarget_s09_reference import digest
from verify_s09_learned_residual_warmup import difference

BASE = Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def main():
    runs = [BASE/"E99_residual_phase_shared_control",BASE/"E98_residual_phase_critic"]
    read = lambda path:json.loads(path.read_text())
    cfg = [read(r/"run_config.json") for r in runs]
    protocols = [read(r/"phase_critic_config.json") for r in runs]
    assert [p["routing"] for p in protocols] == ["shared","phase"]
    assert [p["critic_count"] for p in protocols] == [1,2]
    for key in ("seed","std","steps","evals","eval_count","eval_start","eval_batch","learning_rate",
                "method","model_sha256","reference_sha256","sources","backend"):
        assert cfg[0][key] == cfg[1][key],key
    assert cfg[0]["steps"] == 40960 and cfg[0]["eval_count"] == 64
    for name in ("env_config",):
        assert {k:v for k,v in cfg[0][name].items() if k!="action_variant"} == {
            k:v for k,v in cfg[1][name].items() if k!="action_variant"}
    learned = [read(r/"learned_base_config.json") for r in runs]
    assert {k:v for k,v in learned[0].items() if k!="policy_architecture"} == {
        k:v for k,v in learned[1].items() if k!="policy_architecture"}
    initial = [pickle.loads((r/"step_000000000/params.pkl").read_bytes()) for r in runs]
    assert difference(initial[0][:2],initial[1][:2])["changed_elements"] == 0
    for value in initial[1][2]["phases"]:
        assert difference(initial[0][2],value)["changed_elements"] == 0
    assert read(runs[0]/"step_000000000/episodes.json") == read(runs[1]/"step_000000000/episodes.json")
    with np.load(runs[0]/"step_000000000/initial_states.npz") as a, np.load(runs[1]/"step_000000000/initial_states.npz") as b:
        for key in ("qpos","seeds"):
            np.testing.assert_array_equal(a[key],b[key])
    out = BASE/"report/phase_critic"
    rows,evidence,final = [],{},[]
    for run,protocol in zip(runs,protocols):
        for step in (0,40960):
            audit = read(run/f"step_{step:09d}/native_replay_audit.json")
            assert audit["critic_routing"] == protocol["routing"]
        argv = sys.argv
        sys.argv = [report.__file__,"--run",str(run),"--output",str(out/protocol["routing"])]
        try:
            with redirect_stdout(io.StringIO()):
                report.main()
        finally:
            sys.argv = argv
        stats = read(out/protocol["routing"]/"statistics.json")
        metric = stats["rows"][-1]
        rows.append(dict(run=run.name,routing=protocol["routing"],**stats["success30"],
                         **{k:v for k,v in metric.items() if k not in ("step","success30")},
                         gained=stats["paired"][-1]["gained"],lost=stats["paired"][-1]["lost"]))
        final.append(read(run/"step_000040960/episodes.json"))
        evidence[run.name] = dict(statistics_sha256=digest(out/protocol["routing"]/"statistics.json"),
            protocol_sha256=digest(run/"phase_critic_config.json"),run_config_sha256=digest(run/"run_config.json"))
    pair = transitions(final[0],final[1])
    write_csv(out/"overview.csv",rows)
    result = dict(status="completed_development_only",rows=rows,phase_vs_shared=pair,evidence=evidence,
        source_sha256=digest(Path(__file__)),inherited_online_steps=stats["inherited_online_steps"],
        additional_online_steps_per_arm=40960,scope="one seed, same reused 64-case development set")
    (out/"evidence.json").write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# 起手/保持分阶段critic：配对结果", "",
        "同一基础、同一六残差actor与初始函数、同一奖励和40960在线步。phase组只将共享critic替换为两阶段critic。", "",
        "| 组 | first | last | max | best_step | 新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['routing']} | {r['first']:.2%} | {r['last']:.2%} | {r['max']:.2%} | {r['best_step']} | "
                     f"{r['gained']}/{r['lost']} | {r['hold_sec']:.3f} | {r['max_penetration_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["",f"分阶段相对共享末步：新增{pair['gained']}、丢失{pair['lost']}。",
        "两组各自目录含全部逐例CSV、朝向统计、起末曲线和审计绑定。最大穿透和峰值包括失败案例。",
        "全部保存预览独立核验，不冒称未保存的所有评估案例均逐帧回放。",
        "共享critic34689参数，阶段critic合计69378，容量与路由同时变化，不能单独归因。",
        "不是新的保留集，不是多个训练种子的稳定性结论。基础继承1310720在线步，两组分别新增40960步。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
