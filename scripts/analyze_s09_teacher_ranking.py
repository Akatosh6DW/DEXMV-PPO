"""Verify fixed-pool teacher ranking evidence and interaction accounting."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from probe_s09_recovery_teacher import objective_value, summarize
from probe_s09_teacher_ranking import shortlist
from retarget_s09_reference import digest


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=root/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/D43_teacher_ranking")
    folder=parser.parse_args().folder
    read=lambda path:json.loads(path.read_text())
    summary=read(folder/"summary.json")
    protocol=read(folder/"protocol.json")
    assert summary["status"]=="completed" and not summary["smoke"]
    assert summary["cases"]==len(protocol["training_seeds"])
    assert summary["policy_updates"]==0 and summary["new_search_evaluations"]==0
    for name,sha in protocol["sources"].items():
        assert digest(folder/"sources"/name)==sha
    origin=Path(protocol["origin"])
    assert digest(origin/"protocol.json")==protocol["origin_sha256"]
    source=read(origin/"protocol.json")
    assert digest(Path(source["nominal"]))==protocol["checkpoint_sha256"]
    assert digest(root/"artifacts/orca_benchmark_suite/S09/inputs/model.xml")==protocol["model_sha256"]
    results=read(folder/"results.json")
    assert [r["seed"] for r in results]==protocol["training_seeds"]
    rows,hashes=[],{}
    cost=dict(full_horizon_selection=0,selected_replay_validation=0,canonical_collection=0,canonical_validation=0)
    for result in results:
        seed=result["seed"]
        case=folder/f"seed{seed}"
        old=origin/f"seed{seed}"
        assert digest(old/"search.json")==result["search_sha256"]
        assert digest(old/"initial_state.npz")==result["snapshot_sha256"]
        candidates=read(case/"candidates.json")
        assert len(candidates)==result["candidates"]
        history=read(old/"search.json")
        assert [c["index"] for c in candidates]==shortlist(history,protocol.get("candidate_count_requested",8))
        short=min(candidates,key=lambda r:r["short_cost"])
        long=min(candidates,key=lambda r:r["long_cost"])
        assert short["index"]==result["short_index"] and long["index"]==result["long_index"]
        assert result["short_trace_matches_original"] and result["selected_replay_identical"]
        for c in candidates:
            assert c["delta"]==history[c["index"]]["delta"]
            assert c["short_cost"]==history[c["index"]]["cost"]
            path=case/f"candidate_{c['index']:03d}.npz"
            hashes[str(path.relative_to(folder))]=digest(path)
            with np.load(path,allow_pickle=False) as z:
                assert all(np.isfinite(z[k]).all() for k in z.files)
                count=len(z["alive"])
                assert count==c["executed_steps"]
                assert bool(z["success30"][-1] and z["alive"][-1])==c["success30"]
                assert float(z["best_hold_sec"].max())==c["best_hold_sec"]
                fields={k:z[k] for k in ("valid","loads","penetration_m","forbidden",
                    "height_loss_m","alive","peak_force_n","rope_force_n")}
                native_rows=[{k:v[i] for k,v in fields.items()} for i in range(count)]
                checked=summarize(native_rows,760)
                for k,v in checked.items():
                    np.testing.assert_allclose(v,c[k],atol=1e-12,rtol=0)
                np.testing.assert_allclose(objective_value(checked,np.array(c["delta"])),c["long_cost"],atol=1e-12,rtol=0)
                cost["full_horizon_selection"]+=count
                if c["index"] in {short["index"],long["index"]}:
                    cost["selected_replay_validation"]+=count
        if result["canonical"]:
            canonical=result["canonical"]
            path=case/"canonical_trace.npz"
            assert digest(path)==canonical["trace_sha256"]
            hashes[str(path.relative_to(folder))]=digest(path)
            assert canonical["native_qpos_error"]<3e-6 and canonical["native_qvel_error"]<1e-4
            with np.load(path,allow_pickle=False) as z:
                assert all(np.isfinite(z[k]).all() for k in z.files)
                assert bool(z["success30"][0,-1])==canonical["qualified"]
                cost["canonical_collection"]+=int(z["active"].sum())
                cost["canonical_validation"]+=int(z["active"].sum())
        rows.append(dict(seed=seed,candidates=len(candidates),short_index=short["index"],long_index=long["index"],
            short_success=short["success30"],long_success=long["success30"],
            short_hold_sec=short["best_hold_sec"],long_hold_sec=long["best_hold_sec"],
            any_candidate_success=any(c["success30"] for c in candidates),
            canonical_success=bool(result["canonical"] and result["canonical"]["qualified"])))
    assert cost==summary["costs"]
    for key,field in (("short_successes","short_success"),("long_successes","long_success"),
                      ("any_candidate_successes","any_candidate_success"),("canonical_successes","canonical_success")):
        assert sum(r[field] for r in rows)==summary[key]
    out=folder/"report"
    out.mkdir(exist_ok=True)
    write_csv(out/"overview.csv",rows)
    evidence=dict(status="verified",source_sha256=digest(Path(__file__)),summary_sha256=digest(folder/"summary.json"),
        results_sha256=digest(folder/"results.json"),trace_hashes=hashes,costs=cost,
        scope="Teacher ranking on historical training failures; not student policy improvement")
    (out/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    n=summary["cases"]
    top=protocol.get("candidate_count_requested",8)
    lines=[f"# {folder.name}：同一批补救动作，短时评分是否选错", "",
        f"原D38的{n}个训练查询状态，每个评估旧短时成本前{top}个不重复动作（不足取全部）及零偏移。没有新搜索。",
        "成本公式不变，只将其模拟时长从8秒扩展到原40秒终点。长时选择额外计算成本单列。", "",
        "| 种子 | 候选数 | 短/长所选编号 | 短/长严格成功 | 短/长保持秒 | 池内有成功 | 长赢家接口成功 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['candidates']} | {r['short_index']}/{r['long_index']} | "
            f"{int(r['short_success'])}/{int(r['long_success'])} | {r['short_hold_sec']:.2f}/{r['long_hold_sec']:.2f} | "
            f"{int(r['any_candidate_success'])} | {int(r['canonical_success'])} |")
    lines += ["",f"短时选择{summary['short_successes']}/{n}；长时选择{summary['long_successes']}/{n}；"
        f"候选池中存在成功动作{summary['any_candidate_successes']}/{n}；长时赢家实际动作接口成功{summary['canonical_successes']}/{n}。",
        "候选池上限不是学生成绩。即使全部历史候选都失败，也不能声称已穷尽所有可能抓法。",
        f"物理成本：全时长选择{cost['full_horizon_selection']}步，所选重复核验{cost['selected_replay_validation']}步，"
        f"实际接口采集{cost['canonical_collection']}步、独立验证{cost['canonical_validation']}步；0次BC/PPO更新。",
        "旧短时赢家全轨迹逐元素复现；全候选有限性、成本重算、成功标签及哈希已核对；选中轨迹重复核验。",
        "阶段结果不是新未见测试，不将被历史失败筛出的案例成功率外推到随机朝向任务。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(rows=rows,costs=cost)),flush=True)


if __name__=="__main__":
    main()
