"""Check frozen feedback-gain interventions, without claiming training gains."""

import csv
import json
from pathlib import Path
import pickle

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_late_experts import transitions
from analyze_s09_ranked_recovery import BASE
from analyze_s09_recovery_experts import equal_tree
from make_s09_feedback_gain import scaled_experts
from retarget_s09_reference import digest


def main():
    read = lambda p: json.loads(p.read_text())
    bc = BASE / "B15_feedback_recovery_experts"
    source = bc / "step_000000000/params.pkl"
    source_params = pickle.loads(source.read_bytes())
    source_config = read(bc / "run_config.json")
    original = read(bc / "step_000000000/episodes.json")
    runs = [(bc, 1.0, None),
            (BASE / "D50_feedback_zero", 0.0, "B15_zero"),
            (BASE / "D51_feedback_quarter", .25, "B15_quarter")]
    rows, modes, raw, evidence = [], [], [], {}
    for run, gain, name in runs:
        config = read(run / "run_config.json")
        for key in ("model_sha256", "reference_sha256", "eval_start", "eval_count",
                    "eval_batch", "backend", "orientation", "env_config", "policy_architecture"):
            assert config[key] == source_config[key], key
        assert config["steps"] == 0 and config["eval_count"] == 64
        for filename, sha in config["sources"].items():
            assert digest(run / "sources" / filename) == sha
        folder = run / "step_000000000"
        if name is not None:
            derived = BASE / "feedback_gain_inputs" / name
            metadata = read(derived / "feedback_gain.json")
            assert metadata["gain"] == gain and metadata["policy_updates"] == 0
            assert metadata["source_checkpoint_sha256"] == digest(source)
            assert metadata["derived_checkpoint_sha256"] == digest(derived / "params.pkl")
            assert config["checkpoint_sha256"] == metadata["derived_checkpoint_sha256"]
            expected, changed = scaled_experts(source_params, gain)
            assert changed == metadata["changed_parameters"]
            equal_tree(expected, pickle.loads((derived / "params.pkl").read_bytes()))
            evaluated = pickle.loads((folder / "params.pkl").read_bytes())
            equal_tree(expected, evaluated)
            equal_tree(source_params[1]["experts"][4], evaluated[1]["experts"][4])
        summary = read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 0
        with (run / "metrics.csv").open() as stream:
            metrics = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        assert len(metrics) == 1 and np.isfinite(list(metrics[0].values())).all()
        audit = read(folder / "native_replay_audit.json")
        assert audit["status"] == "verified"
        assert audit["checkpoint_sha256"] == digest(folder / "params.pkl")
        assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
        with np.load(folder / "initial_states.npz") as a, np.load(bc / "step_000000000/initial_states.npz") as b:
            for key in ("seeds", "qpos"):
                np.testing.assert_array_equal(a[key], b[key])
        episodes = read(folder / "episodes.json")
        assert len(episodes) == 64
        assert [r["seed"] for r in episodes] == [r["seed"] for r in original]
        assert metrics[0]["success30"] == np.mean([r["success30"] for r in episodes])
        for r in episodes:
            assert r["residual_rms_rad"] == r["released_rope_force_n"] == 0
            if r["success30"]:
                assert r["hold_sec"] >= 30 and r["final_valid"] and not r["forbidden"]
                assert r["initial_penetration_m"] <= .001 and r["max_released_penetration_m"] <= .003
        raw.extend(dict(run=run.name, gain=gain, **r) for r in episodes)
        for mode in range(6):
            old = [r for r in original if r["selected_reference"] == mode]
            new = [r for r in episodes if r["selected_reference"] == mode]
            modes.append(dict(run=run.name, gain=gain, mode=mode, n=len(new),
                              successes=sum(r["success30"] for r in new), **transitions(old, new)))
        rows.append(dict(run=run.name, gain=gain, successes=sum(r["success30"] for r in episodes),
            **summary["success30"], **transitions(original, episodes), hold_sec=metrics[0]["hold_sec"],
            max_depth_mm=1000 * metrics[0]["max_penetration_m"], peak_force_n=metrics[0]["peak_force_n"]))
        evidence[run.name] = {p: digest(run / p) for p in (
            "run_config.json", "metrics.csv", "step_000000000/params.pkl",
            "step_000000000/episodes.json", "step_000000000/native_replay_audit.json")}
    out = BASE / "report/feedback_gain"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("overview.csv", rows), ("orientation.csv", modes), ("episodes.csv", raw)):
        write_csv(out / name, data)
    (out / "evidence.json").write_text(json.dumps(dict(status="verified_development_diagnostic",
        evidence=evidence, source_sha256=digest(Path(__file__)), source_checkpoint_sha256=digest(source),
        additional_policy_updates=0, inherited_bc_updates=44000,
        caveat="Post-hoc gain changes can cause distribution shift; one BC source, reused dev states"), indent=2) + "\n")
    lines = ["# 冻结BC的反馈增益诊断", "", "只改变B15首层非阶段/初始朝向输入的连接倍率，新增训练更新为0。",
        "增益0不是原I27；其他已训练参数仍来自B15。原I27同开发集为26/64。", "",
        "| 增益 | 成功/64 | 相对增益1新增/丢失 | 平均合格秒 | 最大穿透mm | 峰值N |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['gain']} | {r['successes']} | {r['gained']}/{r['lost']} | "
                     f"{r['hold_sec']:.3f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", "每组只有一次0更新评估，first=last=max，best_step=0，不是学习曲线。",
        "同64开发初态；原模型与派生模型参数范围逐项核对，保存预览动作和原生物理核验通过。",
        "第4类无反馈训练，其专家参数不变，仍纳入全部分母；逐朝向结果见orientation.csv。",
        "这不是Stable-BC训练、局部闭环稳定性证明或新测试集确认。接触极值包含失败案例。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
