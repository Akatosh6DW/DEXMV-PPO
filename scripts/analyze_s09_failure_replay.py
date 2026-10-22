"""Bind the E105 high-load failure replay to its original batch and audit."""

import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import plt, write_csv
from retarget_s09_reference import digest


def main():
    base = Path(__file__).resolve().parents[1] / "artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
    original = base / "E105_experts_long_s14123/step_000655360"
    run = base / "D55_E105_failure_replay"
    folder = run / "step_000000000"
    read = lambda p: json.loads(p.read_text())
    cfg = read(run / "run_config.json")
    assert cfg["steps"] == 0 and (cfg["eval_start"], cfg["eval_count"]) == (130024, 8)
    assert cfg["checkpoint_sha256"] == digest(original / "params.pkl")
    old = read(original / "episodes.json")[24:32]
    new = read(folder / "episodes.json")
    assert len(old) == len(new) == 8
    for before, after in zip(old, new):
        assert all(after[key] == value for key, value in before.items())
    audit = read(folder / "native_replay_audit.json")
    assert audit["status"] == "verified" and audit["exact_original_evaluation_reproduced"]
    assert audit["checkpoint_sha256"] == digest(folder / "params.pkl")
    assert audit["trace_sha256"] == digest(folder / "preview_trajectories.npz")
    assert 130027 in [e["seed"] for e in audit["episodes"]]
    with np.load(folder / "preview_trajectories.npz") as z:
        i = list(z["seeds"]).index(130027)
        mask = z["active"][i].astype(bool)
        a = z["audit"][i, mask]
        t = (np.flatnonzero(mask) + 1) * .05
        hold = z["hold"][i, mask]
    case = next(r for r in new if r["seed"] == 130027)
    released = a[:, 10] > 0
    assert not case["success30"]
    assert float(a[:, 7].max()) == case["peak_force_n"]
    assert float(a[released, 6].max()) == case["max_released_penetration_m"]
    rows = [dict(control_interval_end_sec=float(t[k]), thumb_load_n=float(v[0]),
                 opposing_load_n=float(max(v[1:5])), depth_mm=1000*float(v[6]),
                 peak_force_n=float(v[7]), released=bool(v[10]), valid=bool(v[11]),
                 consecutive_valid_sec=float(hold[k])) for k, v in enumerate(a)]
    stats = dict(original_fields_identical=True, same_eight_development_states=True,
        additional_policy_updates=0, case=case,
        peak_force_interval_end_sec=float(t[np.argmax(a[:, 7])]),
        max_depth_interval_end_sec=float(t[np.argmax(a[:, 6])]),
        audit_sha256=digest(folder / "native_replay_audit.json"),
        original_episodes_sha256=digest(original / "episodes.json"),
        replay_episodes_sha256=digest(folder / "episodes.json"),
        analysis_source_sha256=digest(Path(__file__)),
        time_resolution="50ms control intervals, not exact physics-substep event times")
    out = run / "report"
    out.mkdir(exist_ok=True)
    write_csv(out / "failure_timeline.csv", rows)
    (out / "evidence.json").write_text(json.dumps(stats, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    axes[0].plot(t, a[:, 0], label="Thumb", color="#27816e")
    axes[0].plot(t, np.max(a[:, 1:5], axis=1), label="Opposing finger", color="#b65b50")
    axes[0].set(ylabel="Palmar load (N)", title="E105 endpoint, failed development state 130027")
    axes[0].legend()
    axes[1].plot(t, 1000*a[:, 6], color="#b65b50")
    axes[1].axhline(3, color="black", linestyle="--", label="Released depth limit")
    axes[1].set(ylabel="Maximum depth (mm)")
    axes[1].legend()
    axes[2].plot(t, a[:, 7], color="#555555")
    axes[2].set(ylabel="Peak contact force (N)", xlabel="Control interval end (s)")
    for ax in axes:
        ax.axvline(3, color="#c89535", linestyle=":")
        ax.grid(alpha=.2)
    fig.savefig(out / "failure_timeline.png", dpi=160)
    plt.close(fig)
    lines = ["# E105高载荷失败的重复核验", "",
        "原8例开发批次130024--130031的所有原有逐例字段完全相同；130027已进入保存预览并通过动作与独立原生物理审计。不是8个新测试样本。",
        "", f"该例最长合格{case['hold_sec']:.2f}秒、{case['duration_sec']:.2f}秒终止，未达30秒。",
        f"最大释放后穿透{1000*case['max_released_penetration_m']:.3f}mm、峰值{case['peak_force_n']:.3f}N。",
        f"力峰值所在控制区间结束于{stats['peak_force_interval_end_sec']:.2f}秒，最大穿透区间结束于{stats['max_depth_interval_end_sec']:.2f}秒。",
        "", "![失败时间曲线](failure_timeline.png)", "",
        "曲线每点代表50ms控制周期；穿透和力为该周期的物理子步极值，不能把其时间坐标当作精确子步发生时刻。黄色竖线为3秒撤绳。",
        "第一图为各指掌侧合格接触法向力之和的周期最小值，对向指再取最大；第三图为单接触法向力的周期峰值。两者不是同一统计量，不能将数值直接当作互相矛盾。",
        "该图证明风险可以复现，不证明PPO奖励或某个关节是唯一原因；不能用其他成功案例抵消它，也不改变原成功判据。",
        "原始逐周期CSV及完整哈希见failure_timeline.csv和evidence.json。新增0次参数更新。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
