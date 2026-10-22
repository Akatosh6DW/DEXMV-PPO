"""Aggregate paired raw S07 results, verify traces, and measure grasp style."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices
from orca_s06_env import PAD_NAMES, cylinder_contacts
from prepare_cylinder_s02_source import digest
from run_s07_study import OUT, ROOT, METHODS, SEEDS, BENCHMARKS, write_json

REPORT = OUT / "report"
LABEL = dict(ppo="Pure PPO", guided="BC + PPO", residual="Reference + residual PPO", bc="BC only", reference="Reference only")
COLORS = dict(ppo="#8b9198", guided="#b66a3c", residual="#258b78", bc="#7470a7", reference="#4486b8")


def write_csv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def wilson(k, n):
    p, z = k/n, 1.96
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def verify_run_files(run):
    config = json.loads((run / "run_config.json").read_text())
    for name, checksum in config["source_sha256"].items():
        assert digest(run / "sources" / name) == checksum
    for field in ("model", "reference"):
        assert digest(config[field]) == config[field+"_sha256"]
    if config.get("checkpoint"):
        assert digest(config["checkpoint"]) == config["checkpoint_sha256"]
    return config


def verify_trace(trace, episodes, config, packet, model, gain, bound):
    count = len(trace["qpos"])
    assert all(np.isfinite(v).all() for v in trace.values())
    pair = np.all(trace["pad_audit"][..., :2] > 0, -1) & ~np.any(trace["pad_audit"][..., 2:4] > 0, -1)
    np.testing.assert_array_equal(pair, trace["pad_pair"].astype(bool))
    stable = (pair & (trace["substep_forbidden"] == 0) & (trace["substep_penetration"] <= .003)
              & (trace["broad_hold"] > 0) & trace["alive"].astype(bool) & trace["active"].astype(bool))
    hold = np.zeros((count, 800))
    for t in range(800):
        hold[:, t] = np.where(stable[:, t], (hold[:, t-1] if t else 0)+.05, 0)
    np.testing.assert_allclose(hold, trace["hold"], atol=5e-6)
    max_control_error = 0.
    for i, row in enumerate(episodes[:count]):
        active = trace["active"][i].astype(bool)
        released = active & (trace["released"][i] > 0)
        depth = float(np.max(trace["substep_penetration"][i, released], initial=0))
        forbidden = bool(np.any(trace["substep_forbidden"][i, released] > 0))
        passed = bool(hold[i].max() >= 30 and hold[i, -1] > 0 and not forbidden and depth <= .003)
        assert passed == row["success30"]
        if config["method"] == "residual":
            requested = bound*np.clip(gain*trace["action"][i], -1, 1)
            np.testing.assert_allclose(requested[active], trace["requested_residual"][i, active], atol=2e-7)
            desired = np.clip(trace["video_base"][i]+requested, *model.actuator_ctrlrange.T)
            previous = np.vstack([packet["initial_ctrl"], trace["ctrl"][i, :-1]])
            predicted = np.clip(desired, previous-.1, previous+.1)
            error = float(np.max(np.abs(predicted[active]-trace["ctrl"][i, active])))
            assert error < 1e-6
            max_control_error = max(max_control_error, error)
    return count, max_control_error


def similarity(run, trace, episodes, model, packet):
    d = mujoco.MjData(model)
    pads = np.array([model.geom(n).id for n in PAD_NAMES])
    bodies = model.geom_bodyid[pads]
    oid = model.body("cube").id
    qids, _, _ = model_indices(model)
    targets = packet["target_body_tips"]
    rows, summary = [], []
    for ei in range(len(trace["qpos"])):
        episode_rows = []
        # Sample every 0.5 s and densely during acquisition. Success itself
        # always comes from saved 5 ms runtime audits, never these reconstructions.
        frames = sorted(set(range(0, 30, 2)) | set(range(59, 800, 10)))
        for frame in frames:
            if not trace["active"][ei, frame]:
                continue
            t = (frame+1)*.05
            for field, key in (("qpos", "qpos"), ("qvel", "qvel"), ("ctrl", "ctrl"), ("mocap_pos", "mocap")):
                getattr(d, field)[:] = trace[key][ei, frame]
            mujoco.mj_forward(model, d)
            phase = min(t/1.5, 1.)*(len(targets)-1)
            lo, hi = int(phase), min(int(phase)+1, len(targets)-1)
            target = targets[lo]*(1-(phase-lo))+targets[hi]*(phase-lo)
            rot = d.xmat[oid].reshape(3, 3)
            actual = (d.geom_xpos[pads]-d.xpos[oid]) @ rot
            error = np.linalg.norm(actual-target, axis=1)*1000
            inward = -actual.copy()
            inward[:, 1] = 0
            inward /= np.maximum(np.linalg.norm(inward, axis=1)[:, None], 1e-9)
            normal = -d.xmat[bodies].reshape(3, 3, 3)[:, :, 0] @ rot
            facing = np.sum(normal*inward, axis=1)
            good, bad, _ = cylinder_contacts(model, d)
            base = np.array([np.interp(t, packet["time"], packet["ctrl"][:, k]) for k in range(17)])
            row = dict(run=run, episode=ei, seed=episodes[ei]["seed"], time=t,
                       phase="acquisition" if t <= 1.5 else "holding", valid=bool(trace["hold"][ei, frame] > 0),
                       joint_target_rms_rad=float(np.sqrt(np.mean((trace["ctrl"][ei, frame]-base)**2))),
                       actual_joint_rms_rad=float(np.sqrt(np.mean((d.qpos[qids]-base)**2))),
                       sampled_good_three=bool(np.all(good > .02) and not np.any(bad > .02)))
            for j, finger in enumerate(("thumb", "index", "middle")):
                row.update({finger+"_error_mm": float(error[j]), finger+"_facing_cos": float(facing[j]),
                            finger+"_good_n": float(good[j]), finger+"_bad_n": float(bad[j])})
            rows.append(row)
            episode_rows.append(row)
        for phase in ("acquisition", "holding", "valid_holding"):
            chosen = [r for r in episode_rows if r["phase"] == ("holding" if phase == "valid_holding" else phase)
                      and (phase != "valid_holding" or r["valid"])]
            if not chosen:
                continue
            stats = dict(run=run, episode=ei, seed=episodes[ei]["seed"], success30=episodes[ei]["success30"],
                         phase=phase, frames=len(chosen))
            keys = [k for k in chosen[0] if k.endswith(("_mm", "_cos", "_rad"))]
            for key in keys:
                stats[key] = float(np.mean([r[key] for r in chosen]))
            stats["sampled_good_three_fraction"] = float(np.mean([r["sampled_good_three"] for r in chosen]))
            summary.append(stats)
    return rows, summary


def main():
    REPORT.mkdir(exist_ok=True)
    completion = json.loads((OUT / "evaluation_complete.json").read_text())
    assert completion["selection_sha256"] == digest(OUT / "selection.json")
    frozen = json.loads((OUT / "selection.json").read_text())
    for item in frozen["selected"].values():
        assert digest(item["checkpoint"]) == item["sha256"]
    for path, checksum in json.loads((OUT / "inputs_manifest.json").read_text()).items():
        assert digest(ROOT / path) == checksum
    for method in METHODS:
        for seed in SEEDS:
            run = OUT / "train" / f"{method}_s{seed}"
            config = verify_run_files(run)
            status = json.loads((run / "summary.json").read_text())
            assert status["status"] == "completed" and status["final_step"] == 163840
            assert config["steps"] == 163840 and config["learning_rate"] == .0001
            assert config["seed"] == seed and config["method"] == method
            assert [r["step"] for r in status["rows"]] == list(range(0, 163841, 20480))
    rows, summaries, initials, style, style_summary = [], [], {}, [], []
    residual_stats = []
    verified, max_error, inputs = 0, 0., {}
    dirs = sorted(OUT.glob("eval/*/*/summary.json"))
    assert len(dirs) == completion["jobs"]
    for summary_path in dirs:
        run = summary_path.parent
        bench, name = run.parent.name, run.name
        config = verify_run_files(run)
        env = json.loads((run / "env_config.json").read_text())["s07"]
        expected_mass = .07 if bench == "B4_mass" else .035
        assert np.isclose(env["object_mass"], expected_mass)
        np.testing.assert_allclose(env["object_friction"], [1.35*(.6 if bench == "B3_friction" else 1), .025, .004])
        directory = run / "step_000000000"
        episodes = json.loads((directory / "episodes.json").read_text())
        method = name.split("_s")[0] if "_s" in name else name
        seed = int(name.split("_s")[1].split("_")[0]) if "_s" in name else -1
        with np.load(directory / "initial_states.npz") as z:
            pair_group = bench if bench in BENCHMARKS else "B0_nominal"
            initial = initials.setdefault(pair_group, z["qpos"].copy())
            np.testing.assert_array_equal(initial, z["qpos"])
            np.testing.assert_array_equal(z["seeds"], [r["seed"] for r in episodes])
        for e in episodes:
            rows.append(dict(benchmark=bench, run=name, method=method, train_seed=seed, **e))
        k, n = sum(e["success30"] for e in episodes), len(episodes)
        summaries.append(dict(benchmark=bench, run=name, method=method, train_seed=seed, successes=k, count=n,
            success_rate=k/n, ci_low=wilson(k, n)[0], ci_high=wilson(k, n)[1],
            mean_hold=float(np.mean([r["pad_hold_sec"] for r in episodes])),
            max_depth_mm=max(r["penetration_m"] for r in episodes)*1000,
            max_force_n=max(r["peak_force_n"] for r in episodes), forbidden=sum(r["forbidden"] for r in episodes),
            residual_rms=float(np.mean([r["residual_rms_rad"] for r in episodes]))))
        key = (config["model"], config["reference"])
        if key not in inputs:
            with np.load(config["reference"]) as z:
                packet = dict(z)
            inputs[key] = (mujoco.MjModel.from_xml_path(config["model"]), packet)
        model, packet = inputs[key]
        with np.load(directory / "preview_trajectories.npz") as z:
            trace = dict(z)
        count, error = verify_trace(trace, episodes, config, packet, model, env["residual_gain"], env["residual_bound"])
        verified += count
        max_error = max(max_error, error)
        if config["method"] == "residual":
            active = trace["active"].astype(bool)
            requested = trace["requested_residual"][active]
            effective = trace["effective_residual"][active]
            residual_stats.append(dict(benchmark=bench, run=name, saved_episodes=count,
                gain=env["residual_gain"], bound_rad=env["residual_bound"],
                request_rms_rad=float(np.sqrt(np.mean(requested**2))),
                request_max_rad=float(np.max(np.abs(requested))),
                effective_rms_rad=float(np.sqrt(np.mean(effective**2))),
                effective_max_rad=float(np.max(np.abs(effective))),
                saturation_fraction=float(np.mean(np.abs(requested) >= .99*env["residual_bound"])),
                recomposition_max_error_rad=error))
        if bench == "B0_nominal":
            detail, means = similarity(name, trace, episodes, model, packet)
            style.extend(detail)
            style_summary.extend(means)
        print("AUDIT", bench, name, k, n, flush=True)
    aggregate = []
    for bench in BENCHMARKS:
        for method in ("ppo", "bc", "guided", "reference", "residual"):
            group = [s for s in summaries if s["benchmark"] == bench and s["method"] == method]
            rates = [s["success_rate"] for s in group]
            aggregate.append(dict(benchmark=bench, method=method, train_replicates=0 if method == "reference" else len(group),
                policies=len(group), episodes=sum(s["count"] for s in group), successes=sum(s["successes"] for s in group),
                success_mean=float(np.mean(rates)), success_sd=float(np.std(rates, ddof=1)) if len(rates)>1 else 0.,
                mean_hold=float(np.mean([s["mean_hold"] for s in group])),
                max_depth_mm=max(s["max_depth_mm"] for s in group),
                max_force_n=max(s["max_force_n"] for s in group), forbidden=sum(s["forbidden"] for s in group)))
    paired = []
    for bench in BENCHMARKS:
        for seed in SEEDS:
            for left, right in ((f"residual_s{seed}", "reference"), (f"guided_s{seed}", f"bc_s{seed}"),
                                (f"guided_s{seed}", f"ppo_s{seed}"), (f"residual_s{seed}", f"ppo_s{seed}")):
                a = [r for r in rows if r["benchmark"] == bench and r["run"] == left]
                b = [r for r in rows if r["benchmark"] == bench and r["run"] == right]
                assert [r["seed"] for r in a] == [r["seed"] for r in b]
                win = sum(x["success30"] and not y["success30"] for x, y in zip(a, b))
                loss = sum(y["success30"] and not x["success30"] for x, y in zip(a, b))
                paired.append(dict(benchmark=bench, left=left, right=right, train_seed=seed, count=len(a),
                                   wins=win, losses=loss, ties=len(a)-win-loss, net_percentage_points=100*(win-loss)/len(a)))
    write_csv(REPORT / "episodes.csv", rows)
    write_csv(REPORT / "policy_summary.csv", summaries)
    write_csv(REPORT / "aggregate.csv", aggregate)
    write_csv(REPORT / "paired_changes.csv", paired)
    write_csv(REPORT / "style_samples.csv", style)
    write_csv(REPORT / "style_by_episode.csv", style_summary)
    write_csv(REPORT / "residual_stats.csv", residual_stats)
    training_rows = []
    for method in METHODS:
        for seed in SEEDS:
            run = OUT / "train" / f"{method}_s{seed}"
            summary = json.loads((run / "summary.json").read_text())
            updates = [json.loads(line) for line in (run / "ppo_metrics.jsonl").read_text().splitlines()]
            success = summary["success30"]
            chosen = frozen["selected"][run.name]["development"]
            training_rows.append(dict(method=method, train_seed=seed,
                first_success=success["first"], last_success=success["last"], max_success=success["max"],
                first_max_step=success["best_step"], selected_step=chosen["step"],
                selected_success=chosen["success30"], first_kl=updates[0]["training/kl_mean"],
                max_kl=max(r["training/kl_mean"] for r in updates), last_kl=updates[-1]["training/kl_mean"]))
    write_csv(REPORT / "training_diagnosis.csv", training_rows)
    geometry = []
    for variant, directory in (("corrected", "inputs"), ("uncorrected", "inputs_uncorrected"), ("linear", "inputs_linear")):
        m = mujoco.MjModel.from_xml_path(str(OUT / directory / "model.xml"))
        d = mujoco.MjData(m)
        with np.load(OUT / directory / "reference.npz") as z:
            p = dict(z)
        qids, oqid, _ = model_indices(m)
        d.qpos[qids] = p["ctrl"][-1]
        d.qpos[oqid:oqid+7] = np.r_[p["object_position"], p["object_quaternion"]]
        d.mocap_pos[0] = [0, 0, -1]
        mujoco.mj_forward(m, d)
        rot = d.xmat[m.body("cube").id].reshape(3, 3)
        for j, (finger, name) in enumerate(zip(("thumb", "index", "middle"), PAD_NAMES)):
            gid = m.geom(name).id
            actual = (d.geom_xpos[gid]-d.xpos[m.body("cube").id]) @ rot
            inward = -actual.copy()
            inward[1] = 0
            inward /= max(np.linalg.norm(inward), 1e-9)
            normal = -d.xmat[m.geom_bodyid[gid]].reshape(3, 3)[:, 0] @ rot
            geometry.append(dict(reference=variant, finger=finger,
                static_target_error_mm=float(np.linalg.norm(actual-p["target_body_tips"][-1, j])*1000),
                static_facing_cos=float(normal @ inward)))
    write_csv(REPORT / "reference_geometry.csv", geometry)
    zero_checks = []
    reference_path = OUT / "eval/B0_nominal/reference/step_000000000/preview_trajectories.npz"
    with np.load(reference_path) as z:
        ref_trace = {k: z[k] for k in ("qpos", "ctrl", "hold", "active")}
    for seed in SEEDS:
        with np.load(OUT / f"eval/A_gain/residual_s{seed}_gain0/step_000000000/preview_trajectories.npz") as z:
            differences = {k: float(np.max(np.abs(z[k].astype(float)-v.astype(float)))) for k, v in ref_trace.items()}
            # A nonzero difference must be reported, not silently described as
            # an exact replay; floating-point execution paths can diverge.
            zero_checks.append(dict(train_seed=seed, differences=differences,
                                    exact=all(v == 0 for v in differences.values())))
    write_json(REPORT / "zero_gain_replay_checks.json", zero_checks)
    result = dict(status="verified", jobs=len(dirs), total_test_episodes=len(rows), recomputed_saved_episodes=verified,
        paired_initial_states=True, source_snapshots_verified=True, residual_control_max_error_rad=max_error,
        selection_sha256=completion["selection_sha256"],
        limitation="Only saved traces recomputed; robustness batches save first4, with all episode metrics/initial states retained. 3 training seeds, no real hardware or causal proof that video is indispensable.")
    write_json(REPORT / "verification.json", result)
    plot(aggregate, summaries, style_summary)
    print(json.dumps(result), flush=True)


def plot(aggregate, summaries, style):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    names = ("ppo", "bc", "guided", "reference", "residual")
    for i, method in enumerate(names):
        row = next(r for r in aggregate if r["benchmark"] == "B0_nominal" and r["method"] == method)
        axes[0].bar(i, row["success_mean"]*100, color=COLORS[method], alpha=.7)
        selected = [r for r in summaries if r["benchmark"] == "B0_nominal" and r["method"] == method]
        axes[0].scatter(np.linspace(i-.06, i+.06, len(selected)) if len(selected)>1 else [i],
                        [r["success_rate"]*100 for r in selected], color="black", zorder=4)
        label_y = max(s["success_rate"] for s in selected)*100+4
        axes[0].text(i, label_y, f"{row['success_mean']*100:.1f}%", ha="center")
    axes[0].set(xticks=range(len(names)), xticklabels=["Pure PPO", "BC only", "BC\n+ PPO", "Reference\nonly", "Reference\n+ residual PPO"],
                ylabel="Strict 30 s success (%)", ylim=(0, 110), title="Mean bars; 3 policy seeds, one reference")
    data = np.array([[next(r["success_mean"] for r in aggregate if r["benchmark"] == b and r["method"] == m)
                      for b in BENCHMARKS] for m in names])*100
    im = axes[1].imshow(data, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for i in range(len(names)):
        for j in range(6):
            axes[1].text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", color="white" if data[i,j]>65 else "black")
    axes[1].set(yticks=range(len(names)), yticklabels=[LABEL[m] for m in names], xticks=range(6),
                xticklabels=["Normal", "Position", "Yaw", "Friction", "Mass", "Push"], title="Robustness: same frozen weights (%)")
    fig.colorbar(im, ax=axes[1], shrink=.7)
    fig.savefig(REPORT / "success_and_robustness.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    selections = json.loads((OUT / "selection.json").read_text())["selected"]
    for ax, method in zip(axes, METHODS):
        for seed in SEEDS:
            with (OUT / "train" / f"{method}_s{seed}" / "metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            line, = ax.plot([float(r["step"]) for r in rows], [float(r["success30"])*100 for r in rows], marker=".", label=str(seed))
            chosen = selections[f"{method}_s{seed}"]["development"]
            ax.scatter([chosen["step"]], [chosen["success30"]*100], marker="D", s=45, color=line.get_color(), zorder=4)
        ax.set(title=LABEL[method], xlabel="Online transitions", ylim=(-3, 103),
               xticks=[0,40960,81920,122880,163840], xticklabels=["0","41k","82k","123k","164k"])
        ax.legend(title="Train seed")
    axes[0].set_ylabel("Development success (%)")
    fig.suptitle("8 fixed development states; diamonds mark pre-test checkpoint selection", fontsize=10)
    fig.savefig(REPORT / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    ablation = [r for r in summaries if r["benchmark"] == "A_gain"]
    for seed in SEEDS:
        values = [next(r["success_rate"] for r in ablation if r["run"] == f"residual_s{seed}_gain{g:g}") for g in (0., .5)]
        values += [next(r["success_rate"] for r in summaries if r["benchmark"] == "B0_nominal" and r["run"] == f"residual_s{seed}")]
        axes[0].plot([0, .5, 1], np.array(values)*100, "o-", label=str(seed))
    axes[0].set(xlabel="Inference-only residual gain", ylabel="Strict success (%)", ylim=(-3, 103), xticks=[0,.5,1], title="Turn off / halve correction (not retraining)")
    axes[0].legend()
    ref = [r for r in summaries if r["benchmark"] == "A_reference"]
    ref += [r for r in summaries if r["benchmark"] == "B0_nominal" and r["run"] == "reference"]
    axes[1].bar([r["run"] for r in ref], [r["success_rate"]*100 for r in ref], color=["#8b9198", "#b66a3c", "#258b78"])
    axes[1].set(ylabel="Strict success (%)", ylim=(0, 103), title="Reference ablation, identical cylinder physics")
    fig.savefig(REPORT / "ablation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax, phase in zip(axes, ("holding", "valid_holding")):
        for i, method in enumerate(names):
            group = [r for r in style if r["run"].split("_s")[0] == method and r["phase"] == phase]
            y = [np.mean([r[f+"_error_mm"] for r in group]) if group else np.nan for f in ("thumb", "index", "middle")]
            ax.plot(range(3), y, "o-", label=LABEL[method], color=COLORS[method])
        ax.set(xticks=range(3), xticklabels=["Thumb", "Index", "Middle"], ylabel="Adapted source-target error (mm)",
               title="All active holding samples" if phase == "holding" else "Only valid holding (selection bias)")
    axes[1].legend(fontsize=8)
    fig.savefig(REPORT / "source_similarity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
