"""Fixed-policy Monte Carlo critic-gradient diagnostic, not PPO gradients."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
from jax.flatten_util import ravel_pytree
import numpy as np

from analyze_s09_bc_confirmation import write_csv
from retarget_s09_reference import digest
import run_s09_learned_base_residual as learned
from run_s09_learned_residual_experts import residual_expert_factory


def cosine_matrix(gradients):
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or not np.isfinite(gradients).all():
        raise ValueError("Require a finite matrix of gradients")
    norm = np.linalg.norm(gradients, axis=1)
    denominator = norm[:, None]*norm[None, :]
    return np.divide(gradients@gradients.T, denominator,
                     out=np.full_like(denominator, np.nan), where=denominator>0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    source = read(args.diagnostic/"summary.json")
    assert source["status"] == "completed" and source["policy_updates"] == source["critic_updates"] == 0
    data_path = args.diagnostic/"critic_calibration.npz"
    assert digest(data_path) == source["data_sha256"]
    run = Path(source["source_run"])
    checkpoint = run/"step_000000000/params.pkl"
    assert digest(checkpoint) == source["checkpoint_sha256"]
    cfg, protocol = read(run/"run_config.json"), read(run/"learned_base_config.json")
    base_path, model = Path(protocol["base_checkpoint"]), Path(cfg["input"])/"model.xml"
    assert digest(base_path) == source["base_sha256"] and digest(model) == source["model_sha256"]
    base, _, _ = learned.load_base(base_path, model)
    net = residual_expert_factory(base)(dict(state=(123,),privileged_state=(140,)),17)
    params = jax.tree_util.tree_map(jp.asarray,pickle.loads(checkpoint.read_bytes()))
    with np.load(data_path,allow_pickle=False) as z:
        obs, target, active, mode = z["privileged_obs"],z["returns"],z["active"],z["mode"]
    assert obs.shape[:2] == target.shape == active.shape
    assert np.isfinite(obs).all() and np.isfinite(target).all()
    jax.config.update("jax_default_matmul_precision","highest")
    observations = dict(state=jp.asarray(obs[...,:123]),privileged_state=jp.asarray(obs))
    targets = jp.asarray(target)

    def loss(value,mask):
        prediction = net.value_network.apply(params[0],value,observations)
        return .5*jp.sum(jp.where(mask,(prediction-targets)**2,0))/jp.sum(mask)

    value_grad = jax.jit(jax.value_and_grad(loss))
    frames = np.arange(target.shape[1])[None,:]
    rows, gradients, masks = [], [], []
    for phase,times in (("all",frames>=0),("acquisition",frames<60),
                        ("released_before_30s",(frames>=60)&(frames<600))):
        for group in range(-1,6):
            cases = np.ones(len(mode),bool) if group == -1 else mode == group
            mask = active & cases[:,None] & times
            if not mask.any():
                continue
            value,grad = value_grad(params[2],jp.asarray(mask))
            vector = np.asarray(ravel_pytree(grad)[0],dtype=np.float64)
            assert np.isfinite(vector).all()
            rows.append(dict(group=len(rows),phase=phase,mode=group,n=int(mask.sum()),
                             episodes=int(cases.sum()),half_mse=float(value),gradient_norm=float(np.linalg.norm(vector))))
            gradients.append(vector)
            masks.append(mask)
    gradients = np.stack(gradients)
    cos = cosine_matrix(gradients)
    pairs = []
    for i,a in enumerate(rows):
        for j,b in enumerate(rows):
            if j<=i:
                continue
            pairs.append(dict(a_phase=a["phase"],a_mode=a["mode"],b_phase=b["phase"],b_mode=b["mode"],
                              cosine=float(cos[i,j]) if np.isfinite(cos[i,j]) else None,
                              dot=float(gradients[i]@gradients[j])))
    # Compare every subgroup to the actual frame-weighted, all-data MC direction.
    global_index = next(r["group"] for r in rows if r["phase"] == "all" and r["mode"] == -1)
    for row in rows:
        row["dot_with_global_gradient"] = float(gradients[row["group"]]@gradients[global_index])
        row["first_order_loss_increases_under_global_sgd"] = row["dot_with_global_gradient"]<0
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/"groups.csv",rows)
    write_csv(args.output/"pairs.csv",pairs)
    np.savez_compressed(args.output/"gradients.npz",gradients=gradients,cosine=cos)
    result = dict(status="completed_diagnostic_only",groups=rows,checkpoint_sha256=digest(checkpoint),
                  data_sha256=digest(data_path),source_sha256=digest(Path(__file__)),
                  extra_environment_interactions=0,policy_updates=0,critic_updates=0,
                  target="0.5 mean squared error to existing finite-horizon Monte Carlo return",
                  limitation="Not actual PPO/GAE gradients, Adam direction, later-policy calibration or causal success proof",
                  gradients_sha256=digest(args.output/"gradients.npz"))
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# 固定初始critic的梯度冲突诊断", "",
        "使用D34的32个随机回合，固定原零残差策略和critic，不新增物理交互或更新参数。",
        "负余弦表示两个分组的MC拟合梯度相反；这只对应局部一阶普通梯度下降，不能直接推断Adam/PPO的实际行为。", "",
        "| 阶段 | 朝向(-1为全部) | 帧数 | 梯度范数 | 与全数据梯度点积 | 全局SGD一阶损害此组 |",
        "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['mode']} | {r['n']} | {r['gradient_norm']:.3f} | "
                     f"{r['dot_with_global_gradient']:.3f} | {r['first_order_loss_increases_under_global_sgd']} |")
    lines += ["", "完整成对余弦见pairs.csv，梯度数组和绑定哈希另存。",
              "帧间相关，只有32条随机回报；有限40秒回报不是PPO截段bootstrap标签。负点积不等于抓取退化的已证实原因。"]
    (args.output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
