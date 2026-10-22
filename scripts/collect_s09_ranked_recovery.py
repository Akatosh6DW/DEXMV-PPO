"""Append verified long-ranked training recoveries without changing old data."""

import argparse
import json
from pathlib import Path

import numpy as np

from collect_s09_recovery_demo import merge_training
from retarget_s09_reference import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    read=lambda path:json.loads(path.read_text())
    summary=read(a.teacher/"summary.json")
    proof=read(a.teacher/"report/evidence.json")
    protocol=read(a.teacher/"protocol.json")
    meta=read(a.dataset.parent/"dataset_manifest.json")
    assert summary["status"]=="completed" and not summary["smoke"]
    assert proof["status"]=="verified" and proof["summary_sha256"]==digest(a.teacher/"summary.json")
    assert proof["results_sha256"]==digest(a.teacher/"results.json")
    assert meta["model_sha256"]==protocol["model_sha256"]
    assert digest(a.dataset)==meta["dataset_sha256"]
    split=meta["train_episodes"]
    accepted,added,records=[],[],[]
    for r in read(a.teacher/"results.json"):
        c=r["canonical"]
        if not c or not c["qualified"]:
            continue
        assert r["seed"] in meta["seeds"][:split] and r["seed"] not in meta["seeds"][split:]
        assert not r["short_success"],"Only add new recoveries, not duplicate successful old labels"
        path=a.teacher/f"seed{r['seed']}"/"canonical_trace.npz"
        assert digest(path)==c["trace_sha256"]==proof["trace_hashes"][str(path.relative_to(a.teacher))]
        assert c["native_qpos_error"]<3e-6 and c["native_qvel_error"]<1e-4
        with np.load(path,allow_pickle=False) as z:
            assert z["obs"].shape==(1,800,123) and z["action"].shape==(1,800,17)
            assert bool(z["success30"][0,-1])
            sample={k:z[k].copy() for k in ("obs","action","active")}
        assert all(np.isfinite(v).all() for v in sample.values())
        sample["active"][:,:40]=False
        added.append(sample)
        accepted.append(r["seed"])
        records.append(dict(seed=r["seed"],trace_sha256=digest(path),selected_candidate=r["long_index"],
            hold_sec=c["hold_sec"],student_prefix_steps_excluded=40))
    assert added,"No new qualified recovery labels; do not create an empty experiment"
    with np.load(a.dataset,allow_pickle=False) as z:
        old={k:z[k] for k in ("obs","action","active")}
    new={k:np.concatenate([r[k] for r in added]) for k in old}
    merged=merge_training(old,new,split)
    for k in old:
        np.testing.assert_array_equal(merged[k][:split],old[k][:split])
        np.testing.assert_array_equal(merged[k][split+len(added):],old[k][split:])
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/"trajectories.npz"
    np.savez_compressed(path,**merged)
    meta.update(train_episodes=split+len(added),dataset_sha256=digest(path),parent_dataset_sha256=digest(a.dataset),
        seeds=meta["seeds"][:split]+accepted+meta["seeds"][split:],original_validation_preserved=True,
        additional_simulated_recovery_episodes=meta["additional_simulated_recovery_episodes"]+len(added),
        student_correction_episodes=meta["student_correction_episodes"]+len(added),
        ranked_teacher_protocol_sha256=digest(a.teacher/"protocol.json"),
        ranked_teacher_results_sha256=digest(a.teacher/"results.json"),
        ranked_teacher_summary_sha256=digest(a.teacher/"summary.json"),new_human_videos=0)
    (a.output/"dataset_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    record=dict(status="completed",accepted=records,original_training_preserved=True,
        original_validation_preserved=True,original_train_episodes=split,new_train_episodes=split+len(added),
        validation_episodes=len(old["obs"])-split,source_sha256=digest(Path(__file__)),policy_updates=0,
        dataset_sha256=digest(path),teacher_costs=summary["costs"],
        warning="Additional simulation recovery labels, not new videos or learned-policy success")
    (a.output/"summary.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    main()
