"""Frozen S08 wrap benchmark, using S07's budgets and selection/evaluation flow."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import run_s07_study as shared
from prepare_cylinder_s02_source import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/orca_benchmark_suite/S08"
DOC=ROOT/"docs/experiments/S08"
SOURCE=OUT.parent/"S08_wrap_position"
SEEDS=(8501,8502,8503)
DATASET=OUT/"demonstrations/trajectories.npz"


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/"inputs_manifest.json").exists():
        for name,checksum in json.loads((OUT/"inputs_manifest.json").read_text()).items():
            assert digest(ROOT/name)==checksum,name
        return
    target=OUT/"inputs"
    target.mkdir(exist_ok=False)
    shutil.copyfile(SOURCE/"model.xml",target/"model.xml")
    with np.load(SOURCE/"reference.npz") as z:
        packet=dict(z)
    # Metadata expected by the existing matched environment; no spatial rescaling.
    packet["spatial_scale"]=np.array(1.)
    packet["object_to_world_rotation"]=Rotation.from_quat(packet["object_quaternion"][[1,2,3,0]]).as_matrix()
    np.savez_compressed(target/"reference.npz",**packet)
    for variant in ("linear","uncorrected"):
        folder=OUT/("inputs_"+variant)
        folder.mkdir()
        shutil.copyfile(target/"model.xml",folder/"model.xml")
        alternate={k:v.copy() for k,v in packet.items()}
        if variant=="linear":
            # Fast endpoint approach contrasts with P03's 1.5s linear closure.
            delta=packet["ctrl"][-1]-packet["initial_ctrl"]
            allowance=(packet["time"]/.05)[:,None]*.1
            alternate["ctrl"]=packet["initial_ctrl"]+np.clip(delta,-allowance,allowance)
        else:
            with np.load(OUT.parent/"S08_wrap_pilot/longitudinal/reference.npz") as z:
                alternate["ctrl"]=z["ctrl"].copy()
        np.savez_compressed(folder/"reference.npz",**alternate)
    files=[*OUT.glob("inputs*/*.xml"),*OUT.glob("inputs*/*.npz"),DOC/"包覆方法对比约定.md"]
    shared.write_json(OUT/"inputs_manifest.json",{str(p.relative_to(ROOT)):digest(p) for p in files})
    shared.write_json(OUT/"protocol.json",dict(training_seeds=SEEDS,methods=shared.METHODS,
        steps=163840,evals=9,dev_seeds=list(range(85000,85008)),
        demonstration_seeds=list(range(86000,86032)),test_seeds=list(range(87000,87064)),
        learning_rate=1e-4,bc_updates=20000,residual_bound=.08,
        benchmarks=list(shared.BENCHMARKS),source="P03 geometry reference, not video",
        input_manifest_sha256=digest(OUT/"inputs_manifest.json")))


def launch(directory,arguments):
    if (directory/"summary.json").exists():
        assert json.loads((directory/"summary.json").read_text())["status"]=="completed"
        return
    if directory.exists():
        raise ValueError("Partial output requires diagnosis: "+str(directory))
    directory.parent.mkdir(exist_ok=True,parents=True)
    cmd=[sys.executable,str(ROOT/"scripts/run_s08_job.py"),"--output",str(directory),
        "--height-weight","5",*map(str,arguments)]
    print("START",directory,flush=True)
    with (directory.parent/(directory.name+".log")).open("w") as f:
        f.write(json.dumps(cmd)+"\n"); f.flush()
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print("DONE",directory,flush=True)


def train_all(methods,seeds):
    for method in methods:
        for seed in seeds:
            directory=OUT/"train"/f"{method}_s{seed}"
            args=["--input",OUT/"inputs","--method",method,"--seed",seed,
                "--steps",163840,"--evals",9,"--eval-count",8,"--eval-start",85000,
                "--learning-rate",1e-4,"--std",.1]
            if method=="guided":
                args += ["--dataset",DATASET,"--bc-updates",20000]
            launch(directory,args)
            if not (directory/"iteration_audit.json").exists():
                shared.record_training(directory)


def evaluate_all(workers):
    # Same per-condition counts and gain ablations as S07, new test seed split.
    original_launch=launch
    def new_test_launch(directory,args):
        args=list(args)
        args[args.index("--eval-start")+1]=87000
        original_launch(directory,args)
    shared.launch=new_test_launch
    shared.evaluate_all(workers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","collect","smoke","train","freeze","evaluate"))
    p.add_argument("--methods",nargs="+",choices=shared.METHODS,default=list(shared.METHODS))
    p.add_argument("--seeds",nargs="+",type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument("--workers",type=int,choices=(1,2,3),default=1)
    a=p.parse_args()
    shared.OUT,shared.DOC,shared.SEEDS=OUT,DOC,SEEDS
    shared.launch=launch
    prepare()
    if a.phase in ("collect","smoke"):
        launch(OUT/("demonstrations" if a.phase=="collect" else "smoke_reference"),
            ["--input",OUT/"inputs","--method","collect" if a.phase=="collect" else "reference",
             "--steps",0,"--eval-count",32 if a.phase=="collect" else 8,
             "--eval-start",86000 if a.phase=="collect" else 85000,"--save-all-trajectories"])
    elif a.phase=="train":
        train_all(a.methods,a.seeds)
    elif a.phase=="freeze":
        shared.freeze()
    elif a.phase=="evaluate":
        evaluate_all(a.workers)


if __name__=="__main__":
    main()
