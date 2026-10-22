"""Frozen four-arm S09 study orchestration; resume only verified terminal jobs."""

import argparse
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from retarget_s09_reference import digest
from s09_suspension import OUT, ROOT, SOURCE

SEEDS = (9101,9102,9103)
METHODS = ("ppo","guided","residual")
STRATA = {"uniform":(93000,128), "web":(94000,32), "x":(95000,32),
          "y":(96000,32), "z":(97000,32)}
STEPS = 163840
SOURCES = ("s09_suspension.py","orca_s09_env.py","run_s09_job.py","run_s09_study.py",
           "orca_native_physics.py","orca_contact_forces.py","orca_matched_env.py",
           "train_orca_comparison.py","train_orca_matched.py")


def freeze():
    path=OUT/"protocol.json"
    if path.exists():
        raise FileExistsError("Never overwrite an existing frozen protocol")
    inputs=OUT/"inputs"
    dataset=OUT/"demonstrations/trajectories.npz"
    collection=json.loads((OUT/"demonstrations/summary.json").read_text())
    assert collection["status"]=="completed"
    manifest=json.loads((dataset.parent/"dataset_manifest.json").read_text())
    assert manifest["dataset_sha256"]==digest(dataset) and len(manifest["seeds"])>=4
    assets={}
    for mesh in ET.parse(inputs/"model.xml").getroot().findall("asset/mesh"):
        file=Path(mesh.get("file"))
        if not file.is_absolute():
            file=inputs/file
        assets[str(file.resolve())]=digest(file)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in
        [inputs/"model.xml",inputs/"reference.npz",dataset,dataset.parent/"dataset_manifest.json",SOURCE,
         *[ROOT/"scripts"/s for s in SOURCES]]}
    protocol=dict(version="S09-R0",created=time.strftime("%Y-%m-%d %H:%M:%S"),
        methods=[*METHODS,"reference"],train_seeds=SEEDS,steps=STEPS,
        development_seeds=list(range(92000,92008)),test_strata=STRATA,
        reward=dict(stable=15,drop=-50,coverage=2,approach_before_release=2,dt=.05),
        recipes={"ppo":dict(learning_rate=3e-4,std=.3),
                 "guided":dict(learning_rate=3e-5,std=.3,bc_updates=4000),
                 "residual":dict(learning_rate=3e-4,std=.3,residual_bound=.08)},
        sample_split="Development/reference selection and 90500:90627 collection excluded from final test seeds",
        test_checkpoint="Final step163840, never selected by test success",
        reset="Uniform SO(3), +/-2mm XYZ, +/-5mm/s horizontal velocity; no success filtering",
        reference="Six source-derived orientation templates incl failures; closest undirected cylinder axis",
        criterion="40s episode, release3s,30s strict continuous palmar opposing hold and valid at end; initial<=1mm, released<=3mm, no forbidden support, zero released rope force",
        dataset_successes=len(manifest["seeds"]),dataset_attempts=len(manifest["attempted_seeds"]),
        source_hashes=hashes,asset_hashes=assets,
        software_versions={p:importlib.metadata.version(p) for p in ("mujoco","jax","jaxlib","brax","flax","numpy","scipy")},
        reward_iteration="Separate post-comparison branch, development selection and NEW final test seeds")
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps(protocol),flush=True)


def check_protocol():
    protocol=json.loads((OUT/"protocol.json").read_text())
    for file,expected in protocol["source_hashes"].items():
        assert digest(ROOT/file)==expected,"Frozen source changed: "+file
    for file,expected in protocol["asset_hashes"].items():
        assert digest(Path(file))==expected,"Frozen model asset changed: "+file
    return protocol


def jobs(phase,protocol):
    base=[sys.executable,str(ROOT/"scripts/run_s09_job.py"),"--input",str(OUT/"inputs")]
    if phase=="train":
        for seed in SEEDS:
            for method in METHODS:
                output=OUT/"train"/f"{method}_s{seed}"
                recipe=protocol["recipes"][method]
                command=base+["--output",str(output),"--method",method,"--seed",str(seed),
                    "--steps",str(STEPS),"--evals","5","--eval-count","8","--eval-start","92000",
                    "--learning-rate",str(recipe["learning_rate"]),"--std",str(recipe["std"])]
                if method=="guided":
                    command += ["--dataset",str(OUT/"demonstrations/trajectories.npz"),"--bc-updates","4000"]
                yield output,command,STEPS
    else:
        for orientation,(start,count) in STRATA.items():
            for method in (*METHODS,"reference"):
                for seed in ((0,) if method=="reference" else SEEDS):
                    output=OUT/"eval"/orientation/("reference" if method=="reference" else f"{method}_s{seed}")
                    command=base+["--output",str(output),"--method",method,"--steps","0",
                        "--orientation",orientation,"--eval-count",str(count),"--eval-start",str(start)]
                    if method!="reference":
                        run=OUT/"train"/f"{method}_s{seed}"
                        summary=json.loads((run/"summary.json").read_text())
                        assert summary["status"]=="completed" and summary["final_step"]==STEPS
                        command += ["--checkpoint",str(run/f"step_{STEPS:09d}/params.pkl"),"--seed",str(seed)]
                    yield output,command,0


def run(phase,limit):
    protocol=check_protocol()
    logs=OUT/"execution"
    logs.mkdir(exist_ok=True)
    count=0
    for output,command,expected_steps in jobs(phase,protocol):
        if output.exists():
            if not (output/"summary.json").exists():
                raise RuntimeError("Partial output needs inspection, not overwrite/restart: "+str(output))
            summary=json.loads((output/"summary.json").read_text())
            config=json.loads((output/"run_config.json").read_text())
            assert summary["status"]=="completed" and summary["final_step"]==expected_steps
            for source in ("model","reference"):
                path=OUT/"inputs"/("model.xml" if source=="model" else "reference.npz")
                assert config[source+"_sha256"]==digest(path)
            continue
        if shutil.disk_usage(ROOT).free < 700*1024**2:
            raise RuntimeError("Less than 700MiB reserve; preserve runs rather than delete checkpoints automatically")
        check_protocol()
        log=logs/("_".join(output.relative_to(OUT).parts)+".log")
        environment=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                         PYTHONPATH=str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))
        with log.open("w") as handle:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT)
            (logs/"current.json").write_text(json.dumps(dict(pid=process.pid,command=command,log=str(log)),indent=2)+"\n")
            print("START",process.pid,output,flush=True)
            code=process.wait()
        if code:
            raise RuntimeError(f"Job failed ({code}); inspect {log}")
        summary=json.loads((output/"summary.json").read_text())
        assert summary["final_step"]==expected_steps
        print("DONE",output,json.dumps(summary["success30"]),flush=True)
        count+=1
        if limit and count>=limit:
            break
    (logs/"current.json").write_text(json.dumps(dict(status="terminal",phase=phase,new_jobs=count))+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("freeze","train","eval"))
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    if a.phase=="freeze":
        freeze()
    else:
        run(a.phase,a.limit)


if __name__=="__main__":
    main()
