"""Record local dependency and hardware provenance without exposing credentials."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(argv):
    result = subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,check=False)
    return dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    versions = {}
    for package in ("mujoco","jax","jaxlib","brax","flax","optax","numpy","scipy",
                    "ml-collections","mediapy","Pillow","matplotlib"):
        versions[package] = metadata.version(package)
    sources = [ROOT/"Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src"/name
               for name in ("wrapper.py","mjx_env.py")]
    data = dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),logical_cpu_count=os.cpu_count(),packages=versions,
        gpu=command(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]),
        git_head=command(["git","rev-parse","HEAD"]),submodules=command(["git","submodule","status"]),
        local_framework={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        invocation_overrides=dict(OPENBLAS_NUM_THREADS="1",XLA_PYTHON_CLIENT_PREALLOCATE="false",
            jax_default_matmul_precision="highest",jax_pmap_shmap_merge=False))
    with a.output.open("x") as f:
        json.dump(data,f,indent=2);f.write("\n")
    print(a.output)


if __name__=="__main__":
    main()
