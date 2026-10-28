#!/usr/bin/env bash
# Run in an activated Python 3.11 environment; never replace an existing env.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Activate a Python 3.11 environment"'
test -f Aero-Hand/sim_rl/mujoco_playground/pyproject.toml || {
  printf '%s\n' 'Run git submodule update --init --recursive first.' >&2
  exit 1
}
python -m pip install -r environment/requirements-aero.txt -c environment/constraints-aero-linux-py311.txt
python -m pip install --no-deps -e Aero-Hand/sim_rl/mujoco_playground
python -m pip check
python scripts/restore_developer_release.py
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m unittest discover -s scripts -p 'test_developer_release.py'
printf '%s\n' 'CPU setup complete. See docs/development/README.md for GPU and continuation commands.'
