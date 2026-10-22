#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AERO_RL_PY="${AERO_RL_PY:-/home/ll/miniconda3/envs/aero_rl/bin/python}"
PLAYGROUND_DIR="${ROOT_DIR}/Aero-Hand/sim_rl/mujoco_playground"

cd "${ROOT_DIR}"
"${AERO_RL_PY}" scripts/retarget_dexmv_video_to_orca_prior.py

cd "${PLAYGROUND_DIR}"
PYTHONPATH="${PLAYGROUND_DIR}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"${AERO_RL_PY}" learning/train_jax_ppo.py \
  --env_name=OrcaHandDexMVResidualPoseTrack \
  --suffix=orca_dexmv_residual_smoke \
  --num_timesteps=2048 \
  --num_evals=2 \
  --num_envs=8 \
  --num_eval_envs=4 \
  --episode_length=80 \
  --unroll_length=10 \
  --num_minibatches=2 \
  --num_updates_per_batch=1 \
  --batch_size=32 \
  --policy_hidden_layer_sizes=64,64 \
  --value_hidden_layer_sizes=64,64 \
  --num_videos=0 \
  --use_wandb=false

