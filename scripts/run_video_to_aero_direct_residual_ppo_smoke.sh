#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AERO_PY="${AERO_PY:-/home/ll/miniconda3/envs/aero_rl/bin/python}"

DEMO_DATASET="${DEMO_DATASET:-$ROOT_DIR/artifacts/ppo_bridge/demo_dataset_aero_direct.npz}"
HAND_POSE_DIR="${HAND_POSE_DIR:-$ROOT_DIR/DexMV/examples/retargeting_source/relocate_mustard_example_seq/hand_pose}"
OBJECT_POSE_DIR="${OBJECT_POSE_DIR:-$ROOT_DIR/DexMV/examples/retargeting_source/relocate_mustard_example_seq/object_pose}"
ENV_NAME="AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "$ROOT_DIR/artifacts/ppo_bridge"

echo "[1/3] DexMV video hand pose -> Aero-Hand V2 prior controls"
(
  cd "$ROOT_DIR"
  "$AERO_PY" scripts/retarget_dexmv_video_to_aero_prior.py \
    --hand-dir "$HAND_POSE_DIR" \
    --object-pose-dir "$OBJECT_POSE_DIR" \
    --output "$DEMO_DATASET"
)

echo "[2/3] Verify Aero direct residual-prior env reset/step"
(
  cd "$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground"
  env -u PYTHONPATH "$AERO_PY" - <<'PY'
import jax
from mujoco_playground._src import registry

env = registry.load("AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect")
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, jax.numpy.zeros(env.mjx_model.nu))
print("env_step_ok", "prior_len", env._residual_prior_len, "reward", float(state.reward))
PY
)

echo "[3/3] Train tiny Aero-direct Residual PPO smoke checkpoint"
(
  cd "$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground"
  env PYTHONPATH="$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground" "$AERO_PY" learning/train_jax_ppo.py \
    --env_name="$ENV_NAME" \
    --suffix=dexmv_aero_direct_residual_smoke \
    --num_timesteps=2048 \
    --num_evals=1 \
    --num_envs=32 \
    --num_eval_envs=4 \
    --episode_length=64 \
    --unroll_length=8 \
    --num_minibatches=4 \
    --num_updates_per_batch=1 \
    --batch_size=64 \
    --learning_rate=3e-4 \
    --entropy_cost=1e-2 \
    --policy_hidden_layer_sizes=64,64 \
    --value_hidden_layer_sizes=64,64 \
    --num_videos=0 \
    --use_wandb=false
)

LATEST_LOG="$(find "$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground/logs" -maxdepth 1 -type d -name "${ENV_NAME}-*dexmv_aero_direct_residual_smoke" | sort | tail -n 1)"
echo "latest_log=$LATEST_LOG"
if [[ -n "$LATEST_LOG" ]]; then
  find "$LATEST_LOG/checkpoints" -maxdepth 1 -type d -regex '.*/[0-9]+' | sort | tail -n 1 | sed 's#^#latest_checkpoint=#'
fi
