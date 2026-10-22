#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEXMV_PY="${DEXMV_PY:-/home/ll/miniconda3/envs/dexmv/bin/python}"
AERO_PY="${AERO_PY:-/home/ll/miniconda3/envs/aero_rl/bin/python}"

RETARGET_OUT="${RETARGET_OUT:-$ROOT_DIR/artifacts/dexmv_retarget/example_retargeting.pkl}"
RETARGET_NPY="${RETARGET_NPY:-$ROOT_DIR/artifacts/dexmv_retarget/example_retargeting.npy}"
DEMO_DATASET="${DEMO_DATASET:-$ROOT_DIR/artifacts/ppo_bridge/demo_dataset.npz}"
HAND_POSE_DIR="${HAND_POSE_DIR:-$ROOT_DIR/DexMV/examples/retargeting_source/relocate_mustard_example_seq/hand_pose}"
OBJECT_POSE_DIR="${OBJECT_POSE_DIR:-$ROOT_DIR/DexMV/examples/retargeting_source/relocate_mustard_example_seq/object_pose}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-/home/ll/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="/home/ll/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export PATH="/home/ll/miniconda3/envs/dexmv/bin:$PATH"

mkdir -p "$ROOT_DIR/artifacts/dexmv_retarget" "$ROOT_DIR/artifacts/ppo_bridge"

echo "[1/4] DexMV retarget video-derived hand poses -> Adroit qpos"
(
  cd "$ROOT_DIR/DexMV/examples"
  env -u PYTHONPATH "$DEXMV_PY" retarget_human_hand.py \
    --hand_dir="$HAND_POSE_DIR" \
    --output_file="$RETARGET_OUT"
)

echo "[2/4] Bridge Adroit qpos -> Aero-Hand demo_dataset.npz"
(
  cd "$ROOT_DIR"
  "$AERO_PY" scripts/bridge_dexmv_to_aero_demo.py \
    --retarget-npy "$RETARGET_NPY" \
    --object-pose-dir "$OBJECT_POSE_DIR" \
    --output "$DEMO_DATASET"
)

echo "[3/4] Verify Aero residual-prior env reset/step"
(
  cd "$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground"
  env -u PYTHONPATH "$AERO_PY" - <<'PY'
import jax
from mujoco_playground._src import registry

env = registry.load("AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior")
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, jax.numpy.zeros(env.mjx_model.nu))
print("env_step_ok", "prior_len", env._residual_prior_len, "reward", float(state.reward))
PY
)

echo "[4/4] Train tiny Residual PPO smoke checkpoint"
(
  cd "$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground"
  env PYTHONPATH="$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground" "$AERO_PY" learning/train_jax_ppo.py \
    --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior \
    --suffix=dexmv_residual_smoke \
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

LATEST_LOG="$(find "$ROOT_DIR/Aero-Hand/sim_rl/mujoco_playground/logs" -maxdepth 1 -type d -name 'AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior-*dexmv_residual_smoke' | sort | tail -n 1)"
echo "latest_log=$LATEST_LOG"
if [[ -n "$LATEST_LOG" ]]; then
  find "$LATEST_LOG/checkpoints" -maxdepth 1 -type d -regex '.*/[0-9]+' | sort | tail -n 1 | sed 's#^#latest_checkpoint=#'
fi
