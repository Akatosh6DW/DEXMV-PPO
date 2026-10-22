#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEXMV_PY="${DEXMV_PY:-/home/ll/miniconda3/envs/dexmv/bin/python}"
MUJOCO_DIR="${MUJOCO_DIR:-/home/ll/.mujoco/mujoco210}"
POLICY_PATH="${1:-$ROOT_DIR/DexMV/examples/training_log/rl_relocate-mug-0.8_pure_trpo_short_seed200/iterations/best_policy.pickle}"
STEPS="${2:-500}"

env -u PYTHONPATH \
  DISPLAY="${DISPLAY:-:0}" \
  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  __VK_LAYER_NV_optimus=NVIDIA_only \
  PATH="/home/ll/miniconda3/envs/dexmv/bin:$PATH" \
  LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so" \
  LD_LIBRARY_PATH="$MUJOCO_DIR/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}" \
  MUJOCO_PY_MUJOCO_PATH="$MUJOCO_DIR" \
  MUJOCO_GL=glfw \
  DEXMV_POLICY="$POLICY_PATH" \
  DEXMV_STEPS="$STEPS" \
  "$DEXMV_PY" - <<'PY'
import os
import pickle
import time

from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

policy_path = os.environ["DEXMV_POLICY"]
steps = int(os.environ["DEXMV_STEPS"])

policy = pickle.load(open(policy_path, "rb"))
env = YCBRelocate(
    has_renderer=True,
    object_name="mug",
    friction=(1, 0.5, 0.01),
    object_scale=0.8,
)
obs = env.reset()
ret = 0.0
print(f"viewer started: {policy_path}")
for _ in range(steps):
    action = policy.get_action(obs)[1]["evaluation"]
    obs, reward, done, _ = env.step(action)
    ret += float(reward)
    env.render()
    time.sleep(0.02)
    if done:
        break
print(f"episode_return {ret:.4f}")
PY
