"""Preserve terminal transition bookkeeping across full random resets.

Physics, reward, done, reset randomness, and evaluation are unchanged. Only
training's truncation and episode-logging fields survive the autoreset boundary.
"""

import json
from pathlib import Path
import sys

import run_s09_constrained_ppo as constrained
from mujoco_playground._src import wrapper
from retarget_s09_reference import digest


PRESERVE = "AutoResetWrapper_preserve_info"
FIELDS = ("truncation", "episode_done", "episode_metrics")
SLOT = "s09_terminal_transition"


class CaptureTransition(wrapper.Wrapper):
    def capture(self, state):
        saved = dict(state.info.get(PRESERVE, {}))
        saved[SLOT] = {key:state.info[key] for key in FIELDS}
        return state.replace(info=dict(state.info, **{PRESERVE:saved}))

    def reset(self, rng):
        return self.capture(self.env.reset(rng))

    def step(self, state, action):
        return self.capture(self.env.step(state, action))


class RestoreTransition(wrapper.Wrapper):
    def step(self, state, action):
        nxt = self.env.step(state, action)
        return nxt.replace(info=dict(nxt.info, **nxt.info[PRESERVE][SLOT]))


def transition_wrap(env, **kwargs):
    if not kwargs.pop("full_reset", True):
        raise ValueError("This experiment requires full random resets")
    auto = wrapper.wrap_for_brax_training(env, full_reset=True, **kwargs)
    if not isinstance(auto, wrapper.BraxAutoResetWrapper):
        raise TypeError("Unexpected training wrapper stack")
    auto.env = CaptureTransition(auto.env)
    return RestoreTransition(auto)


def main():
    runner = constrained.improvement.runner
    original = runner.train.train
    output = Path(sys.argv[sys.argv.index("--output")+1])

    def train(*args, **kwargs):
        kwargs["wrap_env_fn"] = transition_wrap
        snapshot = output/"sources"/Path(__file__).name
        metadata = dict(preserved_fields=FIELDS, reward_changed=False, physics_changed=False,
                        evaluation_changed=False, source_sha256=digest(snapshot),
                        upstream_wrapper_sha256=digest(Path(wrapper.__file__)))
        (output/"transition_config.json").write_text(json.dumps(metadata,indent=2)+"\n")
        return original(*args, **kwargs)

    runner.train.train = train
    try:
        constrained.main()
    finally:
        runner.train.train = original


if __name__ == "__main__":
    main()
