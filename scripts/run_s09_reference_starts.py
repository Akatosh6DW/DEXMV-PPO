"""Train full-target BC+PPO with a mixture of normal and demonstration resets.

Only the training environment is wrapped. Evaluation retains the original open
hand, continuous random pose, release timing and 30-second success criterion.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import numpy as np

import run_s09_constrained_ppo as constrained
from mujoco_playground._src import wrapper
from retarget_s09_reference import digest


class ReferenceStarts(wrapper.Wrapper):
    def __init__(self, env, bank, probability):
        super().__init__(env)
        if not 0 <= probability <= 1:
            raise ValueError("Reset probability must be in [0,1]")
        self.bank = jax.tree_util.tree_map(jp.asarray, bank)
        self.probability = float(probability)
        self.count = len(bank.reward)
        if not self.count or bank.obs["state"].shape != (self.count,123):
            raise ValueError("Expected full S09 reset states")
        for name in ("hold_steps","best_steps"):
            np.testing.assert_array_equal(bank.info[name],0)
        if not np.isin(bank.info["step"],(40,80)).all():
            raise ValueError("Unapproved initial phase")

    def reset(self, rng):
        selection, choice, touch = jax.random.split(jax.random.fold_in(rng, 240101),3)
        index = jax.random.randint(choice,(),0,self.count)

        def normal(_):
            state = self.env.reset(rng)
            return state.replace(info=dict(state.info,rsi_index=jp.int32(-1)))

        def demonstration(_):
            state = jax.tree_util.tree_map(lambda x:x[index],self.bank)
            info = dict(state.info,rng=rng,touch_rng=touch,rsi_index=index)
            return state.replace(info=info)

        if self.probability == 0:
            return normal(None)
        return jax.lax.cond(jax.random.uniform(selection)<self.probability,
                            demonstration,normal,operand=None)

    def step(self, state, action):
        nxt = self.env.step(state,action)
        # A mid-motion reset must not extend the original absolute 40s horizon.
        expired = (state.info["rsi_index"]>=0) & (nxt.info["step"]>=800)
        return nxt.replace(done=jp.maximum(nxt.done,expired.astype(jp.float32)))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reset-bank",type=Path,required=True)
    parser.add_argument("--reference-reset-probability",type=float,default=.5)
    args,rest = parser.parse_known_args()
    if "--method" not in rest or rest[rest.index("--method")+1] != "guided":
        parser.error("This independent group uses full-target BC+PPO")
    if not 0 <= args.reference_reset_probability <= 1:
        parser.error("Invalid reset probability")
    if "--checkpoint" not in rest or "--dataset" in rest:
        parser.error("Use a pre-existing BC checkpoint, not simultaneous BC fitting")
    inp = Path(rest[rest.index("--input")+1])
    output = Path(rest[rest.index("--output")+1])
    allowed = constrained.improvement.runner.ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
    if not output.resolve().is_relative_to(allowed):
        parser.error("Use the separate improvement root")
    meta = json.loads((args.reset_bank/"manifest.json").read_text())
    assert meta["status"] == "completed" and not meta["validation_or_test_used"]
    original = inp.parent/"demonstrations"
    source = json.loads((original/"dataset_manifest.json").read_text())
    split = source.get("train_episodes",len(source["seeds"])*3//4)
    assert digest(original/"dataset_manifest.json") == meta["source_manifest_sha256"]
    assert digest(original/"trajectories.npz") == meta["source_dataset_sha256"]
    assert set(meta["seeds"]).issubset(source["seeds"][:split])
    assert digest(inp/"model.xml") == meta["model_sha256"]
    assert digest(inp/"reference.npz") == meta["reference_sha256"]
    assert digest(args.reset_bank/"states.pkl") == meta["state_sha256"]
    bank = pickle.loads((args.reset_bank/"states.pkl").read_bytes())
    assert len(bank.reward) == meta["accepted"]
    runner = constrained.improvement.runner
    original = runner.train.train

    def train(*values,**kwargs):
        env = kwargs["environment"]
        kwargs["environment"] = ReferenceStarts(env,bank,args.reference_reset_probability)
        record = dict(reset_probability=args.reference_reset_probability,
            reset_bank=str(args.reset_bank.resolve()),reset_bank_sha256=meta["state_sha256"],
            bank_manifest_sha256=digest(args.reset_bank/"manifest.json"),
            source_sha256=digest(output/"sources"/Path(__file__).name),
            source_collection_interactions=meta["collection_interactions"],
            source_native_validation_interactions=meta["native_validation_interactions"],
            training_only=True,eval_environment_wrapped=False,teacher_hold_credit_inherited=False,
            absolute_horizon_steps=800,normal_resets_unchanged=True,
            method="full target BC+PPO, reference-state initialization; no action addition",
            limitation="Changed training reset distribution, not new videos or recovery labels; evaluation unchanged")
        (output/"reference_reset_config.json").write_text(json.dumps(record,indent=2)+"\n")
        return original(*values,**kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0],*rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original


if __name__ == "__main__":
    main()
