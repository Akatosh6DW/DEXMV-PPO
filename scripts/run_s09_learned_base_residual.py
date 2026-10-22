"""Isolated S09 residual PPO around frozen, state-dependent expert actors.

The base may be a trained portfolio or a verified BC-only expert run. The original
123/140 observations and full-command history remain intact. Only the residual
actor internally appends the frozen base's 17 normalized commands.
"""

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

import run_s09_job as runner
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_constrained_ppo import OptimizerProxy
from run_s09_mode_experts import ARCHITECTURE as EXPERT_ARCHITECTURE, network_factory as expert_factory
from retarget_s09_reference import digest

from brax.training.acme import running_statistics, specs
from brax.training.networks import FeedForwardNetwork
import jax
import jax.numpy as jp
import numpy as np

BASE_FACTORY = runner.network_factory
ARCHITECTURE = "frozen learned full-target experts plus bounded residual PPO"


def verified_bc_base(checkpoint, config):
    parent = Path(checkpoint).parent
    if not (parent/"run_config.json").exists():
        parent = parent.parent
    recovery_path = parent/"recovery_bc_config.json"
    if config.get("steps") != 0 or config.get("checkpoint") or config.get("policy_selection") or not recovery_path.exists():
        return False
    recovery = json.loads(recovery_path.read_text())
    source = Path(recovery["source_checkpoint"])
    if digest(source) != recovery["source_checkpoint_sha256"] or recovery["ppo_steps"] != 0:
        raise ValueError("BC-only ancestry mismatch")
    source_parent = source.parent if (source.parent/"run_config.json").exists() else source.parent.parent
    ancestor = json.loads((source_parent/"run_config.json").read_text())
    return (ancestor.get("steps") == 0 and ancestor.get("checkpoint") is None
            and not ancestor.get("policy_selection") and ancestor.get("bc_updates", 0) > 0
            and not (source_parent/"recovery_bc_config.json").exists())


def load_base(checkpoint, model):
    checkpoint, model = Path(checkpoint), Path(model)
    parent = checkpoint.parent
    if not (parent/"run_config.json").exists():
        parent = parent.parent
    config = json.loads((parent/"run_config.json").read_text())
    if config.get("policy_architecture") != EXPERT_ARCHITECTURE or config["method"] != "guided":
        raise ValueError("Require an explicitly trained six-expert full-target base")
    for filename, field in ((model, "model_sha256"), (model.parent/"reference.npz", "reference_sha256")):
        if digest(filename) != config[field]:
            raise ValueError("Base task mismatch: "+field)
    params = jax.tree_util.tree_map(jp.asarray, pickle.loads(checkpoint.read_bytes()))
    if len(params[1]["experts"]) != 6:
        raise ValueError("Require exactly six frozen experts")
    net = expert_factory(dict(state=(123,), privileged_state=(140,)), 17)

    def action(obs):
        logits = net.policy_network.apply(params[0], params[1], dict(state=obs))
        return jax.lax.stop_gradient(net.parametric_action_distribution.mode(logits))

    return action, params, config


def compose_action(base_action, correction, low, high, bound):
    """Add radians in normalized coordinates, so zero keeps base bits intact."""
    if not 0 < bound <= .15:
        raise ValueError("Residual bound must lie in (0,.15] radians")
    requested = jp.clip(correction, -1, 1)*bound
    full = jp.clip(base_action + 2*requested/(high-low), -1, 1)
    return full, requested


class LearnedBaseGrasp(ModeConditionedGrasp):
    def __init__(self, model, arm="residual", reference=None, *, base_action, residual_bound=.08, **kwargs):
        if arm != "residual":
            raise ValueError("Use the explicit residual method")
        # Pure internals preserve full-command history and the original R0 reward.
        super().__init__(model, "pure", None, residual_bound=residual_bound, **kwargs)
        self.base_action = base_action
        with self._config.unlocked():
            self._config.action_variant = ARCHITECTURE
            self._config.last_act_semantics = "full normalized command, never the residual"
            self._config.base_source = "frozen learned experts; no timed reference controls"

    def step(self, state, action):
        base = self.base_action(state.obs["state"])
        full, requested = compose_action(base, action, self._lowers, self._uppers, self.residual_bound)
        nxt = super().step(state, full)
        base_q = self._lowers+(base+1)*.5*(self._uppers-self._lowers)
        info = dict(nxt.info)
        # Legacy evaluator field is renamed to learned_base before writing files.
        info.update(video_base=base_q, requested_residual=requested,
                    effective_residual=nxt.data.ctrl-base_q)
        return nxt.replace(info=info)


def residual_factory(base_action):
    def factory(observation_size, action_size, **kwargs):
        if tuple(observation_size["state"]) != (123,) or tuple(observation_size["privileged_state"]) != (140,):
            raise ValueError("Require the original full-command observation interface")
        internal = dict(observation_size, state=(140,))
        net = BASE_FACTORY(internal, action_size, **kwargs)
        actor = net.policy_network

        def apply(normalizer, params, observations):
            raw = observations["state"]
            enriched = dict(observations, state=jp.concatenate([raw, base_action(raw)], axis=-1))
            return actor.apply(normalizer, params, enriched)

        return net.replace(policy_network=FeedForwardNetwork(init=actor.init, apply=apply))
    return factory


def initialize_residual(factory, base_params, seed, std):
    if not .001 < std <= 1:
        raise ValueError("Residual exploration std must lie in (.001,1]")
    shapes = dict(state=(123,), privileged_state=(140,))
    net = factory(shapes, 17)
    global_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    kp, _ = jax.random.split(global_key)
    actor = net.policy_network.init(kp)
    head = actor["params"]["hidden_2"]
    head["kernel"] = jp.zeros_like(head["kernel"])
    head["bias"] = jp.concatenate([jp.zeros(17), jp.full(17, np.log(np.expm1(std-.001)))])
    stats = running_statistics.init_state({k:specs.Array(v,jp.float32) for k,v in shapes.items()})
    # Critic still sees 140 original features; its prior is inherited explicitly.
    critic = copy.deepcopy(base_params[2])
    return net, (stats, actor, critic)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--frozen-base", type=Path, required=True)
    parser.add_argument("--residual-bound", type=float, default=.08)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    args, rest = parser.parse_known_args()
    if not 0 < args.residual_bound <= .15 or not 0 < args.critic_lr < 1:
        parser.error("Invalid residual bound or critic learning rate")
    if rest[rest.index("--method")+1] != "residual" or "--dataset" in rest:
        parser.error("This separate group only trains residual PPO")
    inp = Path(rest[rest.index("--input")+1])
    out = Path(rest[rest.index("--output")+1])
    allowed = runner.ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
    if not out.resolve().is_relative_to(allowed):
        parser.error("Keep outputs outside frozen original experiments")
    base_action, base_params, cfg = load_base(args.frozen_base, inp/"model.xml")
    base_hash = digest(args.frozen_base)
    if "--checkpoint" in rest:
        parent = Path(rest[rest.index("--checkpoint")+1]).parent
        if not (parent/"run_config.json").exists():
            parent = parent.parent
        old = json.loads((parent/"learned_base_config.json").read_text())
        if old["base_sha256"] != base_hash or old["residual_bound_rad"] != args.residual_bound:
            parser.error("Continuation changed the frozen base or action bound")
    factory = residual_factory(base_action)
    record = dict(policy_architecture=ARCHITECTURE, base_checkpoint=str(args.frozen_base.resolve()),
        base_sha256=base_hash, base_config_sha256=digest(args.frozen_base.parent/"run_config.json")
            if (args.frozen_base.parent/"run_config.json").exists() else digest(args.frozen_base.parent.parent/"run_config.json"),
        base_selection=cfg.get("policy_selection"), base_parameters_in_optimizer=False,
        base_is_pure_bc=verified_bc_base(args.frozen_base,cfg), critic_initialized_from_frozen_base=True,
        raw_actor_features=123, internal_actor_features=140, critic_features=140,
        residual_bound_rad=args.residual_bound, critic_learning_rate=args.critic_lr,
        reward_version="S09-R0 unchanged", base_action_uses_current_state=True,
        action_formula="clip(base_normalized(state)+2*bound*clip(delta,-1,1)/(high-low),-1,1)",
        effective_residual_note="actual ctrl minus base target also includes original slew limiting",
        caveat="Extra residual network and different physical exploration scale; not an architecture-only full-PPO ablation")
    original = (runner.OrcaS09Grasp,runner.network_factory,runner.initialize_matched,runner.evaluator,
                runner.train.train,runner.train.optax)

    def metadata():
        path = out/"run_config.json"
        if path.exists():
            current = json.loads(path.read_text())
            current["policy_architecture"] = ARCHITECTURE
            path.write_text(json.dumps(current,indent=2)+"\n")
            (out/"learned_base_config.json").write_text(json.dumps(record,indent=2)+"\n")

    def evaluator(*values, **kwargs):
        evaluate = original[3](*values, **kwargs)
        def run(params,seeds):
            initial,trace = evaluate(params,seeds)
            trace = dict(trace)
            trace["learned_base"] = trace.pop("video_base")
            return initial,trace
        return run

    def train(*values, **kwargs):
        if kwargs.get("normalize_observations"):
            raise ValueError("The frozen base requires raw observations")
        metadata()
        kwargs.update(log_training_metrics=True,training_metrics_steps=10240)
        runner.train.optax = OptimizerProxy(original[5],args.critic_lr)
        return original[4](*values,**kwargs)

    runner.OrcaS09Grasp = lambda model,arm,reference,**kw: LearnedBaseGrasp(
        model,arm,reference,base_action=base_action,residual_bound=args.residual_bound,**kw)
    runner.network_factory = factory
    runner.initialize_matched = lambda env,seed,std: initialize_residual(factory,base_params,seed,std)
    runner.evaluator,runner.train.train = evaluator,train
    sys.argv = [sys.argv[0],*rest]
    try:
        runner.main()
    finally:
        (runner.OrcaS09Grasp,runner.network_factory,runner.initialize_matched,runner.evaluator,
         runner.train.train,runner.train.optax) = original
        metadata()
        if digest(args.frozen_base) != base_hash:
            raise RuntimeError("Frozen base changed during execution")


if __name__ == "__main__":
    main()
