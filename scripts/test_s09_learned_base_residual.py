import unittest
from pathlib import Path
import json
import tempfile

import jax
import jax.numpy as jp
import numpy as np

from run_s09_learned_base_residual import (
    BASE_FACTORY, LearnedBaseGrasp, compose_action, initialize_residual, residual_factory, verified_bc_base, digest,
)


class LearnedBaseResidualTests(unittest.TestCase):
    def test_bc_ancestry_is_verified_not_inferred_from_zero_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root/"source", root/"target"
            source.mkdir(); target.mkdir()
            (source/"params.pkl").write_bytes(b"source")
            cfg = dict(steps=0, checkpoint=None, bc_updates=4000)
            (source/"run_config.json").write_text(json.dumps(cfg))
            (target/"run_config.json").write_text(json.dumps(cfg))
            checkpoint = target/"params.pkl"
            self.assertFalse(verified_bc_base(checkpoint,cfg))
            (target/"recovery_bc_config.json").write_text(json.dumps(dict(
                source_checkpoint=str(source/"params.pkl"), source_checkpoint_sha256=digest(source/"params.pkl"), ppo_steps=0)))
            self.assertTrue(verified_bc_base(checkpoint,cfg))
            (source/"run_config.json").write_text(json.dumps(dict(cfg,steps=40960)))
            self.assertFalse(verified_bc_base(checkpoint,cfg))
            (source/"params.pkl").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verified_bc_base(checkpoint,cfg)

    def test_zero_identity_and_radian_bound(self):
        low,high = jp.linspace(-2,-.1,17),jp.linspace(.2,3,17)
        base = jp.linspace(-1,1,17)
        full,delta = compose_action(base,jp.zeros(17),low,high,.08)
        np.testing.assert_array_equal(full,base)
        np.testing.assert_array_equal(delta,0)
        full,delta = compose_action(base,jp.linspace(-2,2,17),low,high,.08)
        q0,q1 = low+(base+1)*.5*(high-low),low+(full+1)*.5*(high-low)
        np.testing.assert_allclose(q1,jp.clip(q0+delta,low,high),atol=3e-7)
        self.assertLessEqual(float(jp.max(jp.abs(q1-q0))),.080001)
        for bound in (0,-.1,.16,float("nan")):
            with self.assertRaises(ValueError):
                compose_action(base,jp.zeros(17),low,high,bound)

    def test_network_initialization_and_trainable_scope(self):
        base_action = lambda obs:jp.tanh(obs[..., :17])
        factory = residual_factory(base_action)
        standard = BASE_FACTORY(dict(state=(123,),privileged_state=(140,)),17)
        base_params = (None,None,standard.value_network.init(jax.random.PRNGKey(5)))
        net,params = initialize_residual(factory,base_params,4,.03)
        obs = dict(state=jax.random.normal(jax.random.PRNGKey(9),(8,123)),
                   privileged_state=jp.ones((8,140)))
        logits = net.policy_network.apply(params[0],params[1],obs)
        np.testing.assert_array_equal(logits[..., :17],0)
        np.testing.assert_allclose(jax.nn.softplus(logits[...,17:])+.001,.03,atol=1e-7)
        self.assertEqual(params[1]["params"]["hidden_0"]["kernel"].shape,(140,128))
        self.assertNotIn("experts",params[1])
        np.testing.assert_array_equal(net.value_network.apply(params[0],params[2],obs),
            standard.value_network.apply(params[0],base_params[2],obs))
        grad = jax.grad(lambda p:jp.sum(net.policy_network.apply(params[0],p,obs)[...,:17]))(params[1])
        self.assertGreater(float(jp.linalg.norm(grad["params"]["hidden_2"]["kernel"])),0)

    def test_base_commands_are_actor_inputs(self):
        zeros = residual_factory(lambda obs:jp.zeros((*obs.shape[:-1],17)))
        ones = residual_factory(lambda obs:jp.ones((*obs.shape[:-1],17)))
        shape = dict(state=(123,),privileged_state=(140,))
        a,b = zeros(shape,17),ones(shape,17)
        p = a.policy_network.init(jax.random.PRNGKey(3))
        obs = dict(state=jp.zeros(123))
        standard = BASE_FACTORY(shape,17)
        base_params = (None,None,standard.value_network.init(jax.random.PRNGKey(5)))
        _,initialized = initialize_residual(zeros,base_params,4,.03)
        stats = initialized[0]
        self.assertGreater(float(jp.max(jp.abs(a.policy_network.apply(stats,p,obs)-b.policy_network.apply(stats,p,obs)))),1e-5)

    def test_environment_keeps_full_command_history(self):
        model = Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        env = LearnedBaseGrasp(model,base_action=lambda obs:jp.full(17,-.2))
        state = env.reset(jax.random.PRNGKey(130000))
        delta = jp.full(17,.1)
        nxt = env.step(state,delta)
        full,requested = compose_action(jp.full(17,-.2),delta,env._lowers,env._uppers,.08)
        np.testing.assert_array_equal(nxt.obs["state"][51:68],full)
        np.testing.assert_array_equal(nxt.info["last_act"],full)
        np.testing.assert_array_equal(nxt.info["requested_residual"],requested)
        np.testing.assert_allclose(nxt.info["desired_ctrl"],
            jp.clip(nxt.info["video_base"]+requested,env._lowers,env._uppers),atol=3e-7)
        self.assertEqual(env._config.reward_version,"S09-R0")


if __name__ == "__main__":
    unittest.main()
