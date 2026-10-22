import unittest
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

import run_s09_phase_critic as route


class PhaseCriticTests(unittest.TestCase):
    def test_initial_functions_and_gradient_routing(self):
        learned = route.experts.learned
        action = lambda obs:jp.tanh(obs[...,:17])
        shapes = dict(state=(123,),privileged_state=(140,))
        standard = learned.BASE_FACTORY(shapes,17)
        base = (None,None,standard.value_network.init(jax.random.PRNGKey(4)))
        outputs = []
        for enabled in (False,True):
            with route.phase_critics(enabled),route.experts.residual_experts():
                outputs.append(learned.initialize_residual(learned.residual_factory(action),base,9,.03))
        shared,p0 = outputs[0]
        phase,p1 = outputs[1]
        obs = dict(state=jp.ones((6,123)).at[:,95:101].set(jp.eye(6)),
                   privileged_state=jp.ones((6,140)).at[:,93].set(jp.array([0,.5,.99,1,1,1])))
        for network,index in (("policy_network",1),("value_network",2)):
            np.testing.assert_array_equal(getattr(shared,network).apply(p0[0],p0[index],obs),
                getattr(phase,network).apply(p1[0],p1[index],obs))
        for phase_id,row in ((0,2),(1,3)):
            single = jax.tree_util.tree_map(lambda x:x[row],obs)
            grad = jax.grad(lambda value:phase.value_network.apply(p1[0],value,single))(p1[2])
            for i,p in enumerate(grad["phases"]):
                norm = sum(float(jp.sum(jp.abs(x))) for x in jax.tree_util.tree_leaves(p))
                if i==phase_id:
                    self.assertGreater(norm,0)
                else:
                    self.assertEqual(norm,0)

    def test_actual_observation_phase_and_context_restoration(self):
        from run_s09_bc_conditioned import ModeConditionedGrasp
        root = Path(__file__).resolve().parents[1]
        env = ModeConditionedGrasp(root/"artifacts/orca_benchmark_suite/S09/inputs/model.xml")
        state = env.reset(jax.random.PRNGKey(1))
        for step,expected in ((0,0),(59,0),(60,1),(799,1)):
            obs = env._obs(state.data,dict(state.info,step=jp.int32(step)))
            self.assertEqual(int(route.holding_phase(obs)),expected)
        before = route.experts.residual_expert_factory,route.experts.SINGLE_INITIALIZE,route.experts.ARCHITECTURE
        with self.assertRaisesRegex(RuntimeError,"test"),route.phase_critics(True):
            raise RuntimeError("test")
        self.assertEqual(before,(route.experts.residual_expert_factory,route.experts.SINGLE_INITIALIZE,route.experts.ARCHITECTURE))


if __name__=="__main__":
    unittest.main()
