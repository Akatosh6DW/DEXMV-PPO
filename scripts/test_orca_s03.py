"""S03 interface, audit-only physics and development selection checks."""

import unittest

from run_orca_s03_development import BASE, candidate, rank
from run_orca_s03_job import MODEL, REFERENCE, validate_job
from orca_s03_env import OrcaS03Grasp
from orca_suite_env import OrcaSuiteGrasp
import jax
import jax.numpy as jp
import numpy as np


class SelectionTests(unittest.TestCase):
    def test_single_variable(self):
        for method, recipe in BASE.items():
            for r in (1, 2, 3):
                new = candidate(method, r, recipe)
                self.assertEqual(sum(new[k] != recipe[k] for k in recipe), 1)
                self.assertEqual(recipe, BASE[method])

    def test_clean_auc_first(self):
        a = dict(status="completed", clean_auc=.5, last=.7, final_force_n=30.)
        b = dict(status="completed", clean_auc=.4, last=1., final_force_n=10.)
        self.assertGreater(rank([a, a]), rank([b, b]))
        with self.assertRaises(ValueError):
            rank([a])

    def test_training_rejects_test_access_or_old_weights(self):
        base = dict(method="ppo", steps=163840, eval_count=16, eval_start=61000, evals=3)
        validate_job(base)
        for change in (dict(eval_start=71000), dict(checkpoint="old.pkl"), dict(method="zero"),
                       dict(benchmark="B4_mass"), dict(steps=100)):
            with self.assertRaises(ValueError):
                validate_job(dict(base, **change))


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_pmap_shmap_merge", False)

    def test_nominal_matches_frozen_suite_and_audit_does_not_change_physics(self):
        envs = [OrcaSuiteGrasp(MODEL, "residual", REFERENCE, audit=False),
                OrcaS03Grasp(MODEL, "residual", REFERENCE, audit=True)]
        states = [jax.jit(e.reset)(jax.random.PRNGKey(60001)) for e in envs]
        steps = [jax.jit(e.step) for e in envs]
        for _ in range(10):
            states = [f(s, jp.zeros(17)) for f, s in zip(steps, states)]
            np.testing.assert_array_equal(states[0].data.qpos, states[1].data.qpos)
            np.testing.assert_array_equal(states[0].obs["state"], states[1].obs["state"])
            self.assertEqual(float(states[0].reward), float(states[1].reward))

    def test_shaping_changes_only_reward(self):
        envs = [OrcaS03Grasp(MODEL, "pure", REFERENCE, shaping=v) for v in (1., 0.)]
        states = [jax.jit(e.reset)(jax.random.PRNGKey(60002)) for e in envs]
        nxt = [jax.jit(e.step)(s, jp.zeros(17)) for e, s in zip(envs, states)]
        np.testing.assert_array_equal(nxt[0].data.qpos, nxt[1].data.qpos)
        np.testing.assert_array_equal(nxt[0].obs["state"], nxt[1].obs["state"])
        self.assertAlmostEqual(float(nxt[0].reward - nxt[1].reward),
                               float(nxt[0].metrics["acquisition_reward"]*.05), places=6)

    def test_force_and_validation(self):
        env = OrcaS03Grasp(MODEL, "residual", REFERENCE, push_force=.6)
        s = jax.jit(env.reset)(jax.random.PRNGKey(60003))
        step = jax.jit(env.step)
        for index, force in ((199, 0.), (200, .6), (203, .6), (204, 0.)):
            nxt = step(s.replace(info=dict(s.info, step=jp.int32(index))), jp.zeros(17))
            self.assertAlmostEqual(float(jp.linalg.norm(nxt.info["applied_push"])), force, places=6)
        for bad in (dict(shaping=-1), dict(push_force=float("nan")), dict(push_force=3)):
            with self.assertRaises(ValueError):
                OrcaS03Grasp(MODEL, "pure", REFERENCE, **bad)


if __name__ == "__main__":
    unittest.main()
