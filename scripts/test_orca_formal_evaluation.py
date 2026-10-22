"""Physical interventions and nominal evaluator regression checks."""

import importlib.util
import json
from pathlib import Path
import pickle
import unittest

import jax
import mujoco
import numpy as np

from evaluate_orca_formal import change_physics, episode_statistics
from evaluate_orca_matched_native import actor_control, rollout
from orca_matched_env import OrcaMatchedGrasp
from train_orca_matched import initialize_matched

ROOT = Path(__file__).resolve().parents[1]


class FormalEvaluationTests(unittest.TestCase):
    def test_native_mass_inertia_and_effective_friction(self):
        xml = '<mujoco><worldbody><geom type="plane" size="1 1 .1" friction="1.4 .02 .003"/>' \
              '<body name="cube" pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1" ' \
              'friction="1.35 .025 .004"/></body></worldbody></mujoco>'
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertGreater(d.ncon, 0)
        mu = d.contact.friction.copy()
        inertia = m.body_inertia[1].copy()
        change_physics(m, {"friction_scale": .7})
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.contact.friction, mu * .7)
        np.testing.assert_array_equal(m.body_inertia[1], inertia)
        change_physics(m, {"mass_scale": 1.5})
        self.assertAlmostEqual(m.body_mass[1], 1.5)
        np.testing.assert_allclose(m.body_inertia[1], inertia * 1.5)

    def test_nominal_saved_evaluator_remains_exact(self):
        prior_source = ROOT / "artifacts/orca_collision_refit/E05_hic53_residual_seed202_heldout/evaluation_source.py"
        run = ROOT / "artifacts/orca_collision_refit/R15_hic53_native_bound15_seed202"
        if not prior_source.exists():
            self.skipTest("Prior regression fixture unavailable")
        spec = importlib.util.spec_from_file_location("old_native_evaluation", prior_source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cfg = json.loads((run / "run_config.json").read_text())
        env = OrcaMatchedGrasp(ROOT / cfg["model"], "residual", ROOT / cfg["reference"],
                              cfg["weights"], tactile=True, residual_bound=.15)
        net, _ = initialize_matched(env, cfg["seed"], cfg["init_std"])
        params = pickle.loads((run / "step_000655360/params.pkl").read_bytes())
        inference = actor_control(env, net)
        previous = old.rollout(env, inference, params, 19000)
        current = rollout(env, inference, params, 19000)
        for key in previous:
            np.testing.assert_array_equal(previous[key], current[key], err_msg=key)
        combined = {k: v[None] for k, v in current.items()}
        valid, _ = episode_statistics(combined, env)
        self.assertEqual(valid["success30"], 1.)
        combined["initial_penetration_m"] = np.full_like(combined["initial_penetration_m"], .004)
        invalid, _ = episode_statistics(combined, env)
        self.assertEqual(invalid["raw_success30"], 1.)
        self.assertEqual(invalid["success30"], 0.)
        self.assertEqual(invalid["invalid_initial_count"], 1)
        zero = rollout(env, actor_control(env, net, zero_residual=True), params, 19000)
        np.testing.assert_array_equal(zero["requested_residual"], 0.)


if __name__ == "__main__":
    unittest.main()
