"""Geometry checks without training or modifying historical scenes."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from orca_shape_pilot_env import OrcaShapePilot, inertia, shape_xml, vertical_extent
from orca_matched_env import height_retention

MODEL = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"


class ShapeTests(unittest.TestCase):
    def test_slip_reward_is_contact_gated(self):
        values = height_retention(np.array([.265, .260, .255, .265]), .265,
                                  np.ones(4, bool), np.array([True, True, True, False]),
                                  np.ones(4, bool), xp=np)
        np.testing.assert_allclose(values, [1., np.exp(-1), np.exp(-2), 0.])

    def test_native_action_size_without_mjx(self):
        env = OrcaShapePilot.__new__(OrcaShapePilot)
        env._mj_model = mujoco.MjModel.from_xml_string(shape_xml(MODEL, "cylinder"))
        env._mjx_model = None
        self.assertEqual(env.action_size, 17)

    def test_visual_and_collision_match(self):
        model = mujoco.MjModel.from_xml_string(shape_xml(MODEL, "cylinder"))
        for name in ("cube", "hic20_cube_visual"):
            geom = model.geom(name)
            self.assertEqual(geom.type[0], mujoco.mjtGeom.mjGEOM_CYLINDER)
            np.testing.assert_allclose(geom.size[:2], [.0265, .0265])
        bid = model.body("cube").id
        rotation = Rotation.from_quat(model.body_iquat[bid][[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(rotation @ np.diag(model.body_inertia[bid]) @ rotation.T,
                                   np.diag(inertia("cylinder")), atol=1e-12)
        self.assertAlmostEqual(model.body_mass[bid], .035)

    def test_cylinder_support_height(self):
        self.assertAlmostEqual(vertical_extent("cylinder", [1, 0, 0, 0]), .0265)
        q = Rotation.from_euler("y", 45, degrees=True).as_quat()[[3, 0, 1, 2]]
        self.assertAlmostEqual(vertical_extent("cylinder", q), .0265 * np.sqrt(2))

    def test_hand_unchanged(self):
        original = mujoco.MjModel.from_xml_path(str(MODEL))
        for shape in ("cylinder", "cuboid"):
            changed = mujoco.MjModel.from_xml_string(shape_xml(MODEL, shape))
            self.assertEqual(original.nu, changed.nu)
            for name in ("actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm",
                         "actuator_forcerange", "geom_friction", "jnt_range"):
                np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))


if __name__ == "__main__":
    unittest.main()
