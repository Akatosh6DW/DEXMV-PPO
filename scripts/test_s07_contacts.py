"""Check distal-pad classification independently of grasp policy performance."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from orca_s06_env import PAD_NAMES, cylinder_contacts

MODEL = Path(__file__).resolve().parents[1] / "artifacts/orca_benchmark_suite/S07/inputs/model.xml"


class ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def classify(self, finger, point, normal, reversed_pair=False):
        m, d = self.model, self.data
        gid, cube = m.geom(PAD_NAMES[finger]).id, m.geom("cube").id
        body = m.geom_bodyid[gid]
        rotation = d.xmat[body].reshape(3, 3)
        pair = (cube, gid) if reversed_pair else (gid, cube)
        contact = SimpleNamespace(geom=pair, geom1=pair[0], geom2=pair[1], dist=-.0001,
            pos=d.xpos[body] + rotation @ point,
            frame=np.r_[(rotation @ normal) * (-1 if reversed_pair else 1), np.zeros(6)])
        packet = SimpleNamespace(contact=[contact], ncon=1, xmat=d.xmat,
                                 xpos=d.xpos, geom_xpos=d.geom_xpos)
        def force(_model, _data, _index, out):
            out[:] = 0
            out[0] = 1
        # Analytic contact geometry isolates the classification rule. This is
        # not evidence that physical mesh alignment is accurate on hardware.
        with patch("orca_s06_env.mujoco.mj_contactForce", side_effect=force):
            return cylinder_contacts(m, packet)[:2]

    def test_palmar_contacts_under_body_rotation_and_pair_order(self):
        for finger in range(3):
            for reversed_pair in (False, True):
                good, bad = self.classify(finger, np.array([-.005, 0, .03]),
                                          np.array([-1., 0, 0]), reversed_pair)
                self.assertEqual(good[finger], 1)
                self.assertEqual(bad.sum(), 0)

    def test_tip_side_is_not_a_pad_contact(self):
        for finger in range(3):
            good, bad = self.classify(finger, np.array([0, .005, .03]), np.array([0., 1, 0]))
            self.assertEqual(good.sum(), 0)
            self.assertEqual(bad[finger], 1)

    def test_near_joint_is_not_distal_pad(self):
        for finger in range(3):
            threshold = .015 if finger == 0 else .024
            good, bad = self.classify(finger, np.array([-.005, 0, threshold-.001]), np.array([-1., 0, 0]))
            self.assertEqual(good.sum(), 0)
            self.assertEqual(bad[finger], 1)

    def test_back_surface_is_not_pad(self):
        for finger in range(3):
            good, bad = self.classify(finger, np.array([.005, 0, .03]), np.array([1., 0, 0]))
            self.assertEqual(good.sum(), 0)
            self.assertEqual(bad[finger], 1)


if __name__ == "__main__":
    unittest.main()
