"""S02 control and rewards with a cylinder substituted for the cube."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from orca_shape_pilot_env import shape_xml
from orca_suite_env import OrcaSuiteGrasp, AuditedPhysics

SOURCE_MODEL = ROOT/"artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
SOURCE_REFERENCE = ROOT/"artifacts/orca_cube_acquisition/E01_flex50_refinement/candidate_reference.npz"


def cylinder_xml():
    root = ET.fromstring(shape_xml(SOURCE_MODEL, "cylinder"))
    # The original cube's local Y is world up. Keep the body pose and put the
    # cylinder axis there, so the existing opposing fingers grasp its side.
    for name in ("cube", "hic20_cube_visual"):
        root.find(f".//geom[@name='{name}']").set("quat", "0.7071067811865476 0.7071067811865476 0 0")
    return ET.tostring(root, encoding="unicode")


def cylinder_half_height(quaternion):
    rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    axis_z = abs(rotation[2, 1])
    return .0265*(axis_z+np.sqrt(max(0., 1-axis_z**2)))


class OrcaCylinderS02Grasp(OrcaSuiteGrasp):
    @property
    def action_size(self):
        return self.mj_model.nu

    def __init__(self, model, arm, reference, **kwargs):
        requested = ET.fromstring(Path(model).read_text())
        expected = ET.fromstring(cylinder_xml())
        if ET.tostring(requested) != ET.tostring(expected):
            raise ValueError("Only the documented cube-to-cylinder scene change is permitted")
        # The original constructor validates 53 mm task metadata against a cube.
        # Keep its configuration and methods; replace physics before any reset.
        super().__init__(SOURCE_MODEL, arm, reference, **kwargs)
        old = self.mj_model
        compiled = mujoco.MjModel.from_xml_string(cylinder_xml())
        for field in ("nq", "nv", "nu", "nbody", "ngeom", "njnt"):
            if getattr(old, field) != getattr(compiled, field):
                raise ValueError("Scene topology changed: " + field)
        # Preserve S02's friction and doubled-mass evaluation interventions.
        compiled.geom_friction[:] = old.geom_friction
        bid = compiled.body("cube").id
        factor = self.settings.get("mass_scale", 1.)
        compiled.body_mass[bid] *= factor
        compiled.body_inertia[bid] *= factor
        mujoco.mj_setConst(compiled, mujoco.MjData(compiled))
        for field in ("actuator_ctrlrange", "actuator_trnid", "actuator_gainprm",
                      "actuator_biasprm", "actuator_forcerange", "jnt_range",
                      "dof_damping", "geom_friction", "geom_contype", "geom_conaffinity",
                      "body_mass"):
            np.testing.assert_array_equal(getattr(old, field), getattr(compiled, field))
        other = np.arange(compiled.nbody) != bid
        np.testing.assert_array_equal(old.body_inertia[other], compiled.body_inertia[other])
        self._mj_model, self._mjx_model = compiled, None
        self.half_height = cylinder_half_height(self.object_quaternion)
        self._config.spawn_config.support_pos = (
            np.asarray(self._config.spawn_config.cube_pos)-[0, 0, self.half_height+.004]).tolist()
        self.native = AuditedPhysics(compiled, np.asarray(self.sensor_membership), self.contact_cone,
                                     self.n_substeps, self._forbidden_gids, kwargs.get("audit", False))

    def export_scene(self, destination):
        """Include benchmark physics in rendered scenes, not only the base XML."""
        mujoco.mj_saveLastXML(str(destination), self.mj_model)
