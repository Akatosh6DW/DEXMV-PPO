"""Development-only shape transfer; frozen S02/S03 task modules stay unchanged."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from orca_s03_env import OrcaS03Grasp
from orca_suite_env import AuditedPhysics


SHAPES = {"cuboid": ("box", (.0265, .021, .030)),
          "cylinder": ("cylinder", (.0265, .0265))}


def shape_xml(source, shape):
    kind, size = SHAPES[shape]
    root = ET.parse(source).getroot()
    body = root.find(".//body[@name='cube']")
    if body.find("inertial") is not None:
        raise ValueError("Expected compiler-derived inertia")
    for name in ("cube", "hic20_cube_visual"):
        geom = body.find(f"geom[@name='{name}']")
        geom.attrib.pop("mesh", None)
        geom.set("type", kind)
        geom.set("size", " ".join(map(str, size)))
        geom.set("pos", "0 0 0")
        geom.set("quat", "1 0 0 0")
    body.find("geom[@name='cube']").set("mass", ".035")
    return ET.tostring(root, encoding="unicode")


def inertia(shape):
    kind, size = SHAPES[shape]
    if kind == "box":
        half = np.asarray(size)
        return .035 / 3 * (np.sum(half**2) - half**2)
    radius, half_height = size
    transverse = .035 * (3 * radius**2 + 4 * half_height**2) / 12
    return np.array([transverse, transverse, .035 * radius**2 / 2])


def vertical_extent(shape, quaternion):
    rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    kind, size = SHAPES[shape]
    if kind == "box":
        return float(np.abs(rotation[2]) @ size)
    # A tilted cylinder's support distance is not simply its half-height.
    radius, half_height = size
    axis_z = abs(rotation[2, 2])
    return float(half_height * axis_z + radius * np.sqrt(max(0., 1 - axis_z**2)))


class OrcaShapePilot(OrcaS03Grasp):
    @property
    def action_size(self):
        return self.mj_model.nu

    def __init__(self, model, arm, reference, *, shape, audit=True, height_weight=0., **kwargs):
        if not np.isfinite(height_weight) or not 0 <= height_weight <= 20:
            raise ValueError("Invalid development height reward weight")
        super().__init__(model, arm, reference, audit=audit, **kwargs)
        if height_weight:
            # Reuse the existing contact-gated slip feedback in OrcaMatchedGrasp.
            self.weights["height_retention"] = float(height_weight)
        self.evaluation_xml = shape_xml(model, shape)
        compiled = mujoco.MjModel.from_xml_string(self.evaluation_xml)
        original = self.mj_model
        for name in ("nq", "nv", "nu", "nbody", "ngeom", "njnt"):
            assert getattr(original, name) == getattr(compiled, name)
        for name in ("actuator_ctrlrange", "actuator_trnid", "actuator_gainprm",
                     "actuator_biasprm", "actuator_forcerange", "jnt_range",
                     "dof_damping", "geom_friction", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(original, name), getattr(compiled, name))
        bid = compiled.body("cube").id
        # Compare inertia tensors, because MuJoCo may permute principal axes.
        rot = Rotation.from_quat(compiled.body_iquat[bid][[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(rot @ np.diag(compiled.body_inertia[bid]) @ rot.T,
                                   np.diag(inertia(shape)), atol=1e-12)
        np.testing.assert_allclose(compiled.body_mass[bid], .035, atol=1e-12)
        others = np.arange(compiled.nbody) != bid
        np.testing.assert_array_equal(original.body_inertia[others], compiled.body_inertia[others])
        self.shape = shape
        self._mj_model = compiled
        # NativePhysics owns integration. MJX cannot compile cylinder/mesh pairs;
        # do not leave a stale cube MJX model that could silently simulate it.
        self._mjx_model = None
        self.half_height = vertical_extent(shape, self.object_quaternion)
        self._config.spawn_config.support_pos = (
            np.asarray(self._config.spawn_config.cube_pos) - [0, 0, self.half_height + .004]).tolist()
        self.native = AuditedPhysics(compiled, np.asarray(self.sensor_membership),
                                     self.contact_cone, self.n_substeps, self._forbidden_gids, audit)
