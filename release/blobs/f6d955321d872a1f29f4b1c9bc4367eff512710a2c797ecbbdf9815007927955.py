"""Explicit held-out shape models; the original HIC binding stays unchanged."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from mujoco import mjx
import numpy as np
from scipy.spatial.transform import Rotation

from orca_s03_env import OrcaS03Grasp
from orca_suite_env import AuditedPhysics


SHAPES = {"cuboid": ("box", [.0265, .021, .030]),
          "sphere": ("sphere", [.0265])}


def shape_xml(source, shape):
    kind, size = SHAPES[shape]
    root = ET.parse(source).getroot()
    body = root.find(".//body[@name='cube']")
    for name in ("cube", "hic20_cube_visual"):
        geom = body.find(f"geom[@name='{name}']")
        if geom is None:
            raise ValueError("Missing expected object geometry")
        geom.attrib.pop("mesh", None)
        geom.set("type", kind)
        geom.set("size", " ".join(map(str, size)))
        geom.set("pos", "0 0 0")
        geom.set("quat", "1 0 0 0")
    # Keep the same mass; compilation recomputes the new shape's inertia.
    if body.find("inertial") is not None:
        raise ValueError("Explicit source inertia needs an explicit migration")
    body.find("geom[@name='cube']").set("mass", ".035")
    return ET.tostring(root, encoding="unicode")


def verify_shape(source, model, shape):
    for key in ("nq", "nv", "nu", "ngeom", "nbody", "njnt", "nmocap"):
        if getattr(source, key) != getattr(model, key):
            raise ValueError("Shape migration changed topology: " + key)
    for kind, count in (("body", model.nbody), ("geom", model.ngeom),
                        ("joint", model.njnt), ("actuator", model.nu)):
        for i in range(count):
            if getattr(source, kind)(i).name != getattr(model, kind)(i).name:
                raise ValueError("Shape migration reordered " + kind)
    for key in ("actuator_ctrlrange", "actuator_trnid", "actuator_gainprm", "actuator_biasprm",
                "actuator_forcerange", "jnt_range", "dof_damping", "geom_contype",
                "geom_conaffinity", "geom_condim", "geom_friction"):
        np.testing.assert_array_equal(getattr(source, key), getattr(model, key))
    bid = model.body("cube").id
    others = np.arange(model.nbody) != bid
    for key in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        np.testing.assert_allclose(getattr(source, key)[others], getattr(model, key)[others], atol=1e-12)
    gids = [model.geom(n).id for n in ("cube", "hic20_cube_visual")]
    others = ~np.isin(np.arange(model.ngeom), gids)
    for key in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_dataid"):
        np.testing.assert_allclose(getattr(source, key)[others], getattr(model, key)[others], atol=1e-12)
    kind, size = SHAPES[shape]
    mass = .035
    if kind == "box":
        half = np.array(size)
        inertia = mass / 3 * (np.sum(half**2) - half**2)
        expected_type = mujoco.mjtGeom.mjGEOM_BOX
    else:
        inertia = np.full(3, .4 * mass * size[0]**2)
        expected_type = mujoco.mjtGeom.mjGEOM_SPHERE
    np.testing.assert_allclose(model.body_mass[bid], mass, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[bid], inertia, rtol=1e-10)
    np.testing.assert_allclose(model.body_ipos[bid], 0, atol=1e-12)
    for gid in gids:
        if model.geom_type[gid] != expected_type:
            raise ValueError("Visual/collision shape mismatch")
        np.testing.assert_allclose(model.geom_size[gid][:len(size)], size, atol=1e-12)


class OrcaS03ObjectGrasp(OrcaS03Grasp):
    def __init__(self, model, arm, reference, *, shape, audit=True, **kwargs):
        if shape not in SHAPES:
            raise ValueError("Unknown held-out shape")
        if kwargs.get("benchmark", "B0_nominal") != "B0_nominal":
            raise ValueError("Held-out shapes use nominal physical parameters")
        # Validate the original demonstration/hand before explicit task transfer.
        super().__init__(model, arm, reference, audit=audit, **kwargs)
        xml = shape_xml(model, shape)
        compiled = mujoco.MjModel.from_xml_string(xml)
        verify_shape(self.mj_model, compiled, shape)
        self.shape, self.evaluation_xml = shape, xml
        self._mj_model = compiled
        self._mjx_model = mjx.put_model(compiled, impl=self._config.impl)
        rotation = Rotation.from_quat(np.asarray(self.object_quaternion)[[1, 2, 3, 0]]).as_matrix()
        kind, size = SHAPES[shape]
        self.half_height = float(np.abs(rotation[2]) @ size if kind == "box" else size[0])
        self._config.spawn_config.support_pos = (
            np.asarray(self._config.spawn_config.cube_pos) - [0, 0, self.half_height + .004]).tolist()
        self.native = AuditedPhysics(compiled, np.asarray(self.sensor_membership),
                                     self.contact_cone, self.n_substeps, self._forbidden_gids, audit)

    def export_model(self, path):
        path = Path(path)
        path.write_text(self.evaluation_xml + "\n")
