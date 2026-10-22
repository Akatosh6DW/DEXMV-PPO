"""Cylinder three-finger task, isolated from the frozen cube and S02 tasks."""

import mujoco
import numpy as np

from orca_cylinder_s02_env import SOURCE_MODEL, cylinder_half_height
from orca_s05_env import OrcaS05Grasp, PadPhysics
from orca_suite_env import AuditedPhysics

FINGERS = ("thumb", "index", "middle")
PAD_NAMES = ("orca_thumb_tip_proxy_fit_02", "orca_index_tip_proxy_fit_03", "orca_middle_tip_proxy_fit_03")


def cylinder_contacts(model, data):
    """Measure actual loads on distal palmar surfaces, not proximity alone."""
    cube = model.geom("cube").id
    good, bad = np.zeros(3), np.zeros(3)
    for ci, c in enumerate(data.contact[:data.ncon]):
        if cube not in c.geom or c.dist > 0:
            continue
        gid = c.geom2 if c.geom1 == cube else c.geom1
        name = model.geom(gid).name or ""
        finger = next((i for i, f in enumerate(FINGERS) if name.startswith(f"orca_{f}_")), None)
        if finger is None:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, force)
        if force[0] <= 1e-5:
            continue
        body = model.geom_bodyid[gid]
        rot = data.xmat[body].reshape(3, 3)
        point = (c.pos - data.xpos[body]) @ rot
        normal = (c.frame[:3] * (1 if c.geom1 == gid else -1)) @ rot
        valid = (name.startswith(f"orca_{FINGERS[finger]}_tip_")
                 and point[2] >= (.015 if finger == 0 else .024)
                 and -normal[0] >= .7 and abs(normal[1]) <= .5)
        (good if valid else bad)[finger] += force[0]
    potential = []
    for name in PAD_NAMES:
        pad = model.geom(name).id
        delta = data.geom_xpos[cube] - data.geom_xpos[pad]
        distance = np.linalg.norm(delta)
        facing = -data.xmat[model.geom_bodyid[pad]].reshape(3, 3)[:, 0]
        potential.append(max(0., facing @ delta / max(distance, 1e-9))**2 * np.exp(-abs(distance-.035)/.025))
    return good, bad, np.array(potential)


class CylinderPhysics(PadPhysics):
    def sample_contacts(self, data):
        AuditedPhysics.sample_contacts(self, data)
        good, bad, potential = cylinder_contacts(self.model, data)
        # Preserve the shared trace schema: thumb and BOTH opposing fingers.
        self.pad_audit[:2] = np.minimum(self.pad_audit[:2], [good[0] > .02, np.all(good[1:] > .02)])
        self.pad_audit[2:4] = np.maximum(self.pad_audit[2:4], [bad[0] > .02, np.any(bad[1:] > .02)])
        self.pad_audit[4:] = [potential[0], potential[1:].mean()]


class OrcaS06Grasp(OrcaS05Grasp):
    criterion = "S06 cylinder: thumb AND index AND middle distal palmar contacts, 30s, substep audited"

    @property
    def action_size(self):
        return self.mj_model.nu

    def __init__(self, model, arm, reference, **kwargs):
        # Inherited initialization validates the 53 mm cube metadata; compile
        # the real cylinder before any state is reset or physics is stepped.
        super().__init__(SOURCE_MODEL, arm, reference, **kwargs)
        compiled = mujoco.MjModel.from_xml_path(str(model))
        old = self.mj_model
        if compiled.geom("cube").type[0] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("S06 requires a cylinder")
        for field in ("nq", "nv", "nu", "nbody", "ngeom", "njnt"):
            if getattr(old, field) != getattr(compiled, field):
                raise ValueError("Changed topology: " + field)
        np.testing.assert_allclose(compiled.geom("cube").size[:2], [.0265, .0265])
        for field in ("actuator_ctrlrange", "actuator_trnid", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(old, field), getattr(compiled, field))
        self._mj_model, self._mjx_model = compiled, None
        self._sim_dt = float(compiled.opt.timestep)
        self._config.sim_dt = self._sim_dt
        if not np.isclose(self.dt / self._sim_dt, round(self.dt / self._sim_dt)):
            raise ValueError("Physics timestep must divide the control period")
        self.half_height = cylinder_half_height(self.object_quaternion)
        self._config.spawn_config.support_pos = (np.asarray(self._config.spawn_config.cube_pos)-[0, 0, self.half_height+.004]).tolist()
        self.native = CylinderPhysics(compiled, np.asarray(self.sensor_membership), self.contact_cone,
                                      self.n_substeps, self._forbidden_gids, True)

