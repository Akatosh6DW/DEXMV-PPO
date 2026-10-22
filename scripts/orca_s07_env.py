"""S06 replication task with explicitly applied robustness interventions."""

import mujoco
import numpy as np

from orca_s06_env import CylinderPhysics, OrcaS06Grasp


class OrcaS07Grasp(OrcaS06Grasp):
    def __init__(self, model, arm, reference, *, residual_bound=.08, **kwargs):
        super().__init__(model, arm, reference, **kwargs)
        if not 0 < residual_bound <= .15:
            raise ValueError("Residual bound outside the audited range")
        self.residual_bound = float(residual_bound)
        # S06 replaces the cube model after the suite constructor. Apply these
        # interventions to the FINAL cylinder, not the discarded cube model.
        m = self.mj_model
        m.geom_friction[m.geom_contype != 0, 0] *= self.settings.get("friction_scale", 1.)
        oid = m.body("cube").id
        m.body_mass[oid] *= self.settings.get("mass_scale", 1.)
        m.body_inertia[oid] *= self.settings.get("mass_scale", 1.)
        mujoco.mj_setConst(m, mujoco.MjData(m))
        self.native = CylinderPhysics(m, np.asarray(self.sensor_membership), self.contact_cone,
                                      self.n_substeps, self._forbidden_gids, True)
        with self._config.unlocked():
            self._config.s07 = dict(benchmark=self.benchmark, settings=self.settings,
                                   residual_bound=self.residual_bound, residual_gain=self.residual_gain,
                                   object_mass=float(m.body_mass[oid]),
                                   object_friction=m.geom("cube").friction.tolist(),
                                   criterion=self.criterion)
