"""Raw-video cylinder task without the historical HIC53 constructor/data binding.

The inherited observation, reward, residual composition and audited integration
remain identical. Initialization is explicit because those historical classes
hard-code a 53 mm cube and require its old reference packet.
"""

from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.orca_hand import grasp_cube
from orca_matched_env import FINGERS
from orca_refit_env import BASE_WEIGHTS
from orca_residual_env import default_config
from orca_residual_controls import load_reference
from orca_suite_env import OrcaSuiteGrasp, AuditedPhysics, BENCHMARKS


class OrcaRawCanGrasp(OrcaSuiteGrasp):
    @property
    def action_size(self):
        return self.mj_model.nu

    def __init__(self, model, arm, reference, *, benchmark="B0_nominal", audit=True,
                 residual_gain=1.):
        if arm not in ("pure", "residual") or benchmark not in ("B0_nominal", "B1_position", "B5_push"):
            raise ValueError("Unknown experimental arm or condition")
        if not 0 <= residual_gain <= 1 or (arm == "pure" and residual_gain != 1):
            raise ValueError("Invalid residual gain")
        self.arm, self.residual_gain, self.residual_bound = arm, residual_gain, .08
        self.benchmark, self.settings = benchmark, dict(BENCHMARKS[benchmark])
        self.weights, self.speed_scale, self.shaping = dict(BASE_WEIGHTS), .05, 1.
        self.reference_mode = "video"
        self.tactile_enabled, self.tactile_noise, self.tactile_filter = True, .02, .5
        self.evaluation_xml = Path(model).read_text()
        compiled = mujoco.MjModel.from_xml_path(str(model))
        if compiled.nu != 17 or compiled.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
            raise ValueError("Expected original Orca controls with elliptic contacts")
        geom = compiled.geom("cube")
        if int(geom.type[0]) != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("This task requires an actual cylinder")
        with np.load(reference, allow_pickle=False) as packet:
            position, quaternion = packet["object_position"], packet["object_quaternion"]
            initial = packet["initial_ctrl"]
            np.testing.assert_array_equal(initial, np.zeros(17))
            np.testing.assert_allclose(packet["object_extents_m"],
                                       [2*geom.size[0], 2*geom.size[0], 2*geom.size[1]])
        if not np.isfinite(position).all() or not np.isclose(np.linalg.norm(quaternion), 1):
            raise ValueError("Invalid new object placement")
        axis = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()[:, 2]
        self.half_height = float(geom.size[1]*abs(axis[2]) + geom.size[0]*np.sqrt(1-axis[2]**2))
        self.object_quaternion = jp.asarray(quaternion)
        cfg = default_config()
        cfg.sim_dt = float(compiled.opt.timestep)
        cfg.initial_ctrl = initial.tolist()
        cfg.spawn_config.cube_pos = position.tolist()
        cfg.spawn_config.cube_jitter = [self.settings["jitter"]]*2+[0.]
        cfg.spawn_config.support_pos = (position-[0, 0, self.half_height+.004]).tolist()
        # Native stepping avoids unsupported MJX cylinder/mesh collision pairs.
        mjx_env.MjxEnv.__init__(self, cfg, None)
        self._mj_model, self._mjx_model, self._xml_path = compiled, None, str(model)
        grasp_cube.OrcaCubeGrasp._post_init(self)
        self._tip_gids = np.array([compiled.geom(f"orca_{f}_tip_ball_proxy").id for f in FINGERS])
        finger_gids = set(sum(self._finger_contact_gids.values(), []))
        self._forbidden_gids = [g for g in range(compiled.ngeom)
                               if g not in finger_gids and g != self._cube_geom_id]
        mask = np.zeros((5, compiled.ngeom), bool)
        for i, name in enumerate(FINGERS):
            mask[i, self._finger_contact_gids[name]] = True
        self.sensor_membership = jp.asarray(mask)
        self.contact_cone = "elliptic"
        if not np.all(compiled.geom_condim[compiled.geom_contype != 0] == 4):
            raise ValueError("Contact decoder requires condim=4")
        if arm == "residual":
            base, raw = load_reference(reference, self._lowers, self._uppers,
                                       cfg.initial_ctrl, self.dt, cfg.episode_length, cfg.max_ctrl_delta)
            self.reference_ctrl, self.raw_reference_ctrl = jp.asarray(base), jp.asarray(raw)
        else:
            self.reference_ctrl = jp.zeros((cfg.episode_length, 17))
            self.raw_reference_ctrl = self.reference_ctrl
        self.native = AuditedPhysics(compiled, mask, self.contact_cone, self.n_substeps,
                                     self._forbidden_gids, audit)
