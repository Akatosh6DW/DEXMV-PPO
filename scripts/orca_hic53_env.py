"""Source-sized HIC grasp task with explicit object pose and reference binding."""

import json
from pathlib import Path

import jax.numpy as jp
from mujoco import mjx
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from orca_refit_env import OrcaRefitGrasp
from orca_residual_env import default_config
from prepare_hic_orca_object import compiled_vertices, digest
from orca_contact_forces import normal_loads


def require_full_contact_capacity(model):
    for i in range(model.nnumeric):
        numeric = model.numeric(i)
        if numeric.name in ("max_contact_points", "max_geom_pairs") and numeric.data[0] >= 0:
            raise ValueError("Contact-capped candidates are not approved for this matched study")


def task_packet(model, reference):
    with np.load(reference) as z:
        packet = {k: z[k] for k in ("object_position", "object_quaternion", "initial_ctrl",
                                    "object_extents_m", "spatial_scale", "object_to_world_rotation")}
    if not all(np.isfinite(v).all() for v in packet.values()):
        raise ValueError("Nonfinite reference metadata")
    if packet["object_position"].shape != (3,) or packet["initial_ctrl"].shape != (model.nu,):
        raise ValueError("Invalid spawn or initial-control shape")
    low, high = model.actuator_ctrlrange.T
    if np.any(packet["initial_ctrl"] < low) or np.any(packet["initial_ctrl"] > high):
        raise ValueError("Initial hand posture exceeds actuator limits")
    if float(packet["spatial_scale"]) != 1.:
        raise ValueError("HIC53 task does not permit spatially scaled demonstrations")
    box_proxy = int(model.geom("cube").type[0]) == mujoco.mjtGeom.mjGEOM_BOX
    vertices = compiled_vertices(model, "hic20_cube_visual" if box_proxy else "cube")
    np.testing.assert_allclose(np.ptp(vertices, axis=0), packet["object_extents_m"], atol=1e-7)
    if box_proxy:
        np.testing.assert_allclose(2 * model.geom("cube").size, packet["object_extents_m"], atol=1e-9)
        from itertools import product
        vertices = np.array(list(product(*zip(-model.geom("cube").size, model.geom("cube").size))))
    np.testing.assert_allclose(packet["object_extents_m"], [.053] * 3, atol=1e-9)
    quat = packet["object_quaternion"]
    if quat.shape != (4,) or not np.isclose(np.linalg.norm(quat), 1.):
        raise ValueError("Invalid object quaternion")
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(rotation, packet["object_to_world_rotation"], atol=1e-7)
    packet["lowest_z"] = float(np.min((vertices @ rotation.T)[:, 2]))
    return packet


def check_hic53_gates(model, reference, gate_path):
    require_full_contact_capacity(mujoco.MjModel.from_xml_path(str(model)))
    gate = json.loads(Path(gate_path).read_text())
    if (not gate["geometry_and_native_pass"] or not gate["pass_mjx"]
            or not gate.get("mjx_acquisition_pass", False) or gate["backend"] != "gpu"):
        raise ValueError("HIC53 geometry/native/GPU validation not passed")
    if gate["model_sha256"] != digest(model) or gate["reference_sha256"] != digest(reference):
        raise ValueError("Validated model/reference pair has changed")
    for path, expected in gate["asset_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Validated source/mesh has changed: " + path)
    return dict(hic53=digest(gate_path))


class OrcaHic53Grasp(OrcaRefitGrasp):
    def __init__(self, model, arm, reference, weights=None, speed_scale=.05):
        if arm == "bc":
            raise ValueError("Old 36mm BC data are incompatible with this matched task")
        compiled_model = mujoco.MjModel.from_xml_path(str(model))
        require_full_contact_capacity(compiled_model)
        self.contact_cone = ("elliptic" if compiled_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal")
        if not np.all(compiled_model.geom_condim[compiled_model.geom_contype != 0] == 4):
            raise ValueError("Matched force decoder requires uniform condim=4")
        packet = task_packet(compiled_model, reference)
        cfg = default_config()
        cfg.sim_dt = float(compiled_model.opt.timestep)
        cfg.spawn_config.cube_pos = packet["object_position"].tolist()
        cfg.initial_ctrl = packet["initial_ctrl"].tolist()
        cfg.spawn_config.support_pos = (packet["object_position"] + [0, 0, packet["lowest_z"] - .004]).tolist()
        super().__init__(model, arm, reference, weights, config=cfg, speed_scale=speed_scale)
        self.half_height = -packet["lowest_z"]
        self.object_quaternion = jp.asarray(packet["object_quaternion"])

    def _has_cube_contact(self, data, geom_ids):
        contact = data.contact
        force = normal_loads(contact.efc_address, data.efc_force, self.contact_cone, xp=jp)
        active = (contact.dist <= 0) & (force > 1e-5)
        gids = jp.asarray(geom_ids)
        match0 = jp.any(contact.geom[:, 0, None] == gids[None, :], axis=1)
        match1 = jp.any(contact.geom[:, 1, None] == gids[None, :], axis=1)
        hit = ((match0 & (contact.geom[:, 1] == self._cube_geom_id)) |
               (match1 & (contact.geom[:, 0] == self._cube_geom_id)))
        return jp.any(hit & active).astype(jp.float32)

    def reset(self, rng):
        state = super().reset(rng)
        qpos = state.data.qpos.at[self._cube_qids[3:]].set(self.object_quaternion)
        data = self._forward_physics(state.data.replace(qpos=qpos))
        return state.replace(data=data, obs=self._obs(data, state.info))
