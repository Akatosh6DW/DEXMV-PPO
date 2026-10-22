"""S10 PPO interface: common Cartesian approach, learned finger control.

All methods receive the same hand-frame observations and gravity direction.
Only the residual branch adds the selected demonstration at execution time.
"""

from dataclasses import fields

from flax import struct
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from orca_native_physics import NativeData, NativePhysics
from orca_s09_env import OrcaS09Grasp, cylinder_axis
from s09_suspension import FINGERS
from s10_mobile_base import CENTER, MobileCylinder


@struct.dataclass
class MobileData(NativeData):
    audit: object
    mocap_quat: object
    initial_object_quat: object


class MobilePhysics(NativePhysics):
    def __init__(self, xml, selection):
        self.sim = MobileCylinder(xml, template=selection["template"], rule=selection["rule"],
                                  roll=selection.get("roll", 0.))
        self.audit = np.zeros(13, np.float32)
        self.initial_object_quat = np.array([1., 0, 0, 0], np.float32)
        m = self.sim.model
        membership = np.array([[(m.geom(g).name or "").startswith(f"orca_{f}_")
                                for g in range(m.ngeom)] for f in FINGERS])
        super().__init__(m, membership, "elliptic", self.sim.nsub)

    def pack(self, data):
        base = super().pack(data)
        return MobileData(**{f.name:getattr(base, f.name) for f in fields(NativeData)},
                          audit=self.audit.copy(), mocap_quat=data.mocap_quat.astype(np.float32),
                          initial_object_quat=self.initial_object_quat.copy())

    def _single(self, mode, *arrays):
        sim = self.sim
        if mode == "s10_reset":
            pos, quat, initial, vel = arrays
            sim.data = self.scratch_data()
            depth = sim.reset(pos, quat, initial, vel)
            self.initial_object_quat = np.asarray(quat, np.float32).copy()
            self.audit[:] = 0
            self.audit[6], self.audit[9] = depth, 1
        elif mode == "s10_step":
            words, desired, spawn, depth, quat = arrays
            sim.data = self.restore(words)
            sim.spawn = np.asarray(spawn)
            sim.initial_penetration = float(depth[0])
            self.initial_object_quat = np.asarray(quat, np.float32).copy()
            sim.configure_plan(spawn, quat)
            sim.hold = sim.best = 0
            sim.ever_forbidden, sim.max_depth = False, 0.
            row = sim.step(desired)
            self.audit[:] = np.r_[row["loads"], row["forbidden"], row["penetration_m"], row["peak_force_n"],
                                  row["rope_force_n"], row["alive"], row["released"], row["valid"], row["time"]]
        else:
            raise ValueError(mode)
        return self.pack(sim.data)


def rotation_matrix(q):
    w, x, y, z = q / jp.linalg.norm(q)
    return jp.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class OrcaS10Grasp(OrcaS09Grasp):
    physics_backend = "native_mujoco_s10_free_root_cartesian"

    def __init__(self, model, selection, arm="pure", residual_bound=.08):
        from pathlib import Path
        self.path = Path(model)
        self.native = MobilePhysics(self.path.read_text(), selection)
        self.arm, self.residual_bound, self.jitter = arm, residual_bound, .002
        if arm not in ("pure", "residual"):
            raise ValueError(arm)
        self.selection = dict(selection)
        sim = self.native.sim
        self.qids = jp.asarray(sim.qids)
        self.dqids = jp.asarray(self.mj_model.jnt_dofadr[self.mj_model.actuator_trnid[:, 0]])
        self.oid, self.dof = sim.oid, sim.dof
        self.rid = sim.root_qid
        self._lowers, self._uppers = map(jp.asarray, self.mj_model.actuator_ctrlrange.T)
        self.tips = jp.asarray([self.mj_model.geom(f"orca_{f}_tip_proxy_fit_{'02' if f=='thumb' else '03'}").id
                               for f in FINGERS])
        self.center = jp.asarray(CENTER)
        self.original_rot = jp.asarray(sim.original_rot.as_matrix())
        self.original_offset = jp.asarray(sim.bank["object_position"][sim.template]+[0, 0, .25]-sim.original_pos)
        self.approach_weight = self.coverage_weight = 2.
        cfg = config_dict.ConfigDict(dict(ctrl_dt=.05, sim_dt=float(self.mj_model.opt.timestep),
            episode_length=840, action_repeat=1, initial_ctrl=[0.]*17, center=CENTER.tolist(),
            release_sec=5., move_sec=2., residual_bound=residual_bound, reward_version="S10-R0", jitter=.002))
        mjx_env.MjxEnv.__init__(self, cfg)
        self.reference_ctrl = jp.asarray([sim.reference((i+1)*.05) for i in range(840)], jp.float32)

    def reference_base(self, info):
        return self.reference_ctrl[jp.minimum(info["step"], 839)]

    def _obs(self, data, info):
        pos = data.qpos[self.oid:self.oid+3]
        rootpos = data.qpos[self.rid:self.rid+3]
        current_rot = rotation_matrix(data.qpos[self.rid+3:self.rid+7])
        canonical = self.original_rot @ current_rot.T
        axis = canonical @ cylinder_axis(data.qpos[self.oid+3:self.oid+7])
        axis *= jp.where(axis[jp.argmax(jp.abs(axis))] < 0, -1., 1.)
        target_id = self.native.sim.target_id
        target_rot = rotation_matrix(data.mocap_quat[target_id])
        features = jp.concatenate([canonical@jp.array([0., 0., -1.]),
            canonical@(data.mocap_pos[target_id]-rootpos)*20,
            (current_rot.T@target_rot-jp.eye(3)).ravel(),
            jp.array([info["step"]/840, jp.minimum(info["step"]*.05/5, 1.), info["step"]<40])])
        obs = jp.concatenate([self.ctrl_to_action(data.qpos[self.qids]), jp.clip(data.qvel[self.dqids]*.1, -5, 5),
            self.ctrl_to_action(data.ctrl), info["last_act"],
            (canonical@(pos-rootpos)-self.original_offset)*20,
            canonical@(data.qvel[self.dof:self.dof+3]-data.qvel[:3])*2, axis, jp.zeros(1),
            ((data.geom_xpos[self.tips]-pos)@canonical.T*10).ravel(),
            jp.array([jp.minimum(info["step"]*.05/5, 1), 0.]), features, info["touch_obs"]])
        return dict(state=obs, privileged_state=jp.concatenate([obs, data.actuator_force]))

    def reset(self, rng):
        kq, kp, kv, kt = jax.random.split(rng, 4)
        q = jax.random.normal(kq, (4,)); q /= jp.linalg.norm(q)
        pos = self.center+jax.random.uniform(kp, (3,), minval=-.002, maxval=.002)
        vel = jax.random.uniform(kv, (3,), minval=-.005, maxval=.005)*jp.array([1., 1., 0.])
        data = self.native.call("s10_reset", pos, q, jp.zeros(17), vel)
        info = dict(step=jp.int32(0), rng=rng, touch_rng=kt, touch_filter=jp.zeros(5), touch_obs=jp.zeros(10),
            cube_spawn_pos=pos, initial_penetration=data.audit[6], reference_index=jp.int32(self.selection["template"]),
            last_act=self.ctrl_to_action(jp.zeros(17)), hold_steps=jp.int32(0), best_steps=jp.int32(0),
            max_depth=jp.float32(0), ever_forbidden=jp.bool_(False), video_base=jp.zeros(17),
            requested_residual=jp.zeros(17), effective_residual=jp.zeros(17), desired_ctrl=jp.zeros(17))
        metrics = {k:jp.float32(0) for k in ("success30", "success15", "hold_sec", "best_hold_sec", "stable",
            "released", "rope_force_n", "penetration_m", "forbidden", "peak_force_n", "coverage", "approach", "drop")}
        return mjx_env.State(data, self._obs(data, info), jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        action = jp.clip(action, -1, 1)
        base = self.reference_base(info) if self.arm=="residual" else jp.zeros(17)
        residual = action*self.residual_bound if self.arm=="residual" else jp.zeros(17)
        desired = (jp.clip(base+residual, self._lowers, self._uppers) if self.arm=="residual"
                   else self._lowers+(action+1)*.5*(self._uppers-self._lowers))
        data = self.native.call("s10_step", state.data.integration_words, desired, info["cube_spawn_pos"],
                                info["initial_penetration"][None], state.data.initial_object_quat)
        a = data.audit
        released, valid, alive = a[10]>0, a[11]>0, a[9]>0
        hold = jp.where(valid, info["hold_steps"]+1, 0)
        best = jp.maximum(info["best_steps"], hold)
        depth = jp.maximum(info["max_depth"], jp.where(released, a[6], 0))
        forbidden = info["ever_forbidden"] | (released & (a[5]>0))
        success = valid & ~forbidden & (depth<=.003)
        coverage = jp.minimum(a[0]/.02, 1)*jp.max(jp.minimum(a[1:5]/.02, 1))
        distances = jp.linalg.norm(data.geom_xpos[self.tips]-data.qpos[self.oid:self.oid+3], axis=1)
        approach = .5*jp.exp(-distances[0]/.04)+.5*jp.max(jp.exp(-distances[1:]/.04))
        reward = (15*valid-50*(~alive)+2*coverage+2*approach*(~released))*.05
        kr, kn = jax.random.split(info["touch_rng"])
        filtered = .5*info["touch_filter"]+.5*jp.clip(data.finger_force+.02*jax.random.normal(kn, (5,)), 0, 5)
        touch = jp.concatenate([info["touch_filter"]/5, (info["touch_filter"]>=.02).astype(jp.float32)])
        info.update(step=info["step"]+1, last_act=action, hold_steps=hold, best_steps=best,
            max_depth=depth, ever_forbidden=forbidden, video_base=base, requested_residual=residual,
            effective_residual=data.ctrl-base if self.arm=="residual" else jp.zeros(17), desired_ctrl=desired,
            touch_rng=kr, touch_filter=filtered, touch_obs=touch)
        metrics = dict(success30=(success & (best>=600)).astype(jp.float32),
            success15=(success & (best>=300)).astype(jp.float32), hold_sec=hold*.05, best_hold_sec=best*.05,
            stable=valid.astype(jp.float32), released=a[10], rope_force_n=a[8], penetration_m=a[6],
            forbidden=a[5], peak_force_n=a[7], coverage=coverage, approach=approach, drop=(~alive).astype(jp.float32))
        return state.replace(data=data, obs=self._obs(data, info), reward=reward, done=(~alive).astype(jp.float32),
                             metrics=metrics, info=info)
