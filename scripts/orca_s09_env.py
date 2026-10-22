"""Multi-orientation suspended-cylinder task with explicit per-episode state.

Unlike S08, reset samples continuous SO(3). A reference bank supplies a
pose-conditioned base only to the residual arm, never to the scratch actor.
"""

from dataclasses import fields
from pathlib import Path

from flax import struct
import jax
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np

from mujoco_playground._src import mjx_env
from orca_native_physics import NativeData, NativePhysics
from s09_suspension import FINGERS, SuspendedCylinder, body_quaternion


@struct.dataclass
class SuspendedData(NativeData):
    audit: object


class SuspensionPhysics(NativePhysics):
    def __init__(self, xml):
        self.sim = SuspendedCylinder(xml)
        self.audit = np.zeros(13, np.float32)
        model = self.sim.model
        membership = np.array([[(model.geom(g).name or "").startswith(f"orca_{f}_")
                                for g in range(model.ngeom)] for f in FINGERS])
        super().__init__(model, membership, "elliptic", self.sim.nsub)

    def pack(self, data):
        base = super().pack(data)
        return SuspendedData(**{f.name: getattr(base, f.name) for f in fields(NativeData)}, audit=self.audit.copy())

    def _single(self, mode, *arrays):
        sim = self.sim
        if mode == "s09_reset":
            position, quat, initial, velocity = arrays
            sim.data = self.scratch_data()
            depth = sim.reset(position, quat, initial, velocity)
            self.audit[:] = 0
            self.audit[6], self.audit[9] = depth, 1
        elif mode == "s09_step":
            words, desired, spawn, initial_depth = arrays
            sim.data = self.restore(words)
            sim.spawn = np.asarray(spawn)
            sim.initial_penetration = float(initial_depth[0])
            # Cumulative task counters live in JAX State.info, not in shared scratch storage.
            sim.hold = sim.best = 0
            sim.ever_forbidden, sim.max_depth = False, 0.
            row = sim.step(desired)
            self.audit[:] = np.r_[row["loads"], row["forbidden"], row["penetration_m"], row["peak_force_n"],
                row["rope_force_n"], row["alive"], row["released"], row["valid"], row["time"]]
        else:
            raise ValueError(mode)
        return self.pack(sim.data)


def cylinder_axis(quaternion):
    w, x, y, z = quaternion
    axis = -jp.array([2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)])
    # End-for-end and axial rotations are symmetries of this unmarked cylinder.
    sign = jp.where(axis[jp.argmax(jp.abs(axis))] < 0, -1., 1.)
    return axis*sign


class OrcaS09Grasp(mjx_env.MjxEnv):
    actor_obs_dim, critic_obs_dim = 123, 140
    physics_backend = "native_mujoco_s09_unilateral_rope"

    def __init__(self, model, arm="pure", reference=None, *, jitter=.002,
                 residual_bound=.08, approach_weight=2., coverage_weight=2., orientation_mode="uniform"):
        if arm not in ("pure", "residual") or not 0 < residual_bound <= .15:
            raise ValueError("Invalid arm or residual bound")
        self.path = Path(model)
        self.native = SuspensionPhysics(self.path.read_text())
        self.arm, self.jitter, self.residual_bound = arm, jitter, residual_bound
        if orientation_mode not in ("uniform", "web", "x", "y", "z"):
            raise ValueError("Unknown orientation test stratum")
        self.orientation_mode = orientation_mode
        self.approach_weight, self.coverage_weight = approach_weight, coverage_weight
        sim = self.native.sim
        self._lowers, self._uppers = map(jp.asarray, self.mj_model.actuator_ctrlrange.T)
        self.qids = jp.asarray(sim.qids)
        self.dqids = jp.asarray(self.mj_model.jnt_dofadr[self.mj_model.actuator_trnid[:, 0]])
        self.oid, self.dof = sim.oid, sim.dof
        self.tips = jp.asarray([self.mj_model.geom(f"orca_{f}_tip_proxy_fit_{'02' if f=='thumb' else '03'}").id for f in FINGERS])
        cfg = config_dict.ConfigDict(dict(ctrl_dt=.05, sim_dt=float(self.mj_model.opt.timestep),
            episode_length=800, action_repeat=1, initial_ctrl=[0.]*17, center=[.08,.055,.25],
            release_sec=3., residual_bound=residual_bound, approach_weight=approach_weight,
            coverage_weight=coverage_weight, reward_version="S09-R0", jitter=jitter))
        super().__init__(cfg)
        self.center = jp.asarray(cfg.center)
        sim.reset(np.asarray(cfg.center), np.array([1.,0.,0.,0.]))
        web = np.asarray(sim.data.geom_xpos[np.asarray(self.tips)[:2]]).mean(0)-np.asarray(cfg.center)
        self.web_axis = web/np.linalg.norm(web)
        self.fixed_quaternion = jp.asarray(body_quaternion(self.web_axis if orientation_mode=="web"
            else np.eye(3)[("x","y","z").index(orientation_mode)] if orientation_mode in ("x","y","z")
            else [0.,0.,1.]), jp.float32)
        self.reference_ctrl = jp.zeros((1, 800, 17))
        self.reference_axes = jp.array([[0., 0., 1.]])
        if reference is not None:
            with np.load(reference) as z:
                ctrl, quats, time = z["ctrl"], z["object_quaternion"], z["time"]
                if ctrl.ndim != 3 or ctrl.shape[-1] != 17 or quats.shape != (len(ctrl), 4):
                    raise ValueError("S09 needs an orientation bank, not a single S08 trajectory")
                if not all(np.isfinite(v).all() for v in (ctrl, quats, time)):
                    raise ValueError("Nonfinite bank")
                sampled = np.asarray([[np.interp((np.arange(800)+1)*.05, time, c[:, j])
                                       for j in range(17)] for c in ctrl]).transpose(0, 2, 1)
                previous = np.zeros((len(ctrl), 17))
                for i in range(800):
                    sampled[:, i] = np.clip(np.clip(sampled[:, i], *self.mj_model.actuator_ctrlrange.T),
                                             previous-.1, previous+.1)
                    previous = sampled[:, i].copy()
                self.reference_ctrl = jp.asarray(sampled, jp.float32)
                self.reference_axes = jax.vmap(cylinder_axis)(jp.asarray(quats, jp.float32))
        elif arm == "residual":
            raise ValueError("Residual requires a bound orientation reference bank")

    @property
    def xml_path(self):
        return str(self.path)

    @property
    def action_size(self):
        return 17

    @property
    def mj_model(self):
        return self.native.model

    @property
    def mjx_model(self):
        return None

    def ctrl_to_action(self, ctrl):
        return jp.clip(2*(ctrl-self._lowers)/(self._uppers-self._lowers)-1, -1, 1)

    def reference_base(self, info):
        return self.reference_ctrl[info["reference_index"], jp.minimum(info["step"], 799)]

    def _obs(self, data, info):
        pos = data.qpos[self.oid:self.oid+3]
        axis = cylinder_axis(data.qpos[self.oid+3:self.oid+7])
        base = self.ctrl_to_action(self.reference_base(info)) if self.arm == "residual" else jp.zeros(17)
        obs = jp.concatenate([self.ctrl_to_action(data.qpos[self.qids]), jp.clip(data.qvel[self.dqids]*.1, -5, 5),
            self.ctrl_to_action(data.ctrl), info["last_act"], (pos-self.center)*20,
            data.qvel[self.dof:self.dof+3]*2, axis, jp.zeros(1),
            ((data.geom_xpos[self.tips]-pos)*10).ravel(),
            jp.array([jp.minimum(info["step"]*.05/3., 1), 0.]), base,
            jp.array([info["step"]/800]), info["touch_obs"]])
        return dict(state=obs, privileged_state=jp.concatenate([obs, data.actuator_force]))

    def reset(self, rng):
        kq, kp, kv, kt = jax.random.split(rng, 4)
        q = jax.random.normal(kq, (4,))
        q /= jp.linalg.norm(q)
        if self.orientation_mode != "uniform":
            q = self.fixed_quaternion
        position = self.center + jax.random.uniform(kp, (3,), minval=-self.jitter, maxval=self.jitter)
        velocity = jax.random.uniform(kv, (3,), minval=-.005, maxval=.005)*jp.array([1.,1.,0.])
        data = self.native.call("s09_reset", position, q, jp.zeros(17), velocity)
        index = jp.argmax(jp.abs(self.reference_axes@cylinder_axis(q)))
        info = dict(step=jp.int32(0), rng=rng, touch_rng=kt, touch_filter=jp.zeros(5), touch_obs=jp.zeros(10),
            cube_spawn_pos=position, initial_penetration=data.audit[6], reference_index=index,
            last_act=self.ctrl_to_action(jp.zeros(17)), hold_steps=jp.int32(0), best_steps=jp.int32(0),
            max_depth=jp.float32(0), ever_forbidden=jp.bool_(False),
            video_base=jp.zeros(17), requested_residual=jp.zeros(17), effective_residual=jp.zeros(17), desired_ctrl=jp.zeros(17))
        metrics = {k: jp.float32(0) for k in ("success30", "success15", "hold_sec", "best_hold_sec", "stable",
            "released", "rope_force_n", "penetration_m", "forbidden", "peak_force_n", "coverage", "approach", "drop")}
        return mjx_env.State(data, self._obs(data, info), jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        action = jp.clip(action, -1, 1)
        base = self.reference_base(info) if self.arm == "residual" else jp.zeros(17)
        residual = action*self.residual_bound if self.arm == "residual" else jp.zeros(17)
        desired = (jp.clip(base+residual, self._lowers, self._uppers) if self.arm == "residual"
                   else self._lowers+(action+1)*.5*(self._uppers-self._lowers))
        data = self.native.call("s09_step", state.data.integration_words, desired, info["cube_spawn_pos"],
                                info["initial_penetration"][None])
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
        reward = (15*valid-50*(~alive)+self.coverage_weight*coverage
                  +self.approach_weight*approach*(~released))*.05
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
