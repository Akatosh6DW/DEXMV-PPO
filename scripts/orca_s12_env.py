"""Native elastic physics through the project's existing Brax PPO interface."""

import threading

from flax import struct
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np
from mujoco_playground._src import mjx_env

from orca_native_physics import NativePhysics, STATE_SPEC
from orca_s10_env import rotation_matrix
from s12_soft import CENTER, SoftConfig, SoftGrasp
from s12_physics_worker import SoftWorker


@struct.dataclass
class SoftData:
    integration_words: object
    qpos: object
    qvel: object
    ctrl: object
    mocap_pos: object
    mocap_quat: object
    vertices: object
    geom_xpos: object
    actuator_force: object
    audit: object


class SoftPhysics(NativePhysics):
    def __init__(self, cfg):
        self.worker = SoftWorker(cfg)
        self.sim = self.worker.sim
        self.model = self.sim.model
        self.state_size = mujoco.mj_stateSize(self.model, STATE_SPEC)
        self.scratch = self.sim.data
        self.lock = threading.RLock()
        self.audit = np.zeros(14, np.float32)
        self.audit[12] = 1
        self.pool = None
        example = self.pack(self.scratch)
        self.output_spec = jax.tree_util.tree_map(lambda x:jax.ShapeDtypeStruct(x.shape, x.dtype), example)

    def pack(self, data):
        words = np.empty(self.state_size, np.float64)
        mujoco.mj_getState(self.model, data, words, STATE_SPEC)
        return SoftData(words.view(np.uint32).copy(), data.qpos.astype(np.float32),
            data.qvel.astype(np.float32), data.ctrl.astype(np.float32),
            data.mocap_pos.astype(np.float32), data.mocap_quat.astype(np.float32),
            data.flexvert_xpos.astype(np.float32), data.geom_xpos.astype(np.float32),
            data.actuator_force.astype(np.float32), self.audit.copy())

    def _single(self, mode, *arrays):
        return SoftData(**self.worker.single(mode, *arrays))

    def enable_workers(self, count):
        from s12_physics_pool import PhysicsPool
        self.pool = PhysicsPool(self.sim.cfg, count)

    def batched(self, mode, *arrays):
        if self.pool is None:
            return super().batched(mode, *arrays)
        arrays = tuple(np.asarray(a) for a in arrays)
        batch_shape = arrays[0].shape[:-1]
        if not batch_shape:
            return self.single(mode, *arrays)
        batch = [tuple(a[index] for a in arrays) for index in np.ndindex(batch_shape)]
        values = [SoftData(**result) for result in self.pool.map(mode, batch)]
        return jax.tree_util.tree_map(lambda *v:np.stack(v).reshape((*batch_shape, *v[0].shape)), *values)


class OrcaS12Grasp(mjx_env.MjxEnv):
    physics_backend = "native_mujoco_volumetric_elastic_full"

    def __init__(self, cfg=SoftConfig(timestep=.0002), arm="pure", jitter=.002, residual_bound=.08):
        self.native = SoftPhysics(cfg)
        self.arm, self.jitter, self.residual_bound = arm, jitter, residual_bound
        if arm not in ("pure", "residual"):
            raise ValueError(arm)
        sim = self.native.sim
        self.qids, self.dqids = map(jp.asarray, (sim.qids, sim.dqids))
        self.rid = sim.rid
        self._lowers, self._uppers = map(jp.asarray, (sim.low, sim.high))
        self.tip_ids = jp.asarray([sim.model.geom(f"orca_{f}_tip_proxy_fit_{'02' if f=='thumb' else '03'}").id
                                  for f in ("thumb", "index", "middle", "ring", "pinky")])
        self.soft_cfg = cfg
        steps = round(cfg.episode_sec/cfg.control_dt)
        super().__init__(config_dict.ConfigDict(dict(ctrl_dt=cfg.control_dt, sim_dt=cfg.timestep,
            episode_length=steps, action_repeat=1, initial_ctrl=[0.]*17, reward_version="S12-R0")))
        self.reference_ctrl = jp.asarray([sim.reference((i+1)*cfg.control_dt) for i in range(steps)])
        self.initial_relative = jp.asarray(CENTER-sim.original_pos)

    @property
    def mj_model(self):
        return self.native.model

    @property
    def mjx_model(self):
        return None

    @property
    def xml_path(self):
        return "generated:S12"

    @property
    def action_size(self):
        return 17

    def ctrl_to_action(self, ctrl):
        return jp.clip(2*(ctrl-self._lowers)/(self._uppers-self._lowers)-1, -1, 1)

    def reference_base(self, info):
        return self.reference_ctrl[jp.minimum(info["step"], len(self.reference_ctrl)-1)]

    def _obs(self, data, info):
        center = data.vertices.mean(0)
        rot = rotation_matrix(data.qpos[self.rid+3:self.rid+7])
        actor = jp.concatenate([self.ctrl_to_action(data.qpos[self.qids]),
            jp.clip(data.qvel[self.dqids]*.1, -5, 5), self.ctrl_to_action(data.ctrl),
            info["last_act"], (center-data.qpos[self.rid:self.rid+3]-self.initial_relative)*20,
            jp.clip((center-info["previous_center"])/self.soft_cfg.control_dt, -2, 2),
            jp.ptp(data.vertices, axis=0)*20, info["touch"]*.2, info["touch_delta"]*.2,
            jp.array([info["step"]/self._config.episode_length,
                      jp.minimum(info["step"]*self.dt/self.soft_cfg.release_sec, 1),
                      jp.minimum(info["step"]*self.dt/self.soft_cfg.lift_start, 1)]),
            rot.T@jp.array([0., 0., -1.])])
        return dict(state=actor, privileged_state=jp.concatenate([actor, data.actuator_force, data.audit[9:12]]))

    def reset(self, rng):
        offset = jax.random.uniform(rng, (3,), minval=-self.jitter, maxval=self.jitter)
        data = self.native.call("s12_reset", offset)
        info = dict(step=jp.int32(0), rng=rng, offset=offset, last_act=self.ctrl_to_action(jp.zeros(17)),
            previous_center=data.vertices.mean(0), touch=jp.zeros(5), touch_delta=jp.zeros(5),
            hold_steps=jp.int32(0), best_steps=jp.int32(0), ever_invalid=jp.bool_(False),
            video_base=jp.zeros(17), requested_residual=jp.zeros(17), desired_ctrl=jp.zeros(17))
        metrics = {k:jp.float32(0) for k in ("success15", "hold_sec", "best_hold_sec", "strain", "depth",
            "force", "support", "volume_min", "coverage", "drop", "valid", "height_gain")}
        return mjx_env.State(data, self._obs(data, info), jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        action = jp.clip(action, -1, 1)
        base = self.reference_base(info) if self.arm == "residual" else jp.zeros(17)
        residual = action*self.residual_bound if self.arm == "residual" else jp.zeros(17)
        desired = (jp.clip(base+residual, self._lowers, self._uppers) if self.arm == "residual" else
                   self._lowers+(action+1)*.5*(self._uppers-self._lowers))
        data = self.native.call("s12_step", state.data.integration_words, desired, info["offset"])
        a, cfg = data.audit, self.soft_cfg
        valid, alive = a[13]>0, a[12]>0
        hold = jp.where(valid, info["hold_steps"]+1, 0)
        best = jp.maximum(info["best_steps"], hold)
        released = (info["step"]+1)*self.dt >= cfg.release_sec+.5
        quality = (a[9]<=cfg.max_strain)&(a[5]<=cfg.max_penetration)&(a[6]<=cfg.max_force)&(a[8]<.02)
        ever_invalid = info["ever_invalid"] | (released & ~quality)
        center = data.vertices.mean(0)
        gain = center[2]-CENTER[2]
        coverage = jp.minimum(a[0]/.05, 1)*jp.max(jp.minimum(a[1:5]/.05, 1))
        distance = jp.linalg.norm(data.geom_xpos[self.tip_ids]-center, axis=1)
        approach = .5*jp.exp(-distance[0]/.04)+.5*jp.max(jp.exp(-distance[1:]/.04))
        reward = (10*valid+coverage+2*approach*(~released)+2*jp.clip(gain/.05, 0, 1)*released
                  -20*jp.maximum(a[9]-.2, 0)-5*(a[5]>.003)-2*jp.minimum(a[8], 5)
                  -50*(~alive)-.01*jp.mean((action-info["last_act"])**2))*self.dt
        touch = .5*info["touch"]+.5*jp.clip(a[:5], 0, 5)
        info.update(step=info["step"]+1, last_act=action, previous_center=state.data.vertices.mean(0),
            touch=touch, touch_delta=touch-info["touch"], hold_steps=hold, best_steps=best,
            ever_invalid=ever_invalid, video_base=base, requested_residual=residual, desired_ctrl=desired)
        metrics = dict(success15=(valid & (hold*self.dt>=cfg.hold_sec) & ~ever_invalid).astype(jp.float32),
            hold_sec=hold*self.dt, best_hold_sec=best*self.dt, strain=a[9], depth=a[5], force=a[6], support=a[7],
            volume_min=a[10], coverage=coverage, drop=(~alive).astype(jp.float32), valid=a[13], height_gain=gain)
        return state.replace(data=data, obs=self._obs(data, info), reward=reward, done=(~alive).astype(jp.float32),
                             metrics=metrics, info=info)
