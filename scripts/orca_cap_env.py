"""Single-hand fixed-bottle cap loosening, independent of the frozen cube task."""

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.orca_hand.base import OrcaHandEnv
from orca_cap_scene import FINGERS, PITCH, indices


def default_config():
    return config_dict.create(ctrl_dt=.05, sim_dt=.002, action_repeat=1, episode_length=240,
                              impl="jax", nconmax=96, njmax=384, initial_noise=.002,
                              max_ctrl_delta=.1, target_angle=float(np.pi/2),
                              hold_success_sec=1., thread_tolerance=.0002)


class OrcaCap(OrcaHandEnv):
    def __init__(self, model_path, config=None):
        super().__init__(str(model_path), default_config() if config is None else config)
        qids, turn, lift, tips = indices(self.mj_model)
        self.qids = np.asarray(qids)
        self.dqids = self.mj_model.jnt_dofadr[self.mj_model.actuator_trnid[:, 0]]
        self.turn, self.lift, self.tips = turn, lift, np.asarray(tips)
        self.turn_dof = self.mj_model.joint("cap_turn").dofadr[0]
        self.cap_id = self.mj_model.geom("cap").id
        self.cap_bid = self.mj_model.body("cap").id
        self.center = jp.array(self.mj_model.body("fixed_bottle").pos)
        self.low, self.high = jp.array(self.mj_model.actuator_ctrlrange.T)
        self.finger_ids = [np.array([g for g in range(self.mj_model.ngeom)
                                    if self.mj_model.geom(g).name.startswith(f"orca_{f}_")])
                           for f in FINGERS]
        finger_set = set(np.concatenate(self.finger_ids).tolist())
        self.forbidden_ids = np.array([g for g in range(self.mj_model.ngeom)
                                       if g not in finger_set and g != self.cap_id])

    def ctrl_to_action(self, ctrl):
        return jp.clip(2 * (ctrl-self.low)/(self.high-self.low)-1, -1, 1)

    def contact_flags(self, data):
        contact = data.contact
        address = contact.efc_address
        edges = jp.clip(address[:, None] + jp.arange(6)[None, :], 0, len(data.efc_force)-1)
        forces = jp.sum(data.efc_force[edges], axis=1)
        active = (contact.dist <= 0) & (address >= 0) & (forces > 1e-5)

        def flag(gids):
            is0 = jp.any(contact.geom[:, 0, None] == jp.array(gids)[None, :], axis=1)
            is1 = jp.any(contact.geom[:, 1, None] == jp.array(gids)[None, :], axis=1)
            match = (is0 & (contact.geom[:, 1] == self.cap_id)) | (is1 & (contact.geom[:, 0] == self.cap_id))
            return jp.any(active & match)

        finger = jp.stack([flag(g) for g in self.finger_ids])
        cap_match = jp.any(contact.geom == self.cap_id, axis=1)
        peak = jp.max(jp.where(active & cap_match, forces, 0))
        return finger, flag(self.forbidden_ids), peak

    def reset(self, rng):
        rng, key = jax.random.split(rng)
        ctrl = jp.zeros(17).at[0].set(-.25)
        q = jp.clip(ctrl + self._config.initial_noise * jax.random.normal(key, (17,)), self.low, self.high)
        qpos = jp.array(self.mj_model.qpos0).at[self.qids].set(q)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=jp.zeros(self.mj_model.nv),
                                ctrl=ctrl, impl="jax")
        data = mjx.forward(self.mjx_model, data)
        info = dict(rng=rng, step=jp.int32(0), last_act=self.ctrl_to_action(ctrl),
                    hold_steps=jp.int32(0), best_hold_steps=jp.int32(0),
                    invalid_ever=jp.bool_(False))
        metrics = {k: jp.float32(0) for k in ("angle", "degrees", "lift", "thread_error",
                   "hold_sec", "best_hold_sec", "success", "paired", "forbidden", "force", "proximity")}
        return mjx_env.State(data, self._obs(data, info), jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        info["step"] += 1
        action = jp.clip(action, -1, 1)
        target = self.low + .5 * (action + 1) * (self.high - self.low)
        ctrl = jp.clip(target, state.data.ctrl-self._config.max_ctrl_delta,
                       state.data.ctrl+self._config.max_ctrl_delta)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        fingers, forbidden, force = self.contact_flags(data)
        paired = fingers[0] & jp.any(fingers[1:])
        angle, lift = data.qpos[self.turn], data.qpos[self.lift]
        error = jp.abs(lift - PITCH * angle)
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        info["invalid_ever"] |= forbidden | (error > .001) | ~finite
        valid = (angle >= self._config.target_angle) & paired & ~info["invalid_ever"] & (error < self._config.thread_tolerance)
        info["hold_steps"] = jp.where(valid, info["hold_steps"] + 1, 0)
        info["best_hold_steps"] = jp.maximum(info["hold_steps"], info["best_hold_steps"])
        distances = jp.linalg.norm(data.geom_xpos[self.tips] - data.geom_xpos[self.cap_id], axis=1)
        proximity = .5 * jp.exp(-distances[0]/.04) + .5 * jp.max(jp.exp(-distances[1:]/.04))
        progress = jp.clip(angle, 0, self._config.target_angle) - jp.clip(state.data.qpos[self.turn], 0, self._config.target_angle)
        rate = jp.mean((action - info["last_act"])**2)
        reward = 10 * progress + self.dt * (2*proximity + fingers[0] + paired + 10*valid - 5*forbidden - .1*rate - 3*(error > self._config.thread_tolerance))
        done = ~finite | (error > .003)
        info["last_act"] = action
        metrics = dict(angle=angle, degrees=angle*180/jp.pi, lift=lift, thread_error=error,
                       hold_sec=info["hold_steps"]*self.dt, best_hold_sec=info["best_hold_steps"]*self.dt,
                       success=(info["best_hold_steps"]*self.dt >= self._config.hold_success_sec).astype(jp.float32),
                       paired=paired.astype(jp.float32), forbidden=forbidden.astype(jp.float32), force=force,
                       proximity=proximity)
        return state.replace(data=data, obs=self._obs(data, info), reward=reward,
                             done=done.astype(jp.float32), metrics=metrics, info=info)

    def _obs(self, data, info):
        pos = data.geom_xpos[self.cap_id]
        # The first 93 channels retain cube-task semantics for explicit transfer.
        obs = jp.concatenate([self.ctrl_to_action(data.qpos[self.qids]),
                              jp.clip(data.qvel[self.dqids]*.1, -5, 5),
                              self.ctrl_to_action(data.ctrl), info["last_act"],
                              (pos-self.center)*20, data.cvel[self.cap_bid, 3:]*2,
                              data.xquat[self.cap_bid], ((data.geom_xpos[self.tips]-pos)*10).ravel(),
                              jp.array([jp.minimum(info["step"]*self.dt/3, 1),
                                        data.qpos[self.turn]/self._config.target_angle])])
        return dict(state=obs, privileged_state=jp.concatenate([obs, data.actuator_force]))
