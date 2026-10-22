"""Development native cap environment: video base plus bounded PPO correction."""

import json
from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env
from oakink_cap_native import CapNativePhysics
from orca_matched_env import touch_observation
from orca_residual_controls import compose
from prepare_hic_orca_object import digest


class OakInkCap(mjx_env.MjxEnv):
    physics_backend = "native_mujoco_callback"

    def __init__(self, model_directory, reference, method="residual", tactile=True, bound=.15):
        if method not in ("pure", "residual") or not 0 < bound <= .15:
            raise ValueError("Invalid cap controller configuration")
        model_directory, reference = Path(model_directory), Path(reference)
        meta = json.loads((model_directory / "report.json").read_text())
        self._xml_path = str(model_directory / "model.xml")
        if digest(self._xml_path) != meta["model_sha256"] or digest(reference) != meta["reference_sha256"]:
            raise ValueError("Changed source-matched cap model/reference")
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        config = config_dict.create(ctrl_dt=.05, sim_dt=float(self._mj_model.opt.timestep), action_repeat=1,
                                    episode_length=400, initial_ctrl=meta["initial_ctrl"], impl="native",
                                    initial_noise=.002, max_ctrl_delta=.1)
        super().__init__(config)
        self.method, self.tactile, self.bound = method, bool(tactile), float(bound)
        self.release_angle = float(meta["parameters"]["release_angle_rad"])
        self.pitch = float(meta["parameters"]["pitch_m_per_rad"])
        self.low, self.high = (jp.asarray(a) for a in self._mj_model.actuator_ctrlrange.T)
        self.qids = self._mj_model.jnt_qposadr[self._mj_model.actuator_trnid[:, 0]]
        self.dqids = self._mj_model.jnt_dofadr[self._mj_model.actuator_trnid[:, 0]]
        self.turn, self.lift = [self._mj_model.joint(n).qposadr[0] for n in ("cap_turn", "cap_lift")]
        self.turn_dof = self._mj_model.joint("cap_turn").dofadr[0]
        self.cap_bid = self._mj_model.body("oakink_cap").id
        self.tips = np.array([self._mj_model.geom(f"orca_{f}_tip_ball_proxy").id
                              for f in ("thumb", "index", "middle", "ring", "pinky")])
        self.native = CapNativePhysics(self._mj_model, self.release_angle, self.n_substeps)
        with np.load(reference) as packet:
            anchor = int(np.argmin(abs(packet["source_frame_id"] - meta["source_anchor_frame"])))
            time = packet["time"][anchor:] - packet["time"][anchor]
            controls = packet["ctrl"][anchor:]
            raw = np.stack([np.interp((np.arange(400) + 1) * self.dt, time, controls[:, j]) for j in range(17)], axis=1)
            self.bottle_rotation = jp.asarray(packet["bottle_to_simulation"][:3, :3])
        previous = np.asarray(meta["initial_ctrl"])
        sequence = []
        for ctrl in raw:
            previous = np.clip(np.clip(ctrl, np.asarray(self.low), np.asarray(self.high)), previous - .1, previous + .1)
            sequence.append(previous.copy())
        self.reference_ctrl = jp.asarray(sequence)
        data = mujoco.MjData(self._mj_model)
        data.qpos[self.qids] = meta["initial_ctrl"]
        from oakink_cap_dynamics import apply_thread_contact_mode
        apply_thread_contact_mode(self._mj_model, data)
        mujoco.mj_forward(self._mj_model, data)
        self.initial_cap_center = jp.asarray(data.xipos[self.cap_bid])
        self.initial = jp.asarray(meta["initial_ctrl"])

    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self):
        return 17

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        raise NotImplementedError("This environment uses explicit-state native physics, not MJX or domain randomization")

    def ctrl_to_action(self, ctrl):
        return jp.clip(2 * (ctrl - self.low) / (self.high - self.low) - 1, -1, 1)

    def next_base(self, step):
        return self.reference_ctrl[jp.minimum(step, len(self.reference_ctrl) - 1)]

    def reset(self, rng):
        _, key = jax.random.split(rng)
        q = jp.clip(self.initial + .002 * jax.random.normal(key, (17,)), self.low, self.high)
        qpos = jp.asarray(self._mj_model.qpos0).at[self.qids].set(q)
        data = self.native.call("reset", qpos, self.initial)
        info = dict(step=jp.int32(0), last_action=self.ctrl_to_action(self.initial), hold_steps=jp.int32(0),
                    best_hold_steps=jp.int32(0), invalid_ever=data.hand_penetration > .003,
                    touch_rng=jax.random.fold_in(rng, 43921), touch_filter=jp.zeros(5),
                    touch_obs=jp.zeros(10), critic_touch_obs=jp.zeros(10), tactile_mask=jp.asarray(self.tactile),
                    video_base=self.initial if self.method == "residual" else jp.zeros(17),
                    requested_residual=jp.zeros(17), effective_residual=jp.zeros(17), desired_ctrl=self.initial)
        metrics = {k: jp.float32(0) for k in ("degrees", "lift", "thread_error", "released", "paired", "forbidden",
                    "hold_sec", "best_hold_sec", "success", "force", "hand_penetration", "cap_bottle_contact", "valid")}
        return mjx_env.State(data, self._obs(data, info), jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        action = jp.clip(action, -1, 1)
        if self.method == "residual":
            base = self.next_base(info["step"])
            desired, residual = compose(base, action, self.bound, self.low, self.high, xp=jp)
        else:
            base, residual = jp.zeros(17), jp.zeros(17)
            desired = self.low + .5 * (action + 1) * (self.high - self.low)
        ctrl = jp.clip(desired, state.data.ctrl - .1, state.data.ctrl + .1)
        data = self.native.call("step", state.data.integration_words, ctrl)
        info["step"] += 1
        info["video_base"], info["requested_residual"], info["desired_ctrl"] = base, residual, desired
        info["effective_residual"] = data.ctrl - base if self.method == "residual" else jp.zeros(17)
        paired = data.cap_fingers[0] & jp.any(data.cap_fingers[1:])
        thread_error = jp.abs(data.qpos[self.lift] - self.pitch * data.qpos[self.turn])
        info["invalid_ever"] |= data.forbidden | (data.hand_penetration > .005)
        displacement = (data.cap_center - self.initial_cap_center) @ self.bottle_rotation
        released = ~data.thread_active
        # Clearing the 18 mm cap height and avoiding bottle/floor support is
        # stricter than merely rotating past a small loosening angle.
        valid = (released & (displacement[2] >= .018) & paired & ~info["invalid_ever"]
                 & ~data.bottle_contact & ~data.floor_contact)
        info["hold_steps"] = jp.where(valid, info["hold_steps"] + 1, 0)
        info["best_hold_steps"] = jp.maximum(info["best_hold_steps"], info["hold_steps"])
        angle_progress = jp.clip(data.qpos[self.turn], 0, self.release_angle) - jp.clip(state.data.qpos[self.turn], 0, self.release_angle)
        distances = jp.linalg.norm(data.geom_xpos[self.tips] - data.cap_center, axis=1)
        proximity = .5 * jp.exp(-distances[0] / .04) + .5 * jp.max(jp.exp(-distances[1:] / .04))
        rate = jp.mean(((ctrl - state.data.ctrl) / .1) ** 2)
        reward = 10 * angle_progress + self.dt * (2 * proximity + paired + .5 * data.cap_fingers[0]
                  + 5 * released + 5 * released * jp.clip(displacement[2] / .018, 0, 1)
                  + 20 * valid - 5 * data.forbidden - 5 * (data.hand_penetration > .003) - .01 * rate)
        done = (data.floor_contact | (jp.linalg.norm(data.cap_center - self.initial_cap_center) > .30)
                | (data.thread_active & (thread_error > .003)))
        rng, noise_key = jax.random.split(info["touch_rng"])
        info["touch_obs"] = touch_observation(info["touch_filter"], info["tactile_mask"], xp=jp)
        info["critic_touch_obs"] = touch_observation(info["touch_filter"], True, xp=jp)
        noisy = jp.clip(data.finger_force + .02 * jax.random.normal(noise_key, (5,)), 0, 5)
        info["touch_filter"] = .5 * info["touch_filter"] + .5 * noisy
        info["touch_rng"], info["last_action"] = rng, self.ctrl_to_action(desired)
        metrics = dict(degrees=data.qpos[self.turn] * 180 / jp.pi, lift=displacement[2], thread_error=thread_error,
                       released=released.astype(jp.float32), paired=paired.astype(jp.float32), forbidden=data.forbidden.astype(jp.float32),
                       hold_sec=info["hold_steps"] * self.dt, best_hold_sec=info["best_hold_steps"] * self.dt,
                       success=((info["hold_steps"] >= 20) & ~done).astype(jp.float32),
                       force=jp.max(data.finger_force), hand_penetration=data.hand_penetration,
                       cap_bottle_contact=data.bottle_contact.astype(jp.float32), valid=valid.astype(jp.float32))
        return state.replace(data=data, obs=self._obs(data, info), reward=reward, done=done.astype(jp.float32), metrics=metrics, info=info)

    def _obs(self, data, info):
        angle = data.qpos[self.turn]
        phase = jp.array([jp.minimum(info["step"] * self.dt / 20, 1)])
        base = self.next_base(info["step"]) if self.method == "residual" else jp.zeros(17)
        common = jp.concatenate([self.ctrl_to_action(data.qpos[self.qids]), jp.clip(data.qvel[self.dqids] * .1, -5, 5),
                                 self.ctrl_to_action(data.ctrl), info["last_action"],
                                 (data.cap_center - self.initial_cap_center) * 10,
                                 jp.array([jp.sin(angle), jp.cos(angle), angle / self.release_angle,
                                           data.qpos[self.lift] * 10, data.qvel[self.turn_dof] * .1,
                                           data.thread_active.astype(jp.float32)]), data.xquat[self.cap_bid],
                                 ((data.geom_xpos[self.tips] - data.cap_center) * 10).ravel(), phase, base])
        actor = jp.concatenate([common, info["touch_obs"]])
        privileged = jp.concatenate([common, info["critic_touch_obs"], data.actuator_force, data.cap_fingers,
                                      jp.array([data.hand_penetration * 1000, data.bottle_contact.astype(jp.float32)])])
        return dict(state=actor, privileged_state=privileged)
