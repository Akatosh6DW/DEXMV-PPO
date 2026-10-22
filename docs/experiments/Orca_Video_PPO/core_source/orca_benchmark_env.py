# 中文阅读副本：09 基础控制与有效抓握判据
# 原始入口：scripts/orca_benchmark_env.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""Shared demonstration-blind Orca task for scratch/BC PPO comparisons."""

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.orca_hand import grasp_cube


def default_config():
    cfg = grasp_cube.default_config()
    cfg.episode_length = 800
    cfg.reward_config.hold_success_sec = 30.0
    cfg.spawn_config.cube_pos = [.055, .025, .265]
    cfg.spawn_config.support_pos = [.055, .025, .243]
    cfg.spawn_config.cube_jitter = [.001, .001, 0.0]
    cfg.initial_ctrl = [-.25] + [0.0] * 16
    cfg.initial_noise = .002
    cfg.max_ctrl_delta = .10
    cfg.support_config.release_after_sec = 2.0
    cfg.support_config.release_ramp_sec = 1.0
    cfg.support_config.force_release_after_sec = 3.0
    return cfg


class OrcaBenchmark(grasp_cube.OrcaCubeGrasp):
    """Absolute normalized actions; no demonstration in observations/rewards."""

    def __init__(self, config=None):
        super().__init__(config=config if config is not None else default_config())
        self._tip_gids = np.array([
            self.mj_model.geom(f"orca_{f}_tip_ball_proxy").id
            for f in ("thumb", "index", "middle", "ring", "pinky")
        ])
        finger_gids = set(sum(self._finger_contact_gids.values(), []))
        self._forbidden_gids = [i for i in range(self.mj_model.ngeom)
                               if i not in finger_gids and i != self._cube_geom_id]

    def _load_prior_ctrl(self):
        return jp.zeros((1, 17))

    def _make_initial_data(self, **kwargs):
        return mjx_env.make_data(self.mj_model, impl="jax", **kwargs)

    def _forward_physics(self, data):
        return mjx.forward(self.mjx_model, data)

    def _step_physics(self, data, ctrl):
        return mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)

    def ctrl_to_action(self, ctrl):
        return jp.clip(2 * (ctrl - self._lowers) / (self._uppers - self._lowers) - 1,
                       -1, 1)

    def _tip_positions(self, data):
        return data.geom_xpos[self._tip_gids]

    def _has_cube_contact(self, data, geom_ids):
        contact = data.contact
        # Pyramidal contacts: the normal force is the sum of friction-pyramid
        # edge forces. All task geoms have condim=4, hence six edges.
        address = contact.efc_address
        edge_ids = address[:, None] + jp.arange(6)[None, :]
        normal_force = jp.sum(data.efc_force[jp.maximum(edge_ids, 0)], axis=1)
        active = (contact.dist <= 0) & (address >= 0) & (normal_force > 1e-5)
        gids = jp.array(geom_ids)
        match0 = jp.any(contact.geom[:, 0, None] == gids[None, :], axis=1)
        match1 = jp.any(contact.geom[:, 1, None] == gids[None, :], axis=1)
        hit = ((match0 & (contact.geom[:, 1] == self._cube_geom_id)) |
               (match1 & (contact.geom[:, 0] == self._cube_geom_id)))
        return jp.any(hit & active).astype(jp.float32)

    def reset(self, rng):
        rng, qkey, pkey = jax.random.split(rng, 3)
        ctrl = jp.array(self._config.initial_ctrl)
        q = jp.clip(ctrl + self._config.initial_noise * jax.random.normal(qkey, (17,)),
                    self._lowers, self._uppers)
        cube = self._spawn_pos + jax.random.uniform(pkey, (3,), minval=-1, maxval=1) * jp.array(self._config.spawn_config.cube_jitter)
        support = cube - jp.array([0, 0, .022])
        qpos = self._init_q.at[self._hand_qids].set(q)
        qpos = qpos.at[self._cube_qids].set(jp.concatenate([cube, jp.array([1., 0, 0, 0])]))
        data = self._make_initial_data(qpos=qpos,
                                qvel=jp.zeros(self.mj_model.nv), ctrl=ctrl,
                                mocap_pos=support[None, :])
        info = dict(rng=rng, step=jp.int32(0), last_act=self.ctrl_to_action(ctrl),
                    cube_spawn_pos=cube, support_pos=support,
                    stable_hold_steps=jp.int32(0), best_hold_steps=jp.int32(0))
        metrics = {k: jp.float32(0) for k in (
            "hold_sec", "best_hold_sec", "success30", "contact_count",
            "thumb_contact", "forbidden_contact", "released", "cube_z",
            "drop", "proximity", "stable", "ctrl_rate")}
        return mjx_env.State(data, self._obs(data, info), jp.float32(0),
                             jp.float32(0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        info["step"] += 1
        action = jp.clip(action, -1, 1)
        desired = self._lowers + (action + 1) * .5 * (self._uppers - self._lowers)
        delta = self._config.max_ctrl_delta
        # 导读：所有路线每50ms最多改变0.1rad目标；这是目标变化限制，不是实际关节速度硬保证。
        ctrl = jp.clip(desired, state.data.ctrl - delta, state.data.ctrl + delta)
        t = info["step"] * self.dt
        support_drop = jp.clip((t - 2) * .2, 0, .2)
        data = state.data.replace(mocap_pos=(info["support_pos"] - jp.array([0., 0., support_drop]))[None, :])
        data = self._step_physics(data, ctrl)
        contacts = self._get_contact_flags(data)
        forbidden = self._has_cube_contact(data, self._forbidden_gids)
        cube = self._cube_pos(data)
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        dropped = cube[2] < info["cube_spawn_pos"][2] - .055
        done = dropped | ~finite
        above = cube[2] > info["cube_spawn_pos"][2] - .01
        paired = (contacts[0] > 0) & (jp.sum(contacts[1:]) > 0)
        # 导读：撤托后必须有拇指和另一指接触、物体未明显下降、无禁接触，才累计连续保持。
        stable = (t >= 3) & above & paired & (forbidden == 0) & ~done
        info["stable_hold_steps"] = jp.where(stable, info["stable_hold_steps"] + 1, 0)
        info["best_hold_steps"] = jp.maximum(info["best_hold_steps"], info["stable_hold_steps"])
        distances = jp.linalg.norm(self._tip_positions(data) - cube[None, :], axis=1)
        proximity = .5 * jp.exp(-distances[0] / .04) + .5 * jp.max(jp.exp(-distances[1:] / .04))
        centering = jp.exp(-jp.linalg.norm(cube - info["cube_spawn_pos"]) / .025)
        rate = jp.mean((action - info["last_act"]) ** 2)
        # 导读：注意：这是父类历史奖励，当前链路会被OrcaRefitGrasp覆盖；不要据这一行解释最终训练。
        reward = (3 * proximity + 2 * contacts[0] + 2 * paired +
                  15 * stable + centering - 3 * forbidden * (t >= 3) - .1 * rate -
                  50 * done) * self.dt
        info["last_act"] = action
        metrics = dict(hold_sec=info["stable_hold_steps"] * self.dt,
                       best_hold_sec=info["best_hold_steps"] * self.dt,
                       success30=(info["best_hold_steps"] * self.dt >= 30).astype(jp.float32),
                       contact_count=jp.sum(contacts), thumb_contact=contacts[0],
                       forbidden_contact=forbidden, released=(t >= 3).astype(jp.float32),
                       cube_z=cube[2], drop=dropped.astype(jp.float32),
                       proximity=proximity, stable=stable.astype(jp.float32), ctrl_rate=rate)
        return state.replace(data=data, obs=self._obs(data, info), reward=reward,
                             done=done.astype(jp.float32), metrics=metrics, info=info)

    def _obs(self, data, info):
        q = data.qpos[self._hand_qids]
        dq = data.qvel[self._hand_dqids]
        cube = self._cube_pos(data)
        tips = self._tip_positions(data)
        obs = jp.concatenate([
            self.ctrl_to_action(q), jp.clip(dq * .1, -5, 5),
            self.ctrl_to_action(data.ctrl), info["last_act"],
            (cube - info["cube_spawn_pos"]) * 20, self._cube_vel(data) * 2,
            self._cube_quat(data), ((tips - cube[None, :]) * 10).ravel(),
            jp.array([jp.minimum(info["step"] * self.dt / 3, 1),
                      info["stable_hold_steps"] * self.dt / 30]),
        ])
        return dict(state=obs, privileged_state=jp.concatenate([obs, data.actuator_force]))
