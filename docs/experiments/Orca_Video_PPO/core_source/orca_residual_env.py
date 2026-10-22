# 中文阅读副本：08 把每步参考与策略接入环境
# 原始入口：scripts/orca_residual_env.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""Matched grasp tasks with either full PPO control or a strict residual tube."""

import jax.numpy as jp
from mujoco import mjx
import numpy as np

from mujoco_playground._src.manipulation.orca_hand import grasp_cube
from orca_benchmark_env import OrcaBenchmark, default_config as benchmark_config
from orca_residual_controls import compose, load_reference


def default_config():
    cfg = benchmark_config()
    cfg.spawn_config.cube_jitter = [.002, .002, 0.]
    return cfg


class OrcaResidualGrasp(OrcaBenchmark):
    def __init__(self, model_path, arm, reference=None, residual_bound=.08, config=None):
        if arm not in ("pure", "residual"):
            raise ValueError("arm must be pure or residual")
        if not 0 < residual_bound <= .15:
            raise ValueError("This experiment requires a small residual bound in (0, .15] rad")
        self.arm = arm
        self.residual_bound = float(residual_bound)
        cfg = default_config() if config is None else config
        grasp_cube.OrcaCubeGrasp.__init__(self, config=cfg, xml_path=str(model_path))
        self._tip_gids = np.array([self.mj_model.geom(f"orca_{f}_tip_ball_proxy").id
                                  for f in ("thumb", "index", "middle", "ring", "pinky")])
        finger_gids = set(sum(self._finger_contact_gids.values(), []))
        self._forbidden_gids = [i for i in range(self.mj_model.ngeom)
                               if i not in finger_gids and i != self._cube_geom_id]
        geom = self.mj_model.geom("cube")
        self.half_height = float(geom.size[0] if int(geom.type[0]) == 2 else geom.size[2])
        if arm == "residual":
            if reference is None:
                raise ValueError("Residual control requires an explicit video-derived reference")
            base, raw = load_reference(reference, self._lowers, self._uppers, cfg.initial_ctrl,
                                       self.dt, cfg.episode_length, cfg.max_ctrl_delta)
            self.reference_ctrl, self.raw_reference_ctrl = jp.array(base), jp.array(raw)
        else:
            # Pure PPO does not read the reference file or receive its contents.
            self.reference_ctrl = jp.zeros((cfg.episode_length, self.action_size))
            self.raw_reference_ctrl = self.reference_ctrl

    # 导读：以当前控制步查参考，而不是用实时RGB预测；超过轨迹尾部时保持。
    def next_base(self, step):
        return self.reference_ctrl[jp.minimum(step, len(self.reference_ctrl)-1)]

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info["support_pos"] = info["cube_spawn_pos"] - jp.array([0., 0., self.half_height+.004])
        data = state.data.replace(mocap_pos=info["support_pos"][None, :])
        data = self._forward_physics(data)
        info.update(video_base=jp.array(self._config.initial_ctrl),
                    requested_residual=jp.zeros(17), effective_residual=jp.zeros(17),
                    desired_ctrl=data.ctrl)
        metrics = dict(state.metrics, residual_max=jp.float32(0), effective_residual_max=jp.float32(0))
        return state.replace(data=data, info=info, obs=self._obs(data, info), metrics=metrics)

    # 导读：residual分支将17维动作解释为修正；pure分支将同样17维解释为完整关节目标。
    def step(self, state, action):
        if self.arm == "residual":
            base = self.next_base(state.info["step"])
            desired, residual = compose(base, action, self.residual_bound,
                                         self._lowers, self._uppers, xp=jp)
            absolute_action = self.ctrl_to_action(desired)
        else:
            base, residual = jp.zeros(17), jp.zeros(17)
            absolute_action = jp.clip(action, -1, 1)
            desired = self._lowers+(absolute_action+1)*.5*(self._uppers-self._lowers)
        # The shared parent applies exactly the same slew limits, physics and reward.
        nxt = super().step(state, absolute_action)
        # 导读：requested是策略请求值；effective是限幅/限速后实际控制值减参考，二者不能混用。
        effective = nxt.data.ctrl-base if self.arm == "residual" else jp.zeros(17)
        info = dict(nxt.info, video_base=base, requested_residual=residual,
                    effective_residual=effective, desired_ctrl=desired)
        metrics = dict(nxt.metrics, residual_max=jp.max(jp.abs(residual)),
                       effective_residual_max=jp.max(jp.abs(effective)))
        return nxt.replace(info=info, metrics=metrics, obs=self._obs(nxt.data, info))

    # 导读：残差策略看到归一化参考和时间；纯PPO的参考通道填零，维度保持一致。
    def _obs(self, data, info):
        obs = super()._obs(data, info)
        base = self.ctrl_to_action(self.next_base(info["step"])) if self.arm == "residual" else jp.zeros(17)
        extra = jp.concatenate([base, jp.array([info["step"] / self._config.episode_length])])
        return {k: jp.concatenate([v, extra]) for k, v in obs.items()}
