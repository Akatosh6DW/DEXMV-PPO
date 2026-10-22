"""Matched closed-loop and open-loop action sequence evaluation for S03."""

from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np


def evaluator(env, net, seeds, *, replay=False):
    inference = networks.make_inference_fn(net)

    def evaluate(params, action_table):
        policy = inference(params, deterministic=True)

        def episode(key, recorded):
            state = env.reset(key)
            start_qpos = state.data.qpos

            def step(carry, record):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = record if replay else policy(state.obs, subkey)[0]
                nxt = env.step(state, action)
                next_alive = alive & (nxt.done == 0)
                m = nxt.metrics
                executed = action * env.residual_gain if env.arm == "residual" else action
                trace = dict(action=executed, policy_action=action, active=alive, alive=next_alive,
                             hold=jp.where(next_alive, m["hold_sec"], 0),
                             reward=jp.where(alive, nxt.reward, 0),
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             mocap=nxt.data.mocap_pos, forbidden=m["forbidden_contact"],
                             contacts=m["contact_count"], thumb=m["thumb_contact"],
                             released=m["released"], cube_z=m["cube_z"],
                             spawn_z=nxt.info["cube_spawn_pos"][2], cube_speed=m["cube_speed"],
                             finger_normal_force=nxt.info["finger_normal_force"],
                             requested_residual=nxt.info["requested_residual"],
                             effective_residual=nxt.info["effective_residual"],
                             desired_ctrl=nxt.info["desired_ctrl"],
                             video_base=nxt.info["video_base"], push=nxt.info["applied_push"],
                             substep_penetration=m["substep_penetration"],
                             substep_forbidden=m["substep_forbidden"],
                             substep_force_peak=m["substep_force_peak"])
                return (nxt, next_alive, key), trace

            trace = jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key, 987654)),
                                 recorded)[1]
            return start_qpos, trace

        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32)), action_table)

    return jax.jit(evaluate)


def recorded_actions(trace):
    # A terminated source episode supplies no feedback-derived tail commands.
    return np.where(trace["active"][..., None], trace["policy_action"], 0).astype(np.float32)


def assert_matched_prefix(left, right, *, length=800):
    mask = left["active"][:, :length]
    for key in ("qpos", "qvel", "ctrl", "mocap", "alive", "substep_penetration", "hold"):
        np.testing.assert_array_equal(left[key][:, :length][mask], right[key][:, :length][mask],
                                      err_msg="Counterfactual mismatch: " + key)

