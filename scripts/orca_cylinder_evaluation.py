"""Audited cylinder rollouts, including absolute-control BC demonstration capture."""

from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp


def evaluator(env, net, seeds, teacher=None):
    inference = networks.make_inference_fn(net)
    teacher = None if teacher is None else jp.asarray(teacher)

    def evaluate(params):
        policy = inference(params, deterministic=True)

        def episode(key):
            state = env.reset(key)
            initial = state.data.qpos

            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = (policy(state.obs, subkey)[0] if teacher is None else
                          env.ctrl_to_action(teacher[jp.minimum(state.info["step"], 799)]))
                nxt = env.step(state, action)
                valid = alive & (nxt.done == 0)
                m = nxt.metrics
                trace = dict(action=action * env.residual_gain if env.arm == "residual" else action,
                             policy_action=action, active=alive, alive=valid,
                             hold=jp.where(valid, m["hold_sec"], 0), reward=jp.where(alive, nxt.reward, 0),
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             mocap=nxt.data.mocap_pos, forbidden=m["forbidden_contact"],
                             contacts=m["contact_count"], thumb=m["thumb_contact"],
                             released=m["released"], cube_z=m["cube_z"], spawn_z=nxt.info["cube_spawn_pos"][2],
                             cube_speed=m["cube_speed"], finger_normal_force=nxt.info["finger_normal_force"],
                             push=nxt.info["applied_push"], substep_penetration=m["substep_penetration"],
                             substep_forbidden=m["substep_forbidden"], substep_force_peak=m["substep_force_peak"])
                for name in ("video_base", "requested_residual", "effective_residual", "desired_ctrl"):
                    trace[name] = nxt.info[name]
                if teacher is not None:
                    trace["obs"] = state.obs["state"]
                return (nxt, valid, key), trace

            trace = jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key, 987654)),
                                 None, length=800)[1]
            return initial, trace

        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32)))

    return jax.jit(evaluate)
