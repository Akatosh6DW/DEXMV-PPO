"""S08 wrap adapter around the frozen matched PPO/BC training loop."""

import argparse
import functools
import sys

import run_s05_job as runner
import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks

from orca_s08_env import OrcaS08Grasp
from orca_suite_env import BENCHMARKS


def evaluator(env, net, seeds, teacher=None):
    inference = networks.make_inference_fn(net)
    def evaluate(params):
        policy = inference(params, deterministic=True)
        def episode(key):
            state = env.reset(key)
            initial = state.data.qpos
            def step(carry, _):
                state, alive, key = carry
                key, subkey = jax.random.split(key)
                action = (policy(state.obs,subkey)[0] if teacher is None else
                    env.ctrl_to_action(teacher[jp.minimum(state.info["step"],len(teacher)-1)]))
                nxt = jax.lax.cond(alive,lambda s: env.step(s,action),lambda s:s,state)
                next_alive = alive & (nxt.done==0)
                m = nxt.metrics
                trace = dict(obs=state.obs["state"],action=action,active=alive,alive=next_alive,
                    hold=jp.where(next_alive,m["pad_hold_sec"],0),broad_hold=m["hold_sec"],
                    qpos=nxt.data.qpos,qvel=nxt.data.qvel,ctrl=nxt.data.ctrl,mocap=nxt.data.mocap_pos,
                    pad_audit=nxt.data.pad_audit,pad_pair=m["pad_pair"],forbidden=m["forbidden_contact"],
                    contacts=m["contact_count"],thumb=m["thumb_contact"],released=m["released"],
                    cube_z=m["cube_z"],spawn_z=nxt.info["cube_spawn_pos"][2],
                    substep_penetration=m["substep_penetration"],substep_forbidden=m["substep_forbidden"],
                    substep_force_peak=m["substep_force_peak"],requested_residual=nxt.info["requested_residual"],
                    effective_residual=nxt.info["effective_residual"],video_base=nxt.info["video_base"],
                    desired_ctrl=nxt.info["desired_ctrl"],wrap_loads=nxt.data.wrap_loads,
                    initial_penetration=nxt.info["initial_penetration"],
                    reward=jp.where(alive,nxt.reward,0))
                return (nxt,next_alive,key),trace
            return initial,jax.lax.scan(step,(state,jp.bool_(True),jax.random.fold_in(key,987654)),None,length=800)[1]
        return jax.vmap(episode)(jax.vmap(jax.random.PRNGKey)(jp.array(seeds,jp.int32)))
    return jax.jit(evaluate)


BASE_SUMMARIZE = runner.summarize


def summarize(trace,seeds):
    row,episodes = BASE_SUMMARIZE(trace,seeds)
    for i,e in enumerate(episodes):
        e["initial_penetration_m"] = float(trace["initial_penetration"][i,0])
        if e["initial_penetration_m"]>.001:
            e["success30"] = e["success15"] = False
    row.update(success30=float(np.mean([e["success30"] for e in episodes])),
        success15=float(np.mean([e["success15"] for e in episodes])),
        max_initial_penetration_m=max(e["initial_penetration_m"] for e in episodes))
    return row,episodes


def initial_gate(env,count=16):
    states = jax.device_get(jax.jit(jax.vmap(env.reset))(
        jax.vmap(jax.random.PRNGKey)(jp.arange(8800,8800+count))))
    depth = np.asarray(states.info["initial_penetration"])
    assert states.obs["state"].shape == (count,123)
    assert np.isfinite(states.obs["state"]).all()
    assert np.all(states.obs["state"][:,94]==0)
    if env.benchmark == "B0_nominal":
        assert depth.max()<=.001,depth
    return dict(initial_penetration_max_m=float(depth.max()),
        seeds=list(range(8800,8800+count)),actor_dim=123,critic_dim=140,
        nominal_initial_gate=env.benchmark=="B0_nominal",passed=True)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--benchmark",choices=tuple(BENCHMARKS),default="B0_nominal")
    p.add_argument("--residual-bound",type=float,default=.08)
    p.add_argument("--residual-gain",type=float,default=1.)
    args,rest=p.parse_known_args()
    runner.OrcaS05Grasp=functools.partial(OrcaS08Grasp,**vars(args))
    runner.evaluator,runner.summarize,runner.initial_gate=evaluator,summarize,initial_gate
    sys.argv=[sys.argv[0],*rest]
    runner.main()


if __name__=="__main__":
    main()
