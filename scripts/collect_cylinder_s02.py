"""Replay a cylinder reference; admit only complete successful demonstrations."""

import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np
import mujoco

from orca_cylinder_s02_env import OrcaCylinderS02Grasp, ROOT
from orca_cylinder_evaluation import evaluator
from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from probe_hic53_orca import cube_penetration
from train_orca_acquisition import initial_gate
from train_orca_matched import initialize_matched
from run_orca_suite_job import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--start', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_pmap_shmap_merge', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', str(ROOT/'artifacts/orca30/jax_cache'))
    model, reference = args.inputs/'model.xml', args.inputs/'reference.npz'
    env = OrcaCylinderS02Grasp(model, 'pure', reference, audit=True)
    gate = initial_gate(env)
    env.export_scene(args.output/'scene.xml')
    (args.output/'initial_gate.json').write_text(json.dumps(gate, indent=2))
    net, params = initialize_matched(env, 801, .3)
    teacher, _ = load_reference(reference, env._lowers, env._uppers,
                                env._config.initial_ctrl, env.dt, 800,
                                env._config.max_ctrl_delta)
    seeds = list(range(args.start, args.start+args.count))
    started = time.monotonic()
    initial, trace = jax.device_get(evaluator(env, net, seeds, teacher)(params))
    assert np.isfinite(initial).all() and all(np.isfinite(x).all() for x in trace.values())
    row, episodes = summarize(trace, env, seeds)
    # Retain the original collector's all-frame geometry gate, in addition to
    # the stricter native-substep diagnostic recorded for this new object.
    data = mujoco.MjData(env.mj_model)
    chosen = []
    for i, episode in enumerate(episodes):
        eligible = episode['success30'] and not episode['forbidden']
        maximum = 0.
        if eligible:
            for q, m in zip(trace['qpos'][i], trace['mocap'][i]):
                data.qpos[:] = q
                data.mocap_pos[:] = m
                mujoco.mj_forward(env.mj_model, data)
                maximum = max(maximum, cube_penetration(env.mj_model, data))
        episode['collector_penetration_m'] = maximum if eligible else None
        chosen.append(eligible and maximum <= .003)
    chosen = np.asarray(chosen)
    row.update(accepted=int(chosen.sum()), attempts=len(seeds),
               elapsed_sec=time.monotonic()-started, smoke=args.smoke)
    (args.output/'summary.json').write_text(json.dumps(row, indent=2))
    (args.output/'episodes.json').write_text(json.dumps(episodes, indent=2))
    np.savez_compressed(args.output/'preview_trajectories.npz', **{k: v[:4] for k,v in trace.items()})
    np.savez_compressed(args.output/'initial_states.npz', qpos=initial, seeds=seeds)
    print(json.dumps(row), flush=True)
    if args.smoke:
        return
    if chosen.sum() < 4:
        raise ValueError('Fewer than four complete, physically valid demonstrations; BC is blocked')
    dataset = args.output/'trajectories.npz'
    np.savez_compressed(dataset, **{k: trace[k][chosen] for k in ('obs','action','active')})
    manifest = dict(model_sha256=digest(model), reference_sha256=digest(reference),
                    dataset_sha256=digest(dataset), actor_dim=123,
                    physics_backend=env.physics_backend, source_video_count=1,
                    source='DexMV official processed tomato_soup_can demonstrations',
                    retarget_report=json.loads((args.inputs/'retarget_report.json').read_text()),
                    attempts=len(seeds), accepted=int(chosen.sum()),
                    seeds=np.asarray(seeds)[chosen].tolist(),
                    teacher='Object-relative geometric retargeting; no policy weights reused',
                    split='First 75% complete rollouts training; remainder validation, not independent videos')
    (args.output/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
