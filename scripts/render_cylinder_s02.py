"""Render prespecified cylinder cases with the actual benchmark scene parameters."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from orca_cylinder_s02_env import OrcaCylinderS02Grasp, ROOT
from prepare_cylinder_s02_source import digest
from render_orca_suite import selections

OUT = ROOT/'artifacts/orca_cylinder_s02'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((OUT/'protocol.json').read_text())
    output = OUT/'videos'
    output.mkdir(exist_ok=True)
    scenes = output/'scenes'
    scenes.mkdir(exist_ok=True)
    cases, missing = [], []
    for benchmark, method in selections(protocol):
        name = method if method.startswith('zero_') else method+'_s1001'
        run = OUT/'eval'/benchmark/name
        if not (run/'summary.json').exists():
            missing.append(str(run.relative_to(OUT)))
            continue
        trace = run/'step_000000000/preview_trajectories.npz'
        episodes = json.loads((trace.parent/'episodes.json').read_text())
        config = json.loads((run/'run_config.json').read_text())
        assert episodes[0]['seed'] == 40000
        scene = scenes/(benchmark+'.xml')
        if not scene.exists():
            env = OrcaCylinderS02Grasp(OUT/'inputs/model.xml', 'pure', OUT/'inputs/reference.npz',
                                       benchmark=benchmark)
            env.export_scene(scene)
        destination = output/(benchmark+'__'+name+'__case40000')
        command = [sys.executable, str(ROOT/'scripts/render_orca_modes.py'),
                   '--model', str(scene), '--trace', str(trace), '--episode', '0',
                   '--output', str(destination), '--azimuth', '225', '--elevation', '-18',
                   '--distance', '.38', '--lookat', '.035', '.025', '.245']
        if not (destination/'report.json').exists():
            if destination.exists():
                raise FileExistsError('Inspect partial render: '+str(destination))
            print('RENDER', benchmark, method, flush=True)
            with (output/(destination.name+'.log')).open('x') as stream:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL='egl'),
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads((destination/'report.json').read_text())
        assert report['trace_sha256'] == digest(trace)
        assert report['episode'] == 0 and report['frames'] == 800 and report['seconds'] == 40
        assert report['same_saved_states_both_modes'] and report['no_physics_resimulation']
        videos = {}
        for mode in ('stl', 'collision'):
            path = destination/(mode+'.mp4')
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,duration',
                '-of', 'json', str(path)], text=True))['streams'][0]
            assert int(probe['nb_read_frames']) == 800 and probe['r_frame_rate'] == '20/1'
            assert (probe['width'], probe['height']) == (960, 720)
            assert abs(float(probe['duration'])-40) < .01
            assert report['pixel_checks'][mode]['first_std'] > 5
            videos[mode] = dict(path=str(path.relative_to(OUT)), sha256=digest(path), probe=probe)
        cases.append(dict(benchmark=benchmark, method=method,
                          train_seed=None if method.startswith('zero_') else 1001, test_seed=40000,
                          outcome=episodes[0], videos=videos, command=command, camera=report['camera'],
                          source_trace_sha256=digest(trace), scene_sha256=digest(scene),
                          run_config_sha256=digest(run/'run_config.json'),
                          note='Fixed case, failures retained; no physics resimulation; post-termination tail is diagnostic'))
        (output/'manifest.json').write_text(json.dumps(dict(complete=False, cases=cases), indent=2))
    result = dict(complete=not missing, missing=missing, cases=cases,
                  protocol_sha256=digest(OUT/'protocol.json'), selection_source_sha256=digest(Path(__file__)),
                  renderer_sha256=digest(ROOT/'scripts/render_orca_modes.py'))
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# 圆柱固定案例视频', '',
             '训练种子 1001、测试初态 40000；零修正控制器不需要训练。成功和失败均保留。',
             '手心视角；STL 外观和实际碰撞体共用同一保存轨迹，不重新仿真。', '',
             '| 条件 | 方法 | 原始/清洁成功 | 外观 | 碰撞体 |', '|---|---|---|---|---|']
    for item in cases:
        links = ['[播放]('+str(Path(item['videos'][mode]['path']).relative_to('videos'))+')'
                 for mode in ('stl', 'collision')]
        r = item['outcome']
        lines.append(f"| {item['benchmark']} | {item['method']} | {r['success30']}/{r['clean_success30']} | "
                     f"{links[0]} | {links[1]} |")
    lines += ['', f"已生成 {len(cases)} 个案例，缺失 {len(missing)} 个。"]
    (output/'index.md').write_text('\n'.join(lines)+'\n')
    if missing and not args.allow_partial:
        raise RuntimeError('Formal evaluations incomplete')
    print(json.dumps(dict(complete=not missing, cases=len(cases), missing=len(missing))))


if __name__ == '__main__':
    main()
