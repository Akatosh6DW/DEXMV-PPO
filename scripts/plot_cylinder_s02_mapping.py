"""Visualize source geometry and IK errors without evaluating or selecting policies."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def main():
    with np.load(OUT/'inputs/reference.npz') as data:
        ref = {k: data[k].copy() for k in data.files}
    model = mujoco.MjModel.from_xml_path(str(OUT/'inputs/model.xml'))
    data = mujoco.MjData(model)
    qids, obj, _ = model_indices(model)
    tips = [model.geom('orca_'+name+'_tip_proxy_fit_02').id for name in ('thumb', 'index')]
    data.qpos[obj:obj+7] = np.r_[ref['object_position'], ref['object_quaternion']]
    actual = []
    for q in ref['raw_ik_ctrl']:
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        actual.append(data.geom_xpos[tips].copy())
    actual = np.asarray(actual)
    expected = ref['target_tips']
    errors = np.linalg.norm(actual-expected, axis=-1)*1000
    np.testing.assert_allclose(errors, ref['errors_m']*1000, atol=1e-6)
    fig = plt.figure(figsize=(14, 5), layout='constrained')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    rotation, position = ref['object_to_world_rotation'], ref['object_position']
    desired_local = (expected-position) @ rotation*1000
    actual_local = (actual-position) @ rotation*1000
    theta, height = np.meshgrid(np.linspace(0, 2*np.pi, 48), [-26.5, 26.5])
    ax.plot_surface(26.5*np.cos(theta), height, 26.5*np.sin(theta), color='#999999', alpha=.15)
    for i, (name, color) in enumerate((('Thumb', '#c45b37'), ('Index', '#257aa8'))):
        ax.plot(*desired_local[:, i].T, color=color, label=name+' target')
        ax.plot(*actual_local[:, i].T, color=color, linestyle='--', label=name+' IK')
        ax.scatter(*actual_local[-1, i], color=color, marker='x')
    ax.set(xlabel='Object X (mm)', ylabel='Object Y / cylinder axis (mm)', zlabel='Object Z (mm)',
           title='Object-relative contact-pad center correspondence')
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    for i, (name, color) in enumerate((('Thumb', '#c45b37'), ('Index', '#257aa8'))):
        ax.plot(ref['source_time'], errors[:, i], label=name, color=color)
    ax.set(xlabel='Processed source time (s)', ylabel='Euclidean pad-center error (mm)',
           title='IK fitting error before 1.5 s acquisition blend')
    ax.grid(alpha=.2)
    ax.legend()
    output = OUT/'source_analysis'
    output.mkdir(exist_ok=True)
    fig.savefig(output/'mapping_error.png', dpi=160)
    plt.close(fig)
    with (output/'mapping_error.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame', 'time_s', 'thumb_error_mm', 'index_error_mm'])
        writer.writerows(zip(ref['source_indices'], ref['source_time'], errors[:, 0], errors[:, 1]))
    print('Verified stored IK errors and saved', output/'mapping_error.png')


if __name__ == '__main__':
    main()
