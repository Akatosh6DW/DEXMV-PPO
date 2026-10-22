"""Cross-check processed can coordinates using the original MuJoCo 2 engine."""

import argparse
import io
import json
from pathlib import Path
import pickle
import zipfile

import mujoco_py
import numpy as np


class ArraysOnly(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Unexpected external global')
        return allowed[module, name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.source/'dexmv_demonstrations.zip') as archive:
        demos = ArraysOnly(io.BytesIO(archive.read('demonstrations/relocate-tomato_soup_can.pkl'))).load()
    model = mujoco_py.load_model_from_path(str((args.source/'relocate_temp.xml').resolve()))
    sim = mujoco_py.MjSim(model)
    obj = model.body_name2id('tomato_soup_can_0')
    sites = [model.site_name2id('S_'+prefix+'tip') for prefix in ('th', 'ff', 'mf', 'rf', 'lf')]
    rows = []
    for sequence in (0, 77, 96):
        demo = demos['seq_%d/multi_retargeting_v3.pkl' % sequence]
        for field in ('body_pos', 'body_quat', 'site_pos', 'site_quat'):
            getattr(model, field)[:] = demo['model_data'][0][field]
        with np.load(args.source/'processed'/('seq_%03d.npz' % sequence)) as packet:
            for frame in (0, len(demo['sim_data'])//2, len(demo['sim_data'])-1):
                sim.data.qpos[:] = demo['sim_data'][frame]['qpos']
                sim.forward()
                rotation = sim.data.body_xmat[obj].reshape(3, 3)
                local = (sim.data.site_xpos[sites]-sim.data.body_xpos[obj]) @ rotation
                error = float(np.max(np.abs(local-packet['fingertips_object_local'][frame])))
                if error > 1e-8:
                    raise AssertionError('MuJoCo version/frame-coordinate mismatch: %g' % error)
                rows.append(dict(sequence=sequence, frame=frame, max_error_m=error))
    report = dict(passed=True, engine='Original DexMV MuJoCo 2.1 versus native MuJoCo 3 FK',
                  frames=rows, tolerance_m=1e-8)
    (args.source/'fk_crosscheck.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
