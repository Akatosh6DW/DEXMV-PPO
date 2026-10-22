"""Gate reconstructed collision geometry before any new training experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices, contact_state
from fit_orca_collision import BASE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(model, seed):
    data = mujoco.MjData(model)
    qids, cube, _ = model_indices(model)
    rng = np.random.default_rng(seed)
    ctrl = np.array([-.25]+[0.]*16)
    data.ctrl[:] = ctrl
    data.qpos[qids] = np.clip(ctrl+rng.normal(0,.002,17), *model.actuator_ctrlrange.T)
    spawn = np.array([.055,.025,.265])+rng.uniform(-1,1,3)*[.002,.002,0.]
    data.qpos[cube:cube+7] = [*spawn,1,0,0,0]
    data.mocap_pos[0] = spawn-[0,0,.022]
    mujoco.mj_forward(model,data)
    return data, spawn


def cube_hand_penetration(model,data):
    cube = model.geom("cube").id
    depths = []
    for c in data.contact[:data.ncon]:
        if cube in (c.geom1,c.geom2):
            other = c.geom2 if c.geom1==cube else c.geom1
            if model.geom_contype[other]==1:
                depths.append(max(-float(c.dist),0))
    return max(depths,default=0)


def native_smoke(model, reference, seed):
    data, spawn = initialize(model,seed)
    support = data.mocap_pos[0].copy()
    qids,cube,_ = model_indices(model)
    records=[]
    for step in range(800):
        t=(step+1)*.05
        desired=reference[min(step,len(reference)-1)]
        data.ctrl[:]=np.clip(desired,data.ctrl-.1,data.ctrl+.1)
        data.mocap_pos[0]=support-[0,0,np.clip((t-2)*.2,0,.2)]
        for _ in range(5):
            mujoco.mj_step(model,data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        fingers, forbidden, force = contact_state(model,data)
        records.append(dict(time=t,cube_z=float(data.qpos[cube+2]),
                            penetration_m=cube_hand_penetration(model,data), force_n=force,
                            max_qvel=float(np.max(np.abs(data.qvel))),
                            fingers=fingers.tolist(),forbidden=bool(forbidden)))
        if data.qpos[cube+2] < spawn[2]-.055:
            break
    return dict(seed=seed,frames=len(records),max_penetration_m=max(r['penetration_m'] for r in records),
                max_force_n=max(r['force_n'] for r in records),
                max_qvel=max(r['max_qvel'] for r in records), records=records)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.candidate/'cube.xml'
    geometry=json.loads((args.candidate/'geometry_report.json').read_text())
    assert digest(source)==geometry['model_sha256']
    old=mujoco.MjModel.from_xml_path(str(BASE))
    model=mujoco.MjModel.from_xml_path(str(source))
    failures=[]
    for name,entry in geometry['bodies'].items():
        q=entry['new']
        structural='Carpals' in name or 'T-TP-R' in name or 'R-T-AP' in name
        limit=5. if structural else 3.5
        if q['uncovered_p95_mm']>limit or q['proxy_to_mesh_p95_mm']>limit:
            failures.append(f'surface error {name}')
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                  'jnt_pos','jnt_axis','jnt_range','actuator_ctrlrange','actuator_gainprm',
                  'actuator_biasprm','actuator_forcerange','dof_damping','dof_armature'):
        np.testing.assert_allclose(getattr(old,field),getattr(model,field),rtol=1e-8,atol=1e-10)
    assert (model.nq,model.nv,model.nu)==(old.nq,old.nv,old.nu)
    for name in ('cube','cube_support','floor'):
        a,b=old.geom(name),model.geom(name)
        for field in ('size','friction','contype','conaffinity'):
            np.testing.assert_allclose(getattr(a,field),getattr(b,field))
    reset=[]
    for seed in range(64):
        olddata,_=initialize(old,seed)
        newdata,_=initialize(model,seed)
        reset.append(dict(seed=seed,old_depth_m=cube_hand_penetration(old,olddata),
                          new_depth_m=cube_hand_penetration(model,newdata)))
    if max(r['new_depth_m'] for r in reset)>.003:
        failures.append('reset penetration above 3 mm')
    reference_path=BASE.parents[1]/'../orca30/dexmv_slow_acquisition_o21/reference.npz'
    with np.load(reference_path) as z:
        reference=z['ctrl']
    smoke=[native_smoke(model,reference,seed) for seed in (0,1,2)]
    if any(r['max_penetration_m']>.005 or r['max_qvel']>1000 for r in smoke):
        failures.append('native motion instability or penetration above 5 mm')
    result=dict(geometry_and_native_pass=not failures,failures=failures,model=str(source.resolve()),
                model_sha256=digest(source),reset=reset,native_smoke=smoke,
                surface_limits_mm=dict(finger=3.5,structural_palm_thumb_base=5.),
                note='Provisional numerical geometry gate, not a real-hand accuracy certificate; MJX smoke and visual inspection still required')
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('reset','native_smoke')},indent=2))
    print('Reset max depth',max(r['new_depth_m'] for r in reset),flush=True)
    for r in smoke:
        print({k:v for k,v in r.items() if k!='records'},flush=True)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
