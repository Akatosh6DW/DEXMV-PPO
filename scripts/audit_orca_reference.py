"""Separate raw IK fitting error from reference blending and physical success."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def summarize(errors):
    combined=np.linalg.norm(errors.reshape(len(errors),-1),axis=1)*1000
    per_tip=np.linalg.norm(errors,axis=-1)*1000
    return dict(aggregate_mean_mm=float(combined.mean()),aggregate_max_mm=float(combined.max()),
                aggregate_first_mm=float(combined[0]),aggregate_last_mm=float(combined[-1]),
                per_tip_mean_mm=per_tip.mean(axis=0).tolist(),
                per_tip_last_mm=per_tip[-1].tolist(),worst_frame=int(np.argmax(combined)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data=mujoco.MjData(model)
    qids=model.jnt_qposadr[model.actuator_trnid[:,0]]
    tips=[model.geom(f'orca_{f}_tip_ball_proxy').id for f in ('thumb','index','middle')]
    cubeq=model.joint('cube_freejoint').qposadr[0]
    data.qpos[cubeq:cubeq+7]=[.055,.025,.265,1,0,0,0]
    def positions(controls):
        points=[]
        for q in controls:
            data.qpos[qids]=q
            data.ctrl[:]=q
            mujoco.mj_kinematics(model,data)
            points.append(data.geom_xpos[tips].copy())
        return np.asarray(points)
    with np.load(args.reference) as z:
        raw=z['raw_ik_ctrl'];targets=z['target_tips'];reference=z['ctrl'];times=z['time']
    raw_error=positions(raw)-targets
    target_resampled=np.stack([np.interp(times/times[-1],np.linspace(0,1,len(targets)),
                                           targets[:,i,j]) for i in range(3) for j in range(3)],axis=1).reshape(-1,3,3)
    blend_error=positions(reference)-target_resampled
    low,high=model.actuator_ctrlrange.T
    limit=((raw-low)<.001)|((high-raw)<.001)
    result=dict(model=str(args.model),reference=str(args.reference),
                model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                raw_ik=summarize(raw_error),blended_reference=summarize(blend_error),
                thirds=[summarize(e) for e in np.array_split(raw_error,3)],
                raw_limit_fraction_by_joint={model.actuator(i).name:float(limit[:,i].mean()) for i in range(model.nu)},
                reference_change_rad=np.ptp(reference,axis=0).tolist(),
                tip_order=['thumb','index','middle'],
                note='Kinematic marker-target errors, not contact-surface or video ground-truth errors. '
                     'Targets themselves use an old geometry-calibrated endpoint. Blended reference intentionally differs from raw IK.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output/'errors.npz',raw_error=raw_error,blended_error=blend_error)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
