"""Recompute physical failure and slip diagnostics from saved rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np


def diagnose(path):
    with np.load(path) as z:
        t=dict(z)
    active=t['active'].astype(bool)
    released=active&(t['released']>0)
    paired=(t['thumb']>0)&(t['contacts']>=2)
    clean=released&paired&(t['forbidden']==0)
    stable=clean&(t['cube_z']>t['spawn_z']-.01)
    speed=t.get('cube_speed',np.linalg.norm(t['qvel'][...,-6:-3],axis=-1))
    def stats(values):
        return dict(mean=float(np.mean(values)),p50=float(np.quantile(values,.5)),
                    p95=float(np.quantile(values,.95))) if len(values) else None
    valid_den=max(int(released.sum()),1)
    result=dict(path=str(path),episodes=len(active),active_seconds=stats(active.sum(axis=1)*.05),
                released_forbidden_fraction=float(np.sum(released&(t['forbidden']>0))/valid_den),
                released_unpaired_fraction=float(np.sum(released&~paired)/valid_den),
                clean_below_height_fraction=float(np.sum(clean&(t['cube_z']<=t['spawn_z']-.01))/valid_den),
                stable_speed_mps=stats(speed[stable]),released_speed_mps=stats(speed[released]),
                stable_frame_fraction=float(stable.sum()/valid_den))
    if 'component_correction' in t:
        result['correction_squared_fraction']=stats(t['component_correction'][active])
    result['episodes_detail']=[]
    best_hold_rotations=[]
    best_hold_displacements=[]
    for i in range(len(active)):
        idx=np.flatnonzero(active[i])
        if not len(idx):
            raise ValueError('Episode has no active transition')
        last=int(idx[-1])
        released_idx=np.flatnonzero(released[i])
        start=int(released_idx[0]) if len(released_idx) else None
        x0=t['qpos'][i,start,-7:-4] if start is not None else None
        xlast=t['qpos'][i,last,-7:-4]
        hold_end=int(np.argmax(np.where(active[i], t['hold'][i], -1)))
        held=float(t['hold'][i,hold_end])>0 and start is not None
        displacement=None
        rotation=None
        if held:
            displacement=(t['qpos'][i,hold_end,-7:-4]-x0)*1000
            qa=t['qpos'][i,start,-4:]
            qb=t['qpos'][i,hold_end,-4:]
            cosine=np.abs(np.dot(qa,qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)))
            rotation=float(np.rad2deg(2*np.arccos(np.clip(cosine,0,1))))
            best_hold_rotations.append(rotation)
            best_hold_displacements.append(displacement)
        result['episodes_detail'].append(dict(index=i,last_active_step=last,
            max_hold_sec=float(t['hold'][i].max()),final_hold_sec=float(t['hold'][i,-1]),
            displacement_since_release_mm=((xlast-x0)*1000).tolist() if x0 is not None else None,
            best_hold_end_step=hold_end if held else None,
            best_hold_end_displacement_mm=displacement.tolist() if held else None,
            best_hold_end_rotation_deg=rotation,
            final_alive=bool(t['alive'][i,-1]),last_active_contacts=float(t['contacts'][i,last]),
            last_active_forbidden=float(t['forbidden'][i,last]),
            last_active_height_margin_mm=float((t['cube_z'][i,last]-t['spawn_z'][i,last]+.01)*1000)))
    result['rotation_at_best_hold_end_deg']=stats(best_hold_rotations)
    result['median_displacement_at_best_hold_end_mm']=(np.median(best_hold_displacements,axis=0).tolist()
                                                      if best_hold_displacements else None)
    if 'finger_normal_force' in t:
        result['released_force_n_by_finger']=[stats(t['finger_normal_force'][...,i][released]) for i in range(5)]
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('trace',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=diagnose(args.trace)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='episodes_detail'},indent=2))
