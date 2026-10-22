"""Independent paired geometry controls, not learning-policy comparisons."""

from dataclasses import replace
import json
import xml.etree.ElementTree as ET

import numpy as np

from search_s10_alignment import conditions
from s09_suspension import SuspendedCylinder, Suspension
from s10_mobile_base import OUT, INPUT, LIFT, MobileCylinder, rotation, run_episode
from retarget_s09_reference import digest


class FixedReference(SuspendedCylinder):
    def __init__(self, template, bank_mode=False):
        root=ET.fromstring((INPUT/"model.xml").read_text())
        mount=root.find(".//body[@name='right_mount']")
        mount.set("pos"," ".join(map(str,np.fromstring(mount.get("pos"),sep=" ")+LIFT)))
        super().__init__(ET.tostring(root,encoding="unicode"),replace(Suspension(),release_sec=5.,episode_sec=42.))
        with np.load(INPUT/"reference.npz") as z:
            self.bank=dict(z)
        self.template,self.bank_mode=template,bank_mode

    def reset(self, position, quat, initial_ctrl=None, velocity=None):
        if self.bank_mode:
            axis=rotation(quat).apply([0.,-1.,0.])
            axes=np.array([rotation(q).apply([0.,-1.,0.]) for q in self.bank["object_quaternion"]])
            self.template=int(np.argmax(np.abs(axes@axis)))
        return super().reset(position,quat,initial_ctrl,velocity)

    def reference(self,time):
        return np.array([np.interp(max(0.,time-2),self.bank["time"],self.bank["ctrl"][self.template,:,j])
                         for j in range(17)])

    def step(self,desired):
        return dict(super().step(desired),base_position_error_m=0.,base_rotation_error_rad=0.)


def main():
    out=OUT/"alignment_controls"
    out.mkdir(parents=True,exist_ok=False)
    selection=json.loads((OUT/"inputs/selection_v2.json").read_text())
    plan=dict(seeds=list(range(230000,230128)),selection_sha256=digest(OUT/"inputs/selection_v2.json"),
              groups=["fixed_template","fixed_bank","aligned_preplaced","aligned_moving"],
              meaning="Paired fresh 128 states per group; same object, height, release5s and 42s horizon",
              warning="Not the JAX policy evaluation seeds; no cross-table subtraction")
    (out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    rows=[]
    for mode in plan["groups"]:
        sim=(FixedReference(selection["template"],bank_mode=mode=="fixed_bank") if mode.startswith("fixed")
             else MobileCylinder(template=selection["template"],rule=selection["rule"],roll=selection["roll"],
                                 prealigned=mode=="aligned_preplaced"))
        episodes=[]
        for seed,pos,q,vel in conditions(230000,128):
            row,trace=run_episode(sim,pos,q,vel,keep_trace=seed==230000)
            episodes.append(dict(seed=seed,**row))
            (out/f"{mode}_episodes.json").write_text(json.dumps(episodes,indent=2)+"\n")
            if trace is not None:
                np.savez_compressed(out/f"{mode}_first_trace.npz",**trace)
            if (seed-230000+1)%16==0:
                print(mode,seed-230000+1,sum(r["success30"] for r in episodes),flush=True)
        summary=dict(group=mode,count=len(episodes),successes=sum(r["success30"] for r in episodes),
                     success30=float(np.mean([r["success30"] for r in episodes])),
                     max_force=max(r["peak_force_n"] for r in episodes),
                     max_penetration=max(r["max_released_penetration_m"] for r in episodes))
        rows.append(summary)
        (out/"results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(json.dumps(summary),flush=True)
    (out/"summary.json").write_text(json.dumps(dict(status="completed",groups=rows),indent=2)+"\n")


if __name__=="__main__":
    main()
