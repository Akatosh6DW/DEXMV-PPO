"""Geometric contact test for distal, front-facing Orca thumb/index patches."""

import mujoco
import numpy as np

FINGERS = ("thumb", "index")
MIN_Z = (.015, .024)


def is_distal_pad(name, point, normal, finger):
    """Local +Z runs along the phalanx; -X faces its flexion/palmar side."""
    return (name.startswith(f"orca_{FINGERS[finger]}_tip_")
            and point[2] >= MIN_Z[finger]
            and -normal[0] >= .7 and abs(normal[1]) <= .5)


def pad_contacts(model, data):
    """Return good loads, non-pad loads and smooth geometric reach potentials."""
    cube = model.geom("cube").id
    good, bad = np.zeros(2), np.zeros(2)
    for ci, contact in enumerate(data.contact[:data.ncon]):
        if cube not in contact.geom or contact.dist > 0:
            continue
        gid = contact.geom2 if contact.geom1 == cube else contact.geom1
        name = model.geom(int(gid)).name
        finger = next((i for i, f in enumerate(FINGERS) if name.startswith(f"orca_{f}_")), None)
        if finger is None:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, force)
        if force[0] <= 1e-5:
            continue
        body = model.geom_bodyid[gid]
        rotation = data.xmat[body].reshape(3, 3)
        point = (contact.pos - data.xpos[body]) @ rotation
        normal = (contact.frame[:3] * (1 if contact.geom1 == gid else -1)) @ rotation
        (good if is_distal_pad(name, point, normal, finger) else bad)[finger] += force[0]
    potentials = []
    for name in ("orca_thumb_tip_proxy_fit_02", "orca_index_tip_proxy_fit_03"):
        gid = model.geom(name).id
        delta = data.geom_xpos[cube] - data.geom_xpos[gid]
        distance = np.linalg.norm(delta)
        facing = -data.xmat[model.geom_bodyid[gid]].reshape(3, 3)[:, 0]
        alignment = max(0., float(facing @ delta / max(distance, 1e-9)))
        potentials.append(alignment ** 2 * np.exp(-abs(distance - .035) / .025))
    return good, bad, np.array(potentials)
