"""Normal loads for uniform condim=4 contacts in the matched Orca task."""

import numpy as np


def normal_loads(address, efc_force, cone="pyramidal", xp=np):
    if cone not in ("pyramidal", "elliptic"):
        raise ValueError("Unsupported friction cone")
    address = xp.asarray(address)
    if len(efc_force) == 0:
        return xp.zeros(address.shape, dtype=xp.float32)
    width = 6 if cone == "pyramidal" else 4
    indices = address[:, None] + xp.arange(width)[None, :]
    values = xp.asarray(efc_force)[xp.clip(indices, 0, len(efc_force) - 1)]
    valid = (address >= 0) & (address + width <= len(efc_force))
    normal = xp.sum(values, axis=1) if cone == "pyramidal" else values[:, 0]
    return xp.where(valid, xp.maximum(normal, 0.), 0.)
