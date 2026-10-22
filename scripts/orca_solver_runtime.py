"""Opt-in vectorized assembly of MJX's existing elliptic cone Hessian.

The equations follow MuJoCo MJX (Apache-2.0). Only disjoint block writes are
batched; the dense matrix and subsequent multiplications remain unchanged.
No installed dependency is edited. Installation affects this process only.
"""

import hashlib
import inspect

import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import solver, support
from mujoco.mjx._src.types import ConeType, SolverType

_ORIGINAL = solver._update_gradient


def cone_indices(addresses, dimensions):
    rows, cols, blocks, local_rows, local_cols = [], [], [], [], []
    for i, (address, dimension) in enumerate(zip(addresses, dimensions)):
        if address < 0 or dimension not in (3, 4, 6):
            raise ValueError("Unexpected elliptic contact layout")
        for row in range(dimension):
            for col in range(dimension):
                rows.append(address + row)
                cols.append(address + col)
                blocks.append(i)
                local_rows.append(row)
                local_cols.append(col)
    if len(set(zip(rows, cols))) != len(rows):
        raise ValueError("Overlapping blocks cannot use unique scatter indices")
    return tuple(np.asarray(x, dtype=np.int32) for x in (rows, cols, blocks, local_rows, local_cols))


def cone_matrix(diagonal, blocks, addresses, dimensions):
    rows, cols, block, local_row, local_col = cone_indices(addresses, dimensions)
    if len(rows) and (max(rows.max(), cols.max()) >= len(diagonal)):
        raise ValueError("Contact block exceeds constraint matrix")
    return jp.diag(diagonal).at[rows, cols].add(
        blocks[block, local_row, local_col], unique_indices=True)


def vectorized_gradient(model, data, context):
    if model.opt.solver != SolverType.NEWTON or model.opt.cone != ConeType.ELLIPTIC:
        return _ORIGINAL(model, data, context)
    contact = data._impl.contact
    selected = contact.dim > 1
    cm = cone_matrix(data._impl.efc_D * context.active, context.h,
                     contact.efc_address[selected], contact.dim[selected])
    jacobian = data._impl.efc_J
    hessian = jacobian.T @ cm @ jacobian
    hessian = support.full_m(model, data) + hessian
    symmetric = (hessian + hessian.T) * .5
    gradient = context.Ma - data.qfrc_smooth - context.qfrc_constraint
    factor = jax.scipy.linalg.cho_factor(symmetric)
    preconditioned = jax.scipy.linalg.cho_solve(factor, gradient)
    return context.replace(grad=gradient, Mgrad=preconditioned)


def install():
    source = inspect.getsource(_ORIGINAL)
    expected = ("cm = jp.diag(d._impl.efc_D * ctx.active)",
                "cm = cm.at[addr : addr + condim, addr : addr + condim].add(h_cone)",
                "h = d._impl.efc_J.T @ cm @ d._impl.efc_J")
    if not all(fragment in source for fragment in expected):
        raise RuntimeError("MJX gradient implementation changed; re-audit before enabling optimization")
    solver._update_gradient = vectorized_gradient
    return dict(kind="vectorized_disjoint_elliptic_hessian_writes",
                original_gradient_sha256=hashlib.sha256(source.encode()).hexdigest(),
                retained_dense_matrix=True, contact_truncation=False)
