"""Prove the batched contact writes preserve the original dense matrix."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from orca_solver_runtime import cone_indices, cone_matrix


class SolverRuntimeTests(unittest.TestCase):
    def test_disjoint_mixed_dimensions_exactly_match_original(self):
        dimensions = np.array([3, 4, 6, 4])
        addresses = np.array([7, 10, 14, 20])
        rng = np.random.default_rng(42)
        diagonal = jp.asarray(rng.normal(size=30), dtype=jp.float32)
        blocks = jp.asarray(rng.normal(size=(4, 6, 6)), dtype=jp.float32)
        def original(diag, h):
            matrix = jp.diag(diag)
            for i, (dim, address) in enumerate(zip(dimensions, addresses)):
                matrix = matrix.at[address:address+dim, address:address+dim].add(h[i, :dim, :dim])
            return matrix
        actual = jax.jit(lambda d, h: cone_matrix(d, h, addresses, dimensions))(diagonal, blocks)
        np.testing.assert_array_equal(actual, jax.jit(original)(diagonal, blocks))
        np.testing.assert_array_equal(
            jax.vmap(lambda d, h: cone_matrix(d, h, addresses, dimensions))(
                jp.stack([diagonal, diagonal * 2]), jp.stack([blocks, blocks * 2])),
            jax.vmap(original)(jp.stack([diagonal, diagonal * 2]), jp.stack([blocks, blocks * 2])))

    def test_rejects_overlap_invalid_and_overflow(self):
        for addresses, dimensions in (([0, 2], [4, 4]), ([-1], [4]), ([0], [5])):
            with self.assertRaises(ValueError):
                cone_indices(addresses, dimensions)
        with self.assertRaises(ValueError):
            cone_matrix(jp.zeros(3), jp.zeros((1, 6, 6)), np.array([0]), np.array([4]))


if __name__ == "__main__":
    unittest.main()
