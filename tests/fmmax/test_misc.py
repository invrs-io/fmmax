"""Tests for `fmmax._misc`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp

from fmmax import misc


class DiagTest(unittest.TestCase):
    def test_diag_matches_expected(self):
        shapes = ((5,), (2, 5), (9, 1, 8))
        for shape in shapes:
            with self.subTest(shape):
                v = jax.random.uniform(jax.random.PRNGKey(0), shape)
                d = misc.diag(v)
                expected = jnp.zeros(shape + (shape[-1],))
                for ind in itertools.product(*[range(dim) for dim in shape[:-1]]):
                    expected = expected.at[ind].set(jnp.diag(v[ind]))
                onp.testing.assert_allclose(d, expected)


class MatrixAdjointTest(unittest.TestCase):
    def test_adjoint_matches_expected(self):
        shapes = ((5, 5), (2, 5, 5), (9, 1, 8, 8))
        for shape in shapes:
            with self.subTest(shape):
                m = jax.random.uniform(jax.random.PRNGKey(0), shape)
                ma = misc.matrix_adjoint(m)
                expected = jnp.zeros(shape)
                for ind in itertools.product(*[range(dim) for dim in shape[:-2]]):
                    expected = expected.at[ind].set(jnp.conj(m[ind]).T)
                onp.testing.assert_allclose(ma, expected)


class BatchCompatibleTest(unittest.TestCase):
    def test_value_matches_expected(self):
        shapes_and_expected = [
            ([(1, 2), (2,)], True),
            ([(1, 2), (3,)], False),
            ([(1, 2), (4, 2, 2)], True),
            ([(1, 2), ()], True),
        ]
        for shapes, expected in shapes_and_expected:
            with self.subTest(shapes):
                self.assertEqual(misc.batch_compatible_shapes(*shapes), expected)


class AtLeastNDTest(unittest.TestCase):
    def test_shape_matches_expected(self):
        shapes_n_expected = [
            [(2, 1), 1, (2, 1)],
            [(2, 1), 2, (2, 1)],
            [(2, 1), 3, (1, 2, 1)],
            [(2, 1), 4, (1, 1, 2, 1)],
        ]
        for shape, n, expected_shape in shapes_n_expected:
            with self.subTest(n):
                x = onp.zeros(shape)
                self.assertSequenceEqual(misc.atleast_nd(x, n).shape, expected_shape)
