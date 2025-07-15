"""Tests for `examples.metal_dipole`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import numpy as onp

from examples import metal_dipole


class MetalDipoleTest(unittest.TestCase):
    def test_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
        ) = metal_dipole.simulate_metal_dipole(
            approximate_num_terms=200,
            grid_spacing_fields=0.1,
        )

        self.assertSequenceEqual(ex.shape, x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        with self.subTest("ex"):
            onp.testing.assert_allclose(onp.mean(onp.abs(ex)), 0.0, atol=1e-4)
        with self.subTest("ey"):
            onp.testing.assert_allclose(onp.mean(onp.abs(ey)), 14.12513, atol=1e-3)
        with self.subTest("ez"):
            onp.testing.assert_allclose(onp.mean(onp.abs(ez)), 0.0, atol=1e-4)
        with self.subTest("hx"):
            onp.testing.assert_allclose(onp.mean(onp.abs(hx)), 11.87493, atol=1e-3)
        with self.subTest("hy"):
            onp.testing.assert_allclose(onp.mean(onp.abs(hy)), 0.0, atol=1e-4)
        with self.subTest("hz"):
            onp.testing.assert_allclose(onp.mean(onp.abs(hz)), 9.952436, atol=1e-4)
