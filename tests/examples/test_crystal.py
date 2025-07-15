"""Tests for `examples.crystal`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import numpy as onp

from examples import crystal

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)


class CrystalDipoleTest(unittest.TestCase):
    def test_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_internal_source(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
        )

        self.assertSequenceEqual(ex.shape, x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        with self.subTest("efield"):
            onp.testing.assert_allclose(
                [onp.mean(onp.abs(ex)), onp.mean(onp.abs(ey)), onp.mean(onp.abs(ez))],
                [8.311728e00, 1.293643e-06, 5.886467e00],
                rtol=1e-4,
            )
        with self.subTest("hfield"):
            onp.testing.assert_allclose(
                [onp.mean(onp.abs(hx)), onp.mean(onp.abs(hy)), onp.mean(onp.abs(hz))],
                [1.396684e-06, 7.920396e00, 1.485839e-06],
                rtol=1e-4,
            )


class CrystalGaussianBeamTest(unittest.TestCase):
    def test_CW_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_gaussian_beam(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
            wavelengths=crystal.WAVELENGTH,
        )

        self.assertSequenceEqual(ex.shape, (1,) + x.shape + z.shape + (1,))
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        with self.subTest("efield"):
            onp.testing.assert_allclose(
                [onp.mean(onp.abs(ex)), onp.mean(onp.abs(ey)), onp.mean(onp.abs(ez))],
                [4.830812e-01, 4.060432e-04, 3.208762e-01],
                rtol=1e-4,
                atol=1e-6,
            )
        with self.subTest("hfield"):
            onp.testing.assert_allclose(
                [onp.mean(onp.abs(hx)), onp.mean(onp.abs(hy)), onp.mean(onp.abs(hz))],
                [0.000666, 0.559154, 0.0008],
                rtol=1e-4,
                atol=1e-6,
            )

    def test_broadband_regression(self):
        (
            (ex, ey, ez),
            (hx, hy, hz),
            (x, y, z),
            (section_xy, section_xz, section_yz),
        ) = crystal.simulate_crystal_with_gaussian_beam(
            brillouin_grid_shape=(2, 3),
            resolution_fields=0.1,
            wavelengths=crystal.MULTIPLE_WAVELENGTHS,
        )

        self.assertSequenceEqual(
            ex.shape, crystal.MULTIPLE_WAVELENGTHS.shape + x.shape + z.shape + (1,)
        )
        self.assertSequenceEqual(ex.shape, ey.shape)
        self.assertSequenceEqual(ex.shape, ez.shape)
        self.assertSequenceEqual(ex.shape, hx.shape)
        self.assertSequenceEqual(ex.shape, hy.shape)
        self.assertSequenceEqual(ex.shape, hz.shape)

        wavelength_idx = 1
        with self.subTest("efield"):
            onp.testing.assert_allclose(
                [
                    onp.mean(onp.abs(ex[wavelength_idx, ...])),
                    onp.mean(onp.abs(ey[wavelength_idx, ...])),
                    onp.mean(onp.abs(ez[wavelength_idx, ...])),
                ],
                [4.830811e-01, 4.060428e-04, 3.208762e-01],
                rtol=1e-4,
            )
        with self.subTest("hfield"):
            onp.testing.assert_allclose(
                [
                    onp.mean(onp.abs(hx[wavelength_idx, ...])),
                    onp.mean(onp.abs(hy[wavelength_idx, ...])),
                    onp.mean(onp.abs(hz[wavelength_idx, ...])),
                ],
                [0.000666, 0.559154, 0.0008],
                rtol=1e-4,
                atol=1e-6,
            )
