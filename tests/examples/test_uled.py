"""Tests for `examples.uled`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp

from examples import uled
from fmmax import basis, fmm

SIM_CONFIG_KWARGS = {
    "formulation": fmm.Formulation.POL,
    "truncation": basis.Truncation.CIRCULAR,
    "approximate_num_terms": 400,
}


class MicroLedTest(unittest.TestCase):
    def test_regression(self):
        # Checks that results match reference values. This helps protect against
        # regressions in accuracy of the simulator.
        extraction_efficiency, _, efields, hfields, _ = uled.simulate_uled(
            resolution=25,
            resolution_fields=25,
            dipole_fwhm=0,
            dipole_y_offset=(0,),
            **SIM_CONFIG_KWARGS,
        )
        with self.subTest("extraction efficiency"):
            onp.testing.assert_allclose(
                extraction_efficiency,
                [0.496093, 0.496177, 0.247206],
                atol=1e-3,
            )

        with self.subTest("efields"):
            self.assertSequenceEqual(efields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(efields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [64.48044, 8.389313, 2.740909],
                    [8.386348, 64.499756, 2.740864],
                    [15.824419, 15.819216, 8.312174],
                ],
                rtol=1e-3,
            )

        with self.subTest("hfields"):
            self.assertSequenceEqual(hfields.shape, (3, 1, 1, 56, 56, 48, 3))
            onp.testing.assert_allclose(
                jnp.mean(jnp.abs(hfields) ** 2, axis=(1, 2, 3, 4, 5)),
                [
                    [35.596813, 411.6974, 33.963284],
                    [411.5761, 35.610237, 33.963463],
                    [141.883, 141.92323, 1.209434],
                ],
                rtol=1e-3,
            )
