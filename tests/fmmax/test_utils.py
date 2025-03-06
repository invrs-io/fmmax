"""Tests for `fmmax.utils`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import utils


class InterpolateTest(unittest.TestCase):
    @parameterized.expand(
        [
            (4.0, 2.0, 0.0, 2.0),
            (4.0, 2.0, 1.0, 4.0),
            (4.0, 2.0, 0.5, (jnp.sqrt(2) * 0.5 + jnp.sqrt(4) * 0.5) ** 2),
            (4.0 + 1.0j, 2.0, 0.0, 2.0),
            (4.0 + 1.0j, 2.0, 1.0, 4.0 + 1.0j),
            (4.0 + 1.0j, 2.0, 0.5, (jnp.sqrt(2) * 0.5 + jnp.sqrt(4 + 1.0j) * 0.5) ** 2),
        ]
    )
    def test_interpolated_matches_expected(self, p_solid, p_void, density, expected):
        result = utils.interpolate_permittivity(p_solid, p_void, density)
        onp.testing.assert_allclose(result, expected)
