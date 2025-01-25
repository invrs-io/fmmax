"""Tests for shapes of arrays returned by functions in `fmmax.fmm`.

Copyright (c) Martin F. Schubert
"""

import dataclasses
import unittest

import jax.numpy as jnp
from parameterized import parameterized

from fmmax import basis, fmm

PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(basis.X, basis.Y)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=4,
    truncation=basis.Truncation.CIRCULAR,
)

BATCH_SHAPES = [
    [(), (), ()],
    [(3,), (), (10, 1)],
    [(3,), (), (10, 3)],
    [(3,), (2, 1), (10, 3)],
]


class ShapesTest(unittest.TestCase):
    def _assert_shapes_match(self, a, b):
        for key in dataclasses.asdict(a).keys():
            la = a.__dict__[key]
            lb = b.__dict__[key]
            if not isinstance(la, jnp.ndarray):
                continue
            self.assertSequenceEqual(la.shape, lb.shape, msg=f"`{key}` shape mismatch.")

    @parameterized.expand(BATCH_SHAPES)
    def test_patterned_isotropic(
        self,
        wavelength_batch_shape,
        wavevector_batch_shape,
        permittivity_batch_shape,
    ):
        reference = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y),
            permittivity=jnp.ones(permittivity_batch_shape + (10, 10)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        self._assert_shapes_match(reference, result)

    @parameterized.expand(BATCH_SHAPES)
    def test_uniform_anisotropic(
        self,
        wavelength_batch_shape,
        wavevector_batch_shape,
        permittivity_batch_shape,
    ):
        reference = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        result = fmm.eigensolve_anisotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y),
            permittivity_xx=jnp.ones(permittivity_batch_shape + (1, 1)),
            permittivity_xy=jnp.zeros(permittivity_batch_shape + (1, 1)),
            permittivity_yx=jnp.zeros(permittivity_batch_shape + (1, 1)),
            permittivity_yy=jnp.ones(permittivity_batch_shape + (1, 1)),
            permittivity_zz=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        self._assert_shapes_match(reference, result)

    @parameterized.expand(BATCH_SHAPES)
    def test_patterned_anisotropic(
        self,
        wavelength_batch_shape,
        wavevector_batch_shape,
        permittivity_batch_shape,
    ):
        reference = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        result = fmm.eigensolve_anisotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y),
            permittivity_xx=jnp.ones(permittivity_batch_shape + (10, 10)),
            permittivity_xy=jnp.zeros(permittivity_batch_shape + (10, 10)),
            permittivity_yx=jnp.zeros(permittivity_batch_shape + (10, 10)),
            permittivity_yy=jnp.ones(permittivity_batch_shape + (10, 10)),
            permittivity_zz=jnp.ones(permittivity_batch_shape + (10, 10)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        self._assert_shapes_match(reference, result)

    @parameterized.expand(BATCH_SHAPES)
    def test_uniform_general_anisotropic(
        self,
        wavelength_batch_shape,
        wavevector_batch_shape,
        permittivity_batch_shape,
    ):
        reference = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        result = fmm.eigensolve_general_anisotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y),
            permittivity_xx=jnp.ones(permittivity_batch_shape + (1, 1)),
            permittivity_xy=jnp.zeros(permittivity_batch_shape + (1, 1)),
            permittivity_yx=jnp.zeros(permittivity_batch_shape + (1, 1)),
            permittivity_yy=jnp.ones(permittivity_batch_shape + (1, 1)),
            permittivity_zz=jnp.ones(permittivity_batch_shape + (1, 1)),
            permeability_xx=jnp.ones(permittivity_batch_shape + (1, 1)),
            permeability_xy=jnp.zeros(permittivity_batch_shape + (1, 1)),
            permeability_yx=jnp.zeros(permittivity_batch_shape + (1, 1)),
            permeability_yy=jnp.ones(permittivity_batch_shape + (1, 1)),
            permeability_zz=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        self._assert_shapes_match(reference, result)

    @parameterized.expand(BATCH_SHAPES)
    def test_patterned_general_anisotropic(
        self,
        wavelength_batch_shape,
        wavevector_batch_shape,
        permittivity_batch_shape,
    ):
        reference = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones(permittivity_batch_shape + (1, 1)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        result = fmm.eigensolve_general_anisotropic_media(
            wavelength=jnp.ones(wavelength_batch_shape),
            in_plane_wavevector=jnp.ones(wavevector_batch_shape + (2,)),
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y),
            permittivity_xx=jnp.ones(permittivity_batch_shape + (10, 10)),
            permittivity_xy=jnp.zeros(permittivity_batch_shape + (10, 10)),
            permittivity_yx=jnp.zeros(permittivity_batch_shape + (10, 10)),
            permittivity_yy=jnp.ones(permittivity_batch_shape + (10, 10)),
            permittivity_zz=jnp.ones(permittivity_batch_shape + (10, 10)),
            permeability_xx=jnp.ones(permittivity_batch_shape + (10, 10)),
            permeability_xy=jnp.zeros(permittivity_batch_shape + (10, 10)),
            permeability_yx=jnp.zeros(permittivity_batch_shape + (10, 10)),
            permeability_yy=jnp.ones(permittivity_batch_shape + (10, 10)),
            permeability_zz=jnp.ones(permittivity_batch_shape + (10, 10)),
            formulation=fmm.Formulation.FFT,
            expansion=EXPANSION,
        )
        self._assert_shapes_match(reference, result)
