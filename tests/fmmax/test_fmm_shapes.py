"""Tests for shapes of arrays returned by functions in `fmmax.fmm`.

Copyright (c) Martin F. Schubert
"""

import unittest

import jax.numpy as jnp
import numpy as onp
from jax import tree_util
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
]


class ShapesTest(unittest.TestCase):
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
        for a, b in zip(tree_util.tree_leaves(result), tree_util.tree_leaves(reference)):
            self.assertSequenceEqual(a.shape, b.shape)
        
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
        for a, b in zip(tree_util.tree_leaves(result), tree_util.tree_leaves(reference)):
            self.assertSequenceEqual(a.shape, b.shape)

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
        for a, b in zip(tree_util.tree_leaves(result), tree_util.tree_leaves(reference)):
            self.assertSequenceEqual(a.shape, b.shape)

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
        for a, b in zip(tree_util.tree_leaves(result), tree_util.tree_leaves(reference)):
            self.assertSequenceEqual(a.shape, b.shape)

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
        for a, b in zip(tree_util.tree_leaves(result), tree_util.tree_leaves(reference)):
            self.assertSequenceEqual(a.shape, b.shape)
