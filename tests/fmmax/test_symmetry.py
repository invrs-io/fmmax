"""Tests that fields have the expected symmetry.

Copyright (c) 2025 invrs.io LLC
"""

import unittest

import jax.numpy as jnp
import numpy as onp

from fmmax import basis, fields, fmm, scattering, sources


class SymmetryTest(unittest.TestCase):
    def test_fields_have_expected_symmetry(self):
        # Test that a symmetric permittivity distribution and a centered source yield
        # fields with corresponding symmetry.
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        x, y = basis.unit_cell_coordinates(
            primitive_lattice_vectors=primitive_lattice_vectors,
            shape=(10, 10),
            num_unit_cells=(1, 1),
        )
        permittivity = jnp.where(
            jnp.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) < 0.3, 2.0, 1.0
        )

        # Ensure that permittivity is symmetric.
        onp.testing.assert_array_equal(permittivity, onp.rot90(permittivity, 1))
        onp.testing.assert_array_equal(permittivity, onp.rot90(permittivity, 2))
        onp.testing.assert_array_equal(permittivity, onp.rot90(permittivity, 3))

        in_plane_wavevector = jnp.zeros((2,))
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=64,
        )

        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.55),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

        s_matrix = scattering.stack_s_matrix([solve_result], layer_thicknesses=[1.0])
        dipole = sources.dirac_delta_source(
            location=jnp.asarray([[0.5, 0.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )
        zeros = jnp.zeros_like(dipole)
        jx = jnp.concatenate([dipole, zeros, zeros], axis=-1)
        jy = jnp.concatenate([zeros, dipole, zeros], axis=-1)
        jz = jnp.concatenate([zeros, zeros, dipole], axis=-1)

        (
            bwd_amplitude_0_end,
            fwd_amplitude_before_start,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            bwd_amplitude_after_end,
            fwd_amplitude_N_end,
        ) = sources.amplitudes_for_source(
            jx=jx,
            jy=jy,
            jz=jz,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )

        electric_field, magnetic_field = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_0_end),
            backward_amplitude=bwd_amplitude_0_end,
            layer_solve_result=solve_result,
        )

        (ex, ey, ez), (hx, hy, hz), (x, y) = fields.fields_on_grid(
            electric_field=electric_field,
            magnetic_field=magnetic_field,
            layer_solve_result=solve_result,
            shape=permittivity.shape,
        )

        # Fields for x-oriented dipole are 180 degree rotation symmetric
        onp.testing.assert_allclose(ex[:, :, 0], onp.rot90(ex[:, :, 0], 2), rtol=2e-3)

        # Fields for y-oriented dipole are 180 degree rotation symmetric
        onp.testing.assert_allclose(ey[:, :, 1], onp.rot90(ey[:, :, 1], 2), rtol=2e-3)

        # Fields for z-oriented dipole are 90 degree rotation symmetric
        onp.testing.assert_allclose(ez[:, :, 2], onp.rot90(ez[:, :, 2], 1), rtol=2e-3)
        onp.testing.assert_allclose(ez[:, :, 2], onp.rot90(ez[:, :, 2], 2), rtol=2e-3)
        onp.testing.assert_allclose(ez[:, :, 2], onp.rot90(ez[:, :, 2], 3), rtol=2e-3)
