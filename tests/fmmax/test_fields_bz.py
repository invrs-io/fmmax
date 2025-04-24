"""Tests for `fields` involving Brillouin zone integration.

Copyright (c) Martin F. Schubert
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fields, fmm, scattering, sources

jax.config.update("jax_enable_x64", True)


PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(u=basis.X, v=basis.Y)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=100,
    truncation=basis.Truncation.CIRCULAR,
)


class BZIntegratedFieldsTest(unittest.TestCase):
    @parameterized.expand([[0.314], [(0.314, 0.628)]])
    def test_fields_on_grid_match_expected(self, wavelength):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(3, 3),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        (
            _,
            _,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            _,
            _,
        ) = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )
        amplitudes_interior = fields.stack_amplitudes_interior_with_source(
            s_matrices_interior_before_source=((s_matrix, s_matrix),),
            s_matrices_interior_after_source=((s_matrix, s_matrix),),
            backward_amplitude_before_end=bwd_amplitude_before_end,
            forward_amplitude_after_start=fwd_amplitude_after_start,
        )

        # Manually carry out Brillouin zone integration.
        efield, hfield, _ = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            grid_shape=(30, 30),
            num_unit_cells=(3, 3),
        )
        efield_expected = [onp.sum(f, axis=(0, 1)) for f in efield]
        hfield_expected = [onp.sum(f, axis=(0, 1)) for f in hfield]

        # Automatically perform Brillouin zone integration.
        efield_integrated, hfield_integrated, _ = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            grid_shape=(30, 30),
            brillouin_grid_axes=(0, 1),
        )

        onp.testing.assert_allclose(efield_integrated, efield_expected)
        onp.testing.assert_allclose(hfield_integrated, hfield_expected)

    @parameterized.expand([[0.314], [(0.314, 0.628)]])
    def test_fields_on_coordinates_match_expected(self, wavelength):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(3, 3),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        (
            _,
            _,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            _,
            _,
        ) = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )
        amplitudes_interior = fields.stack_amplitudes_interior_with_source(
            s_matrices_interior_before_source=((s_matrix, s_matrix),),
            s_matrices_interior_after_source=((s_matrix, s_matrix),),
            backward_amplitude_before_end=bwd_amplitude_before_end,
            forward_amplitude_after_start=fwd_amplitude_after_start,
        )

        # Manually carry out Brillouin zone integration.
        x = jnp.arange(90) / 30
        y = jnp.zeros_like(x)
        efield, hfield, _ = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            x=x,
            y=y,
        )
        efield_expected = [onp.sum(f, axis=(0, 1)) for f in efield]
        hfield_expected = [onp.sum(f, axis=(0, 1)) for f in hfield]

        # Automatically perform Brillouin zone integration.
        efield_integrated, hfield_integrated, _ = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            x=x,
            y=y,
            brillouin_grid_axes=(0, 1),
        )

        onp.testing.assert_allclose(efield_integrated, efield_expected)
        onp.testing.assert_allclose(hfield_integrated, hfield_expected)


class AmplitudesFromFieldsFromAmplitudesTest(unittest.TestCase):
    @parameterized.expand([[(1, 1)], [(3, 3)]])
    def test_dipole_source(self, bz_shape):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=bz_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        bwd, _, _, _, _, _ = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )
        fwd = jnp.zeros_like(bwd)

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd,
            backward_amplitude=bwd,
            layer_solve_result=solve_result,
        )
        efield, hfield, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            brillouin_grid_axes=(0, 1),
        )

        fwd_recovered, bwd_recovered = sources.amplitudes_for_fields(
            ex=efield[0],
            ey=efield[1],
            hx=hfield[0],
            hy=hfield[1],
            layer_solve_result=solve_result,
            brillouin_grid_axes=(0, 1),
        )

        onp.testing.assert_allclose(fwd_recovered, fwd, atol=1e-12)
        onp.testing.assert_allclose(bwd_recovered, bwd, atol=1e-12)

    @parameterized.expand([[(1, 1)], [(3, 3)]])
    def test_plane_wave(self, bz_shape):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=bz_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        # Model a plane wave. Only the central point in the Brillouin zone
        # grid as nonzero amplitude.
        fwd = jnp.zeros(bz_shape + (2 * EXPANSION.num_terms, 1), dtype=complex)
        fwd = fwd.at[bz_shape[0] // 2, bz_shape[1] // 2, 0, 0].set(1.0)
        bwd = jnp.zeros_like(fwd)

        flux, _ = fields.amplitude_poynting_flux(
            forward_amplitude=fwd,
            backward_amplitude=bwd,
            layer_solve_result=solve_result,
        )
        self.assertSequenceEqual(
            flux.shape,
            bz_shape + (2 * EXPANSION.num_terms, 1),
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd,
            backward_amplitude=bwd,
            layer_solve_result=solve_result,
        )
        efield, hfield, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            brillouin_grid_axes=(0, 1),
        )

        fwd_recovered, bwd_recovered = sources.amplitudes_for_fields(
            ex=efield[0],
            ey=efield[1],
            hx=hfield[0],
            hy=hfield[1],
            layer_solve_result=solve_result,
            brillouin_grid_axes=(0, 1),
        )
        onp.testing.assert_allclose(fwd_recovered, fwd, atol=1e-12)
        onp.testing.assert_allclose(bwd_recovered, bwd, atol=1e-12)


class FluxIntegrationTest(unittest.TestCase):
    @parameterized.expand([[(1, 1)], [(3, 3)]])
    def test_dipole_source(self, bz_shape):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=bz_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        bwd_amplitude_0_end, _, _, _, _, _ = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )

        _, flux = fields.amplitude_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_0_end),
            backward_amplitude=bwd_amplitude_0_end,
            layer_solve_result=solve_result,
        )
        self.assertSequenceEqual(
            flux.shape,
            bz_shape + (2 * EXPANSION.num_terms, 1),
        )
        # Sum over the Fourier orders and the Brillouin zone grid axes. This can be
        # understood as simply summing over the Fourier orders of a larger unit cell,
        # consisting of the original unit cell tiled as specified by the Brillouin
        # grid shape.
        expected_flux = onp.sum(flux)

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_0_end),
            backward_amplitude=bwd_amplitude_0_end,
            layer_solve_result=solve_result,
        )
        efield, hfield, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            brillouin_grid_axes=(0, 1),
        )
        self.assertSequenceEqual(
            efield[0].shape,
            tuple(100 * d for d in bz_shape) + (1,),
        )
        flux_on_grid = fields.time_average_z_poynting_flux(efield, hfield)
        flux_on_grid = onp.mean(flux_on_grid, axis=(-3, -2))
        onp.testing.assert_allclose(flux_on_grid, expected_flux)

    @parameterized.expand([[(1, 1)], [(3, 3)]])
    def test_plane_wave(self, bz_shape):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=bz_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )

        # Model a plane wave. Only the central point in the Brillouin zone
        # grid as nonzero amplitude.
        fwd = jnp.zeros(bz_shape + (2 * EXPANSION.num_terms, 1), dtype=complex)
        fwd = fwd.at[bz_shape[0] // 2, bz_shape[1] // 2, 0, 0].set(1.0)
        bwd = jnp.zeros_like(fwd)

        flux, _ = fields.amplitude_poynting_flux(
            forward_amplitude=fwd,
            backward_amplitude=bwd,
            layer_solve_result=solve_result,
        )
        self.assertSequenceEqual(
            flux.shape,
            bz_shape + (2 * EXPANSION.num_terms, 1),
        )
        # Sum over the Fourier orders and the Brillouin zone grid axes. This can be
        # understood as simply summing over the Fourier orders of a larger unit cell,
        # consisting of the original unit cell tiled as specified by the Brillouin
        # grid shape.
        expected_flux = onp.sum(flux)

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd,
            backward_amplitude=bwd,
            layer_solve_result=solve_result,
        )
        efield, hfield, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            brillouin_grid_axes=(0, 1),
        )
        flux_on_grid = fields.time_average_z_poynting_flux(efield, hfield)
        self.assertSequenceEqual(
            flux_on_grid.shape,
            tuple(100 * d for d in bz_shape) + (1,),
        )
        flux_on_grid = onp.mean(flux_on_grid, axis=(-3, -2))
        onp.testing.assert_allclose(flux_on_grid, expected_flux)


class NumUnitCellsAndBrillouinGridShapeTest(unittest.TestCase):
    @parameterized.expand(
        [
            (None, None, (4, 3, 1, 100, 100, 1)),
            ((4, 3), None, (4, 3, 1, 400, 300, 1)),
            ((4, 3), (0, 1), (1, 400, 300, 1)),
            (None, (0, 1), (1, 400, 300, 1)),
            ((2, 2), (0, 1), (1, 200, 200, 1)),
        ]
    )
    def test_field_shape_matches_expected(
        self,
        num_unit_cells,
        brillouin_grid_axes,
        expected_shape,
    ):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(4, 3),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros((4, 3, 1, 2 * EXPANSION.num_terms, 1)),
            backward_amplitude=jnp.zeros((4, 3, 1, 2 * EXPANSION.num_terms, 1)),
            layer_solve_result=solve_result,
        )
        (ex, _, _), _, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            num_unit_cells=num_unit_cells,
            brillouin_grid_axes=brillouin_grid_axes,
        )
        self.assertSequenceEqual(ex.shape, expected_shape)
