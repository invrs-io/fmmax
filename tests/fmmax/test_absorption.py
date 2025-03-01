"""Tests for absorption-related functions in `fmmax.fields`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, beams, fields, fmm, scattering, sources, utils


class AbsorptionTest(unittest.TestCase):
    @parameterized.expand(
        [
            [(1, 0), (0, 1), 0],
            [(1, 0), (0, 1), 45],
            [(2, 0), (0, 1), 0],
            [(2, 0), (0, 1), 45],
            [(1, 1), (1, -1), 0],
            [(1, 1), (1, -1), 45],
        ]
    )
    def test_absorption_matches_expected(self, u, v, incident_angle_deg):
        primitive_lattice_vectors = basis.LatticeVectors(
            jnp.asarray(u, dtype=float),
            jnp.asarray(v, dtype=float),
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors, approximate_num_terms=20
        )

        permittivity_ambient = (1.5 + 0.0j) ** 2
        permittivity_substrate = (1.5 + 0.3j) ** 2
        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=jnp.asarray(0.55),
            polar_angle=jnp.deg2rad(incident_angle_deg),
            azimuthal_angle=jnp.zeros(()),
            permittivity=permittivity_ambient,
        )

        _eigensolve_fn = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=jnp.asarray(0.55),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        solve_result_ambient = _eigensolve_fn(
            permittivity=jnp.asarray([[permittivity_ambient]])
        )
        solve_result_substrate = _eigensolve_fn(
            permittivity=jnp.asarray([[permittivity_substrate]])
        )
        thickness_ambient = jnp.asarray(1.0)
        thickness_substrate = jnp.asarray(1.0)

        solve_results = (solve_result_ambient, solve_result_substrate)
        thicknesses = (thickness_ambient, thickness_substrate)
        s_matrix = scattering.stack_s_matrix(solve_results, thicknesses)

        n = expansion.num_terms
        fwd_ambient_start = jnp.zeros((2 * n, 1), dtype=complex)
        fwd_ambient_start = fwd_ambient_start.at[0, 0].set(1)
        bwd_substrate_end = jnp.zeros_like(fwd_ambient_start)

        bwd_ambient_end = s_matrix.s21 @ fwd_ambient_start
        bwd_ambient_start = fields.propagate_amplitude(
            bwd_ambient_end, thickness_ambient, solve_result_ambient
        )
        fwd_substrate_start = s_matrix.s11 @ fwd_ambient_start
        fwd_substrate_end = fields.propagate_amplitude(
            fwd_substrate_start, thickness_substrate, solve_result_substrate
        )

        incident, reflected = fields.amplitude_poynting_flux(
            fwd_ambient_start, bwd_ambient_start, solve_result_ambient
        )
        transmitted, _ = fields.amplitude_poynting_flux(
            fwd_substrate_end, bwd_substrate_end, solve_result_substrate
        )

        integrated_absorption = fields.layer_integrated_absorption(
            forward_amplitude_start=fwd_substrate_start,
            backward_amplitude_end=bwd_substrate_end,
            layer_solve_result=solve_result_substrate,
            layer_thickness=thickness_substrate,
            layer_znum=100,
            grid_shape=(100, 100),
            brillouin_grid_axes=None,
        )

        onp.testing.assert_allclose(
            jnp.sum(incident),
            jnp.sum(transmitted) + jnp.mean(integrated_absorption) - jnp.sum(reflected),
            rtol=0.01,
        )

    @parameterized.expand(
        [
            [(1, 0), (0, 1), 0],
            [(1, 0), (0, 1), 45],
            [(2, 0), (0, 1), 0],
            [(2, 0), (0, 1), 45],
            [(1, 1), (1, -1), 0],
            [(1, 1), (1, -1), 45],
        ]
    )
    def test_absorption_matches_expected_patterned_layer(
        self, u, v, incident_angle_deg
    ):
        primitive_lattice_vectors = basis.LatticeVectors(
            jnp.asarray(u, dtype=float),
            jnp.asarray(v, dtype=float),
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors, approximate_num_terms=20
        )

        permittivity_ambient = (1.5 + 0.0j) ** 2

        density = jnp.ones((50, 50))
        density = jnp.pad(density, ((25, 25), (25, 25)))
        permittivity_substrate = utils.interpolate_permittivity(
            permittivity_solid=(1.5 + 0.1j) ** 2,
            permittivity_void=(1.5 + 0.0j) ** 2,
            density=density,
        )
        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=jnp.asarray(0.55),
            polar_angle=jnp.deg2rad(incident_angle_deg),
            azimuthal_angle=jnp.zeros(()),
            permittivity=permittivity_ambient,
        )

        _eigensolve_fn = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=jnp.asarray(0.55),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        solve_result_ambient = _eigensolve_fn(
            permittivity=jnp.asarray([[permittivity_ambient]])
        )
        solve_result_substrate = _eigensolve_fn(permittivity=permittivity_substrate)
        thickness_ambient = jnp.asarray(1.0)
        thickness_substrate = jnp.asarray(1.0)

        solve_results = (solve_result_ambient, solve_result_substrate)
        thicknesses = (thickness_ambient, thickness_substrate)
        s_matrix = scattering.stack_s_matrix(solve_results, thicknesses)

        n = expansion.num_terms
        fwd_ambient_start = jnp.zeros((2 * n, 1), dtype=complex)
        fwd_ambient_start = fwd_ambient_start.at[0, 0].set(1)
        bwd_substrate_end = jnp.zeros_like(fwd_ambient_start)

        bwd_ambient_end = s_matrix.s21 @ fwd_ambient_start
        bwd_ambient_start = fields.propagate_amplitude(
            bwd_ambient_end, thickness_ambient, solve_result_ambient
        )
        fwd_substrate_start = s_matrix.s11 @ fwd_ambient_start
        fwd_substrate_end = fields.propagate_amplitude(
            fwd_substrate_start, thickness_substrate, solve_result_substrate
        )

        incident, reflected = fields.amplitude_poynting_flux(
            fwd_ambient_start, bwd_ambient_start, solve_result_ambient
        )
        transmitted, _ = fields.amplitude_poynting_flux(
            fwd_substrate_end, bwd_substrate_end, solve_result_substrate
        )

        integrated_absorption = fields.layer_integrated_absorption(
            forward_amplitude_start=fwd_substrate_start,
            backward_amplitude_end=bwd_substrate_end,
            layer_solve_result=solve_result_substrate,
            layer_thickness=thickness_substrate,
            layer_znum=100,
            grid_shape=(100, 100),
            brillouin_grid_axes=None,
        )

        onp.testing.assert_allclose(
            jnp.sum(incident),
            jnp.sum(transmitted) + jnp.mean(integrated_absorption) - jnp.sum(reflected),
            rtol=0.01,
        )

    @parameterized.expand(
        [
            [0.5, 0.1],
            [1.0, 0.1],
            [2.0, 0.1],
            [2.0, 0.05],
            [2.0, 0.02],
            [2.0, 0.01],
        ]
    )
    def test_gaussian_bema(self, scale, imag_component):
        primitive_lattice_vectors = basis.LatticeVectors(
            basis.X * scale, basis.Y * scale
        )
        expansion = basis.generate_expansion(primitive_lattice_vectors, 80)

        wavelength = jnp.asarray(0.55)
        permittivity_ambient = (1.5 + 0.0j) ** 2

        density = jnp.ones((1, 1))
        permittivity_substrate = utils.interpolate_permittivity(
            permittivity_solid=(1.5 + imag_component * 1j) ** 2,
            permittivity_void=(1.5 + 0.0j) ** 2,
            density=density,
        )

        brillouin_grid_shape = (1, 1)
        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=wavelength,
            polar_angle=jnp.zeros(()),
            azimuthal_angle=jnp.zeros(()),
            permittivity=permittivity_ambient,
        )
        in_plane_wavevector += basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )

        _eigensolve_fn = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        solve_result_ambient = _eigensolve_fn(
            permittivity=jnp.asarray([[permittivity_ambient]])
        )
        solve_result_substrate = _eigensolve_fn(
            permittivity=jnp.asarray([[permittivity_substrate]])
        )
        thickness_ambient = jnp.asarray(1.0)
        thickness_substrate = jnp.asarray(1.0)

        solve_results = (solve_result_ambient, solve_result_substrate)
        thicknesses = (thickness_ambient, thickness_substrate)

        s_matrix = scattering.stack_s_matrix(solve_results, thicknesses)

        # ---------------------------------------------------------------------

        beam_waist = 0.3

        def _paraxial_gaussian_field_fn(x, y, z):
            # Returns the fields of a z-propagating, x-polarized Gaussian beam.
            # See https://en.wikipedia.org/wiki/Gaussian_beam

            # Adjust array dimensions for proper batching
            wavelength_padded = wavelength[..., jnp.newaxis, jnp.newaxis]

            k = 2 * jnp.pi / wavelength_padded
            z_r = (
                jnp.pi
                * beam_waist**2
                * jnp.sqrt(permittivity_ambient)
                / wavelength_padded
            )
            w_z = beam_waist * jnp.sqrt(1 + (z / z_r) ** 2)
            r = jnp.sqrt(x**2 + y**2)
            ex = (
                beam_waist
                / w_z
                * jnp.exp(-(r**2) / w_z**2)
                * jnp.exp(
                    1j
                    * (
                        (k * z)  # Phase
                        + k
                        * r**2
                        / 2
                        * z
                        / (z**2 + z_r**2)  # Wavefront curvature
                        - jnp.arctan(z / z_r)  # Gouy phase
                    )
                )
            )
            ey = jnp.zeros_like(ex)
            ez = jnp.zeros_like(ex)
            hx = jnp.zeros_like(ex)
            hy = ex * jnp.sqrt(permittivity_ambient)
            hz = jnp.zeros_like(ex)
            return (ex, ey, ez), (hx, hy, hz)

        # Solve for the fields of the beam with the desired rotation and shift.
        x, y = basis.unit_cell_coordinates(
            primitive_lattice_vectors=primitive_lattice_vectors,
            shape=(100, 100),
            num_unit_cells=brillouin_grid_shape,
        )
        (beam_ex, beam_ey, _), (beam_hx, beam_hy, _) = beams.shifted_rotated_fields(
            field_fn=_paraxial_gaussian_field_fn,
            x=x,
            y=y,
            z=jnp.zeros_like(x),
            beam_origin_x=jnp.amax(x) / 2,
            beam_origin_y=jnp.amax(y) / 2,
            beam_origin_z=thickness_ambient,
            polar_angle=jnp.asarray(0.0),
            azimuthal_angle=jnp.asarray(0.0),
            polarization_angle=jnp.asarray(0.0),
        )

        brillouin_grid_axes = (0, 1)
        # Add an additional axis for the number of sources
        fwd_ambient_start, _ = sources.amplitudes_for_fields(
            ex=beam_ex[..., jnp.newaxis],
            ey=beam_ey[..., jnp.newaxis],
            hx=beam_hx[..., jnp.newaxis],
            hy=beam_hy[..., jnp.newaxis],
            layer_solve_result=solve_result_ambient,
            brillouin_grid_axes=brillouin_grid_axes,
        )

        # ---------------------------------------------------------------------

        bwd_substrate_end = jnp.zeros_like(fwd_ambient_start)

        bwd_ambient_end = s_matrix.s21 @ fwd_ambient_start
        bwd_ambient_start = fields.propagate_amplitude(
            bwd_ambient_end, thickness_ambient, solve_result_ambient
        )
        fwd_substrate_start = s_matrix.s11 @ fwd_ambient_start
        fwd_substrate_end = fields.propagate_amplitude(
            fwd_substrate_start, thickness_substrate, solve_result_substrate
        )

        incident, reflected = fields.amplitude_poynting_flux(
            fwd_ambient_start, bwd_ambient_start, solve_result_ambient
        )
        transmitted, _ = fields.amplitude_poynting_flux(
            fwd_substrate_end, bwd_substrate_end, solve_result_substrate
        )

        integrated_absorption = fields.layer_integrated_absorption(
            forward_amplitude_start=fwd_substrate_start,
            backward_amplitude_end=bwd_substrate_end,
            layer_solve_result=solve_result_substrate,
            layer_thickness=thickness_substrate,
            layer_znum=100,
            grid_shape=(100, 100),
            brillouin_grid_axes=None,
        )

        absorption_scale = onp.amax(integrated_absorption)
        self.assertGreater(onp.amin(integrated_absorption), -0.01 * absorption_scale)

        onp.testing.assert_allclose(
            jnp.sum(incident),
            jnp.sum(transmitted) + jnp.mean(integrated_absorption) - jnp.sum(reflected),
            rtol=0.01,
        )
