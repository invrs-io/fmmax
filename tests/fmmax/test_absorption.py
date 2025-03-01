"""Tests for absorption-related functions in `fmmax.fields`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fields, fmm, scattering


class AbsorptionTest(unittest.TestCase):
    @parameterized.expand(
        [
            ((1, 0), (0, 1), 0)
        ]
    )
    def test_absorption_matches_expected(self, u, v, incident_angle_deg):
        primitive_lattice_vectors = basis.LatticeVectors(u, v)
        expansion = basis.generate_expansion(primitive_lattice_vectors, approximate_num_terms=20)

        permittivity_ambient = (1.5 + 0.0j)**2
        permittivity_substrate = (1.5 + 0.3j)**2

        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=jnp.asarray(0.55),
            polar_angle=jnp.deg2rad(incident_angle_deg),
            azimuthal_angle=jnp.zeros(()),
            permittivity=permittivity_ambient
        )

        _eigensolve_fn = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=jnp.asarray(0.55),
            in_plane_wavevector=jnp.asarray(in_p),
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        solve_result_ambient = _eigensolve_fn(permittivity=jnp.asarray([[permittivity_ambient]]))
        solve_result_substrate = _eigensolve_fn(permittivity=jnp.asarray([[permittivity_substrate]])
        thickness_ambient = jnp.asarray(1.0)
        thickness_substrate = jnp.asarray(1.0)

        solve_results = (solve_result_ambient, solve_result_substrate)
        thicknesses = (thickness_ambient, thickness_substrate)
        s_matrices_interior = scattering.stack_s_matrices_interior(solve_results, thicknesses)
        # s-matrix for layers before (and including) the last layer
        s_matrix = s_matrices_interior[-1][0]

        n = expansion.num_terms
        fwd_ambient_start = jnp.zeros((2 * n, 1), dtype=complex)
        fwd_ambient_start = fwd_ambient_start.at[0, 0].set(1)
        bwd_substrate_end = jnp.zeros_like(fwd_ambient_start)

        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=fwd_ambient_start,
            backward_amplitude_N_end=bwd_substrate_end,
        )

        bwd_ambient_end = s_matrix.s12 @ fwd_ambient_start
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


        efield, hfield, (x, y, z) = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=solve_results,
            layer_thicknesses=thicknesses,
            layer_znum=(znum, znum),
            grid_shape=(100, 100),
            num_unit_cells=(1, 1)
        )

        absorption = -fields._poynting_vector_divergence(
            efield=efield,
            hfield=hfield,
            primitive_lattice_vectors=primitive_lattice_vectors,
            num_unit_cells=(1, 1),
            dz=thickness_substrate / znum
        )

        intensity = jnp.sum(jnp.abs(jnp.asarray(efield))**2, axis=0)

        integrated_absorption = fields.layer_integrated_absorption(
            forward_amplitude_start=fwd_substrate_start,
            backward_amplitude_end=bwd_substrate_end,
            layer_solve_result=solve_result_substrate,
            layer_thickness=thickness_substrate,
            layer_znum=znum,
            grid_shape=(100, 100),
            brillouin_grid_axes=None
        )
