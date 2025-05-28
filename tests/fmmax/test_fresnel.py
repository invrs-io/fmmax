"""Tests refection and transmission against the Fresnel expressions.

Copyright (c) Martin F. Schubert
"""

import functools
import unittest

import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fields, fmm, scattering


def fresnel_rt(n1, n2, theta_i):
    """Compute reflection and transmission by the Fresnel equations."""
    cos_theta_i = onp.cos(theta_i)
    cos_theta_t = onp.sqrt(1 - (n1 / n2 * onp.sin(theta_i)) ** 2)

    rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    ts = (2 * n1 * cos_theta_i) / (n1 * cos_theta_i + n2 * cos_theta_t)

    rp = -(n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
    tp = (2 * n1 * cos_theta_i) / (n2 * cos_theta_i + n1 * cos_theta_t)

    Rs = onp.abs(rs) ** 2
    Rp = onp.abs(rp) ** 2

    Ts = onp.abs(ts) ** 2 * (n2 * cos_theta_t).real / (n1 * cos_theta_i).real
    Tp = onp.abs(tp) ** 2 * (n2 * cos_theta_t.conj() / (n1 * cos_theta_i)).real

    assert onp.isclose(Rs + Ts, 1)
    assert onp.isclose(Rp + Tp, 1)
    return ((rs, rp), (ts, tp)), ((Rs, Rp), (Ts, Tp))


class FresnelComparisonTest(unittest.TestCase):
    @parameterized.expand(
        [
            (1.0 + 0.0j, 1.4 + 0.0j, 0.0),
            (1.0 + 0.0j, 1.4 + 0.0j, 30.0),
            (1.0 + 0.0j, 1.4 + 0.00001j, 0.0),
            (1.0 + 0.0j, 1.4 + 0.00001j, 30.0),
            (1.4 + 0.0j, 1.0 + 0.0j, 0.0),
            (1.4 + 0.0j, 1.0 + 0.0j, 30.0),
            (1.4 + 0.00001j, 1.0 + 0.0j, 0.0),
            (1.4 + 0.00001j, 1.0 + 0.0j, 30.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 0.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 10.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 20.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 30.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 40.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 50.0),
            (1.0 + 0.0j, 1.0 + 1.0j, 60.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 0.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 10.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 20.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 30.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 40.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 50.0),
            (1.0 + 0.0j, 2.0 + 0.0j, 60.0),
        ]
    )
    def test_validate_fmm(self, n_ambient, n_substrate, incident_angle_deg):
        wavelength = jnp.asarray(0.537)
        incident_angle = jnp.deg2rad(incident_angle_deg)

        expansion = basis.Expansion(basis_coefficients=onp.asarray([[0, 0]]))
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)

        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=wavelength,
            polar_angle=incident_angle,
            azimuthal_angle=jnp.zeros(()),
            permittivity=jnp.asarray(n_ambient, dtype=complex) ** 2,
        )

        eigensolve_fn = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

        solve_result_ambient = eigensolve_fn(
            permittivity=jnp.asarray([[n_ambient]], dtype=complex) ** 2
        )
        solve_result_substrate = eigensolve_fn(
            permittivity=jnp.asarray([[n_substrate]], dtype=complex) ** 2
        )

        layer_solve_results = (solve_result_ambient, solve_result_substrate)
        layer_thicknesses = (jnp.ones(()), jnp.ones(()))

        # Assemble scattering matrix
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=layer_thicknesses,
        )
        s_matrix = s_matrices_interior[-1][0]

        # Compute amplitudes. We excite with y-polarized and x-polarized efields.
        fwd_amplitude_ambient_start = jnp.asarray([[1, 0], [0, 1]], dtype=complex)
        fwd_amplitude_ambient_end = fields.propagate_amplitude(
            amplitude=fwd_amplitude_ambient_start,
            distance=layer_thicknesses[0],
            layer_solve_result=solve_result_ambient,
        )
        bwd_amplitude_ambient_end = s_matrix.s21 @ fwd_amplitude_ambient_start
        fwd_amplitude_substrate_start = s_matrix.s11 @ fwd_amplitude_ambient_start
        bwd_amplitude_substrate_start = jnp.zeros_like(fwd_amplitude_ambient_start)

        # Compute power reflection and transmission coefficients
        incident_flux, reflected_flux = fields.amplitude_poynting_flux(
            forward_amplitude=fwd_amplitude_ambient_end,
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=solve_result_ambient,
        )
        transmitted_flux, _ = fields.amplitude_poynting_flux(
            forward_amplitude=fwd_amplitude_substrate_start,
            backward_amplitude=bwd_amplitude_substrate_start,
            layer_solve_result=solve_result_substrate,
        )
        Rs, Rp = -jnp.diag(reflected_flux) / jnp.diag(incident_flux)
        Ts, Tp = jnp.diag(transmitted_flux) / jnp.diag(incident_flux)

        # Compute the incident, reflected, and transmitted electric fields.
        e_incident, h_incident = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd_amplitude_ambient_end,
            backward_amplitude=jnp.zeros_like(fwd_amplitude_ambient_end),
            layer_solve_result=solve_result_ambient,
        )
        e_reflected, h_reflected = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=solve_result_ambient,
        )
        e_transmitted, h_transmitted = fields.fields_from_wave_amplitudes(
            forward_amplitude=fwd_amplitude_substrate_start,
            backward_amplitude=jnp.zeros_like(fwd_amplitude_substrate_start),
            layer_solve_result=solve_result_substrate,
        )
        # Squeeze out the Fourier coefficient axis, since we only deal with the
        # zeroth order.
        e_incident = jnp.squeeze(jnp.asarray(e_incident), axis=1)
        e_reflected = jnp.squeeze(jnp.asarray(e_reflected), axis=1)
        e_transmitted = jnp.squeeze(jnp.asarray(e_transmitted), axis=1)
        h_incident = jnp.squeeze(jnp.asarray(h_incident), axis=1)
        h_reflected = jnp.squeeze(jnp.asarray(h_reflected), axis=1)
        h_transmitted = jnp.squeeze(jnp.asarray(h_transmitted), axis=1)
        assert e_incident.shape == (3, 2)

        # Use electric field transmission for s polarization.
        assert e_incident[0, 0] == e_incident[2, 0] == 0
        assert e_reflected[0, 0] == e_reflected[2, 0] == 0
        assert e_transmitted[0, 0] == e_transmitted[2, 0] == 0
        rs = e_reflected[1, 0] / e_incident[1, 0]
        ts = e_transmitted[1, 0] / e_incident[1, 0]

        # Use magnetic field transmission for p polarization.
        assert h_incident[0, 1] == h_incident[2, 1] == 0
        assert h_reflected[0, 1] == h_reflected[2, 1] == 0
        assert h_transmitted[0, 1] == h_transmitted[2, 1] == 0
        rp = -h_reflected[1, 1] / h_incident[1, 1]
        tp = h_transmitted[1, 1] / h_incident[1, 1] * n_ambient / n_substrate

        # Compare complex reflection and transmission coefficients, and real-valued
        # power reflection and transmission coefficients.
        (
            ((expected_rs, expected_rp), (expected_ts, expected_tp)),
            ((expected_Rs, expected_Rp), (expected_Ts, expected_Tp)),
        ) = fresnel_rt(n1=n_ambient, n2=n_substrate, theta_i=incident_angle)

        rtol = 1e-5
        atol = 1e-5
        with self.subTest("rs"):
            onp.testing.assert_allclose(rs, expected_rs, rtol=rtol, atol=atol)
        with self.subTest("rp"):
            onp.testing.assert_allclose(rp, expected_rp, rtol=rtol, atol=atol)
        with self.subTest("ts"):
            onp.testing.assert_allclose(ts, expected_ts, rtol=rtol, atol=atol)
        with self.subTest("tp"):
            onp.testing.assert_allclose(tp, expected_tp, rtol=rtol, atol=atol)
        with self.subTest("Rs"):
            onp.testing.assert_allclose(Rs, expected_Rs, rtol=rtol, atol=atol)
        with self.subTest("Rp"):
            onp.testing.assert_allclose(Rp, expected_Rp, rtol=rtol, atol=atol)
        with self.subTest("Ts"):
            onp.testing.assert_allclose(Ts, expected_Ts, rtol=rtol, atol=atol)
        with self.subTest("Tp"):
            onp.testing.assert_allclose(Tp, expected_Tp, rtol=rtol, atol=atol)
