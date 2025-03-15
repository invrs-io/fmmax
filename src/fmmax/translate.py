"""Functions related to eigensolve with translated permittivity distributions.

Copyright (c) 2024 Martin F. Schubert
"""

import functools

import jax.numpy as jnp

from fmmax import basis, fft, fmm


def translate_layer_solve_result(
    solve_result: fmm.LayerSolveResult,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
) -> fmm.LayerSolveResult:
    """Obtain solve result for layer with translated permittivity distribution.

    Rather than performing eigensolve on translated permittivity distribution, this
    function allows one to translate the solve result. This can result in significant
    compute savings for structures such as slanted gratings.

    Args:
        solve_result: The ``LayerSolveResult`` for the permittivity distribution prior
            to translation.
        dx: The translation of the permittivity distribution along the ``x`` direction.
        dy: The translation of the permittivity distribution along the ``y`` direction.

    Returns:
        The ``LayerSolveResult`` for the layer with translated permittivity.
    """
    assert dx.shape == dy.shape == ()

    _shift = functools.partial(
        _apply_shift_eigenvectors,
        expansion=solve_result.expansion,
        primitive_lattice_vectors=solve_result.primitive_lattice_vectors,
        dx=dx,
        dy=dy,
    )

    ev_te, ev_tm = jnp.split(solve_result.eigenvectors, 2, axis=-2)
    shifted_eigenvectors = jnp.concatenate([_shift(ev_te), _shift(ev_tm)], axis=-2)

    _shift_toeplitz = functools.partial(
        _apply_shift_toeplitz,
        expansion=solve_result.expansion,
        primitive_lattice_vectors=solve_result.primitive_lattice_vectors,
        dx=dx,
        dy=dy,
    )

    m0, m1 = jnp.split(solve_result.transverse_permeability_matrix, 2, axis=-2)
    m00, m01 = jnp.split(m0, 2, axis=-1)
    m10, m11 = jnp.split(m1, 2, axis=-1)
    shifted_transverse_permeability_matrix = jnp.block(
        [
            [_shift_toeplitz(m00), _shift_toeplitz(m01)],
            [_shift_toeplitz(m10), _shift_toeplitz(m11)],
        ],
    )

    return fmm.LayerSolveResult(
        wavelength=solve_result.wavelength,
        in_plane_wavevector=solve_result.in_plane_wavevector,
        primitive_lattice_vectors=solve_result.primitive_lattice_vectors,
        expansion=solve_result.expansion,
        eigenvalues=solve_result.eigenvalues,
        eigenvectors=shifted_eigenvectors,
        z_permittivity_matrix=_shift_toeplitz(solve_result.z_permittivity_matrix),
        inverse_z_permittivity_matrix=_shift_toeplitz(
            solve_result.inverse_z_permittivity_matrix
        ),
        z_permeability_matrix=_shift_toeplitz(solve_result.z_permeability_matrix),
        inverse_z_permeability_matrix=_shift_toeplitz(
            solve_result.inverse_z_permeability_matrix
        ),
        transverse_permeability_matrix=shifted_transverse_permeability_matrix,
        tangent_vector_field=solve_result.tangent_vector_field,
    )


def _apply_shift_eigenvectors(
    eigenvectors: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
) -> jnp.ndarray:
    """Returns eigenvectors that result from translating a structure."""
    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=jnp.zeros((2,)),
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]
    return eigenvectors * jnp.exp(-1j * (kx * dx + ky * dy))[..., jnp.newaxis]


def _apply_shift_toeplitz(
    mat: jnp.ndarray,
    expansion: basis.Expansion,
    primitive_lattice_vectors: basis.LatticeVectors,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
) -> jnp.ndarray:
    """Return toeplitz matrix that results from translating a structure."""
    idx = fft._standard_toeplitz_indices(expansion)

    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    u = reciprocal_vectors.u
    v = reciprocal_vectors.v
    kx = 2 * jnp.pi * (idx[:, :, 0] * u[0] + idx[:, :, 1] * v[0])
    ky = 2 * jnp.pi * (idx[:, :, 0] * u[1] + idx[:, :, 1] * v[1])
    return mat * jnp.exp(-1j * (kx * dx + ky * dy))
