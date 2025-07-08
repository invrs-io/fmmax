"""Functions related transforming to and from the Fourier basis.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import jax.numpy as jnp

from fmmax import basis


def _standard_toeplitz_indices(expansion: basis.Expansion) -> jnp.ndarray:
    """Computes the indices for a standard Toeplitz matrix for `basis_coefficients`.

    Args:
        expansion: The field expansion to be used.

    Returns:
        The indices, with shape `(num, num, 2)`.
    """
    i, j = jnp.meshgrid(
        jnp.arange(expansion.num_terms),
        jnp.arange(expansion.num_terms),
        indexing="ij",
    )
    basis_coefficients = jnp.asarray(expansion.basis_coefficients)
    idx = basis_coefficients[i, :] - basis_coefficients[j, :]
    return idx
