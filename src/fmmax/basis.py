"""Functions related to vectors and field expansion in the FMM scheme.

Copyright (c) 2025 invrs.io LLC
"""

from typing import Tuple

import jax.numpy as jnp

from fmmax._basis import (  # noqa: F401
    Expansion,
    LatticeVectors,
    Truncation,
    X,
    Y,
    _basis_coefficients_circular,
    _basis_coefficients_parallelogramic,
    _cross_product,
    _HashableArray,
    _reciprocal,
    generate_expansion,
    min_array_shape_for_expansion,
    plane_wave_in_plane_wavevector,
    transverse_wavevectors,
    unit_cell_coordinates,
    validate_shape_for_expansion,
)


def brillouin_zone_in_plane_wavevector(
    brillouin_grid_shape: Tuple[int, int],
    primitive_lattice_vectors: LatticeVectors,
) -> jnp.ndarray:
    """Compute in-plane wavevectors suitable for Brillouin zone integration.

    The wavevectors are evenly spaced within the first Brillouin zone; for odd grid
    shapes, they subdivide the first Brillouin zone evenly. For even grid shapes, they
    are offset so ``(0, 0)`` is included among the wavevectors.

    Args:
        brillouin_grid_shape: The shape of the wavevector grid.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.

    Returns:
        The in-plane wavevectors, with shape ``brillouin_grid_shape + (2,)``.
    """
    if len(brillouin_grid_shape) != 2 or brillouin_grid_shape < (1, 1):
        raise ValueError(
            f"`brillouin_grid_shape` must be length-2 with positive values, "
            f"but got {brillouin_grid_shape}."
        )

    udim, vdim = brillouin_grid_shape
    i, j = jnp.meshgrid(
        jnp.arange(-(udim // 2), udim - (udim // 2)) / udim,
        jnp.arange(-(vdim // 2), vdim - (vdim // 2)) / vdim,
        indexing="ij",
    )
    assert i.shape == brillouin_grid_shape
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    ku = reciprocal_vectors.u
    kv = reciprocal_vectors.v
    return jnp.stack(
        [
            2 * jnp.pi * (i * ku[..., 0] + j * kv[..., 0]),
            2 * jnp.pi * (i * ku[..., 1] + j * kv[..., 1]),
        ],
        axis=-1,
    )
