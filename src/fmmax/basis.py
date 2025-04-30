# FMMAX
# Copyright (C) 2025 Martin F. Schubert

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Functions related to vectors and field expansion in the FMM scheme.

Copyright (c) Martin F. Schubert
"""

from typing import Tuple

import jax.numpy as jnp

# ruff: noqa: F401
from fmmax._orig.basis import (
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
