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
"""Defines several utility functions.

Copyright (c) Martin F. Schubert
"""

from typing import Tuple

import jax
import jax.numpy as jnp


def solve(a: jnp.ndarray, b: jnp.ndarray, *, force_x64_solve: bool) -> jnp.ndarray:
    """Solves ``A @ x = b``, optionally using 64-bit precision."""
    output_dtype = jnp.promote_types(a.dtype, b.dtype)
    if force_x64_solve and jax.config.read("jax_enable_x64"):
        a = a.astype(jnp.promote_types(a.dtype, jnp.float64))
        b = b.astype(jnp.promote_types(b.dtype, jnp.float64))
    return jnp.linalg.solve(a, b).astype(output_dtype)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Following code is Copyright (c) Meta Platforms, Inc. and affiliates.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def interpolate_permittivity(
    permittivity_solid: jnp.ndarray,
    permittivity_void: jnp.ndarray,
    density: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolates the permittivity with a scheme that avoids zero crossings.

    The interpolation uses the scheme introduced in [2019 Christiansen], which avoids
    zero crossings that can occur with metals or lossy materials having a negative
    real component of the permittivity. https://doi.org/10.1016/j.cma.2018.08.034

    Args:
        permittivity_solid: The permittivity of solid regions.
        permittivity_void: The permittivity of void regions.
        density: The density, specifying which locations are solid and which are void.

    Returns:
        The interpolated permittivity.
    """
    n_solid = jnp.real(jnp.sqrt(permittivity_solid))
    k_solid = jnp.imag(jnp.sqrt(permittivity_solid))
    n_void = jnp.real(jnp.sqrt(permittivity_void))
    k_void = jnp.imag(jnp.sqrt(permittivity_void))
    n = density * n_solid + (1 - density) * n_void
    k = density * k_solid + (1 - density) * k_void
    return (n + 1j * k) ** 2


def angular_frequency_for_wavelength(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Returns the angular frequency for the specified wavelength."""
    return 2 * jnp.pi / wavelength  # Since by our convention c == 1.


def absolute_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    """Returns the absolute axes for given relative axes and array dimensionality."""
    if not all(a in list(range(-ndim, ndim)) for a in axes):
        raise ValueError(
            f"All elements of `axes` must be in the range ({ndim}, {ndim - 1}) "
            f"but got {axes}."
        )
    absolute_axes = tuple([d % ndim for d in axes])
    if len(absolute_axes) != len(set(absolute_axes)):
        raise ValueError(
            f"Found duplicates in `axes`; computed absolute axes are {absolute_axes}."
        )
    return absolute_axes
