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

import jax
import jax.numpy as jnp

# ruff: noqa: F401
from fmmax._orig.utils import (
    absolute_axes,
    angular_frequency_for_wavelength,
    interpolate_permittivity,
)


def solve(a: jnp.ndarray, b: jnp.ndarray, *, force_x64_solve: bool) -> jnp.ndarray:
    """Solves ``A @ x = b``, optionally using 64-bit precision."""
    output_dtype = jnp.promote_types(a.dtype, b.dtype)
    if force_x64_solve and jax.config.read("jax_enable_x64"):
        a = a.astype(jnp.promote_types(a.dtype, jnp.float64))
        b = b.astype(jnp.promote_types(b.dtype, jnp.float64))
    return jnp.linalg.solve(a, b).astype(output_dtype)
