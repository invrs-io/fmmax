"""Defines several utility functions.

Copyright (c) 2025 invrs.io LLC
"""

import jax
import jax.numpy as jnp

from fmmax._utils import (  # noqa: F401
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
