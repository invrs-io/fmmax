"""Functions related to fields in the FMM algorithm.

Copyright (c) 2025 invrs.io LLC
"""

from typing import Tuple

import jax.numpy as jnp

from fmmax._fields import (  # noqa: F401
    _fields_on_grid,
    _layer_fields_3d,
    _stack_fields_3d,
    _validate_amplitudes_shape,
    _validate_matching_lengths,
    amplitude_poynting_flux,
    colocate_amplitudes,
    directional_poynting_flux,
    eigenmode_poynting_flux,
    field_conversion_matrix,
    fields_from_wave_amplitudes,
    fields_on_coordinates,
    fields_on_grid,
    layer_amplitudes_interior,
    layer_fields_3d,
    layer_fields_3d_on_coordinates,
    propagate_amplitude,
    stack_amplitudes_interior,
    stack_amplitudes_interior_with_source,
    stack_fields_3d,
    stack_fields_3d_auto_grid,
    stack_fields_3d_on_coordinates,
)


def time_average_z_poynting_flux(
    electric_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    magnetic_field: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Computes the time-average z-directed Poynting flux, given the physical fields.

    The calculation of Poynting flux is an element-wise operation. When the Poynting
    flux is calculated over a single unit cell, the resulting array may be *averaged*
    to yield a flux equal to that in all orders computed by ``amplitude_poynting_flux``.

    Args:
        electric_field: The tuple of electric fields ``(ex, ey, ez)`` defined on the
            real-space grid.
        magnetic_field: The tuple of magnetic fields ``(hx, hy, hz)`` defined on the
            real-space grid.

    Returns:
        The time-average z-directed Poynting flux, with the same shape as ``ex``.
    """
    ex, ey, _ = electric_field
    hx, hy, _ = magnetic_field
    return 0.5 * jnp.real(ex * jnp.conj(hy) - ey * jnp.conj(hx))
