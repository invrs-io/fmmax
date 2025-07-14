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
"""Defines miscellaneous private functions.

Copyright (c) Martin F. Schubert
"""


from typing import Tuple

import jax.numpy as jnp

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Following code is Copyright (c) Meta Platforms, Inc. and affiliates.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def diag(x: jnp.ndarray) -> jnp.ndarray:
    """A batch-compatible version of `numpy.diag`."""
    shape = x.shape + (x.shape[-1],)
    y = jnp.zeros(shape, x.dtype)
    i = jnp.arange(x.shape[-1])
    return y.at[..., i, i].set(x)


def matrix_adjoint(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the adjoint for a batch of matrices."""
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jnp.conj(jnp.transpose(x, axes=axes))


def batch_compatible_shapes(*shapes: Tuple[int, ...]) -> bool:
    """Returns `True` if all the shapes are batch-compatible."""
    max_dims = max([len(s) for s in shapes])
    shapes = tuple([(1,) * (max_dims - len(s)) + s for s in shapes])
    max_shape = [max(dim_shapes) for dim_shapes in zip(*shapes)]
    for shape in shapes:
        if any([a not in (1, b) for a, b in zip(shape, max_shape)]):
            return False
    return True


def atleast_nd(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Adds leading dimensions to `x`, ensuring that it is at least n-dimensional."""
    dims_to_add = tuple(range(max(0, n - x.ndim)))
    return jnp.expand_dims(x, axis=dims_to_add)
