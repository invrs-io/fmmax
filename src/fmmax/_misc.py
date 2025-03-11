"""Defines miscellaneous private functions.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from typing import Tuple

import jax.numpy as jnp


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
