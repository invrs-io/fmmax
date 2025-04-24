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
"""Functions related to scattering matrix computation for the FMM algorithm.

Copyright (c) Martin F. Schubert
"""

import dataclasses
import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

from fmmax import fmm, misc, utils

# ruff: noqa: F401
from fmmax._orig.scattering import (
    ScatteringMatrix,
    _extend_s_matrix,
    append_layer,
    prepend_layer,
    set_end_layer_thickness,
    set_start_layer_thickness,
)


def stack_s_matrix(
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    force_x64_solve: bool = False,
) -> ScatteringMatrix:
    """Computes the s-matrix for a stack of layers.

    If only a single layer is provided, the scattering matrix is just the
    identity matrix, and start and end layer data is for the same layer.

    Args:
        layer_solve_results: The eigensolve results for layers in the stack.
        layer_thicknesses: The scalar thicknesses for layers in the stack.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The ``ScatteringMatrix``.
    """
    return _stack_s_matrices(
        layer_solve_results, layer_thicknesses, force_x64_solve=force_x64_solve
    )[-1]


def stack_s_matrices_interior(
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    force_x64_solve: bool = False,
) -> Tuple[Tuple[ScatteringMatrix, ScatteringMatrix], ...]:
    """Computes scattering matrices before and after each layer in the stack.

    Specifically, for each layer ``i`` two ``ScatteringMatrix`` are returned. The
    first relates fields in the substack ``0, ..., i``, while the second relates
    the fields in the substack ``i, ..., N``, where ``N`` is the maximum layer
    index. These two scattering matrices are denoted as the corresponding
    to the "before" substack and the "after" substack.

    Args:
        layer_solve_results: The eigensolve results for layers in the stack.
        layer_thicknesses: The scalar thicknesses for layers in the stack.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The tuple of ``(scattering_matrix_before, scattering_matrix_after)``.
    """
    before = _stack_s_matrices(
        layer_solve_results,
        layer_thicknesses,
        force_x64_solve=force_x64_solve,
    )

    # Compute the scattering matrix for the substack "after" each layer. We do
    # this by computing the scattering matrix for the substack "before" each layer
    # in the reversed stack. Then, reverse each resulting scattering matrix.
    reverse = _stack_s_matrices(
        layer_solve_results[::-1],
        layer_thicknesses[::-1],
        force_x64_solve=force_x64_solve,
    )
    after = tuple(
        [
            ScatteringMatrix(
                s11=s.s22,
                s12=s.s21,
                s21=s.s12,
                s22=s.s11,
                start_layer_solve_result=s.end_layer_solve_result,
                start_layer_thickness=s.end_layer_thickness,
                end_layer_solve_result=s.start_layer_solve_result,
                end_layer_thickness=s.start_layer_thickness,
            )
            for s in reverse[::-1]
        ]
    )
    return tuple(zip(before, after))


def _stack_s_matrices(
    layer_solve_results: Sequence[fmm.LayerSolveResult],
    layer_thicknesses: Sequence[jnp.ndarray],
    force_x64_solve: bool,
) -> Tuple[ScatteringMatrix, ...]:
    """Computes the s-matrices for a stack of layers.

    The matrices are for various substacks: the first scattering matrix is
    for the first layer only, the second is for the stack consisting of the
    first and second layer, etc.

    Note that the ``LayerSolveResult`` attributes of the resulting ``ScatteringMatrix``
    has the ``tangent_vector_field`` attribute stripped out.

    Args:
        layer_solve_results: The eigensolve results for layers in the stack. All
            eigensolve results must have identical batch shape.
        layer_thicknesses: The scalar thicknesses for layers in the stack.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The tuple of ``ScatteringMatrix``.
    """
    if len(layer_solve_results) != len(layer_thicknesses):
        raise ValueError(
            f"`layer_solve_results` and `layer_thicknesses` should have the same "
            f"length but got {len(layer_solve_results)} and {len(layer_thicknesses)}."
        )

    # Remove the tangent vector fields from the solve results.
    layer_solve_results = tuple(
        dataclasses.replace(solve_result, tangent_vector_field=None)
        for solve_result in layer_solve_results
    )

    # Broadcast all layer solve results so their attributes have the full batch shape.
    batch_shape = jnp.broadcast_shapes(
        *[lsr.batch_shape for lsr in layer_solve_results]
    )
    layer_solve_results = [
        fmm.broadcast_result(lsr, batch_shape) for lsr in layer_solve_results
    ]

    # Broadcast all layer thicknesses so they have a common shape.
    t_shape = jnp.broadcast_shapes(*[jnp.shape(t) for t in layer_thicknesses])
    layer_thicknesses = [jnp.broadcast_to(t, t_shape) for t in layer_thicknesses]

    # The initial scattering matrix is just the identity matrix, with the
    # necessary batch dimensions.
    eye = misc.diag(jnp.ones_like(layer_solve_results[0].eigenvalues))
    layer_s_matrix_0 = ScatteringMatrix(
        s11=eye,
        s12=jnp.zeros_like(eye),
        s21=jnp.zeros_like(eye),
        s22=eye,
        start_layer_solve_result=layer_solve_results[0],
        start_layer_thickness=layer_thicknesses[0],
        end_layer_solve_result=layer_solve_results[0],
        end_layer_thickness=layer_thicknesses[0],
    )

    if len(layer_solve_results) == 1:
        return (layer_s_matrix_0,)

    layer_s_matrix_1 = _pair_s_matrix(
        layer_solve_result=layer_solve_results[0],
        layer_thickness=layer_thicknesses[0],
        next_layer_solve_result=layer_solve_results[1],
        next_layer_thickness=layer_thicknesses[1],
        force_x64_solve=force_x64_solve,
    )

    if len(layer_solve_results) == 2:
        return (layer_s_matrix_0, layer_s_matrix_1)

    # If we have more than two layers, stack the remaining layer solve results and
    # thicknesses so they can be scanned over.

    stacked_remaining_solve_results = tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *layer_solve_results[2:],
    )
    stacked_remaining_thicknesses = jnp.stack(layer_thicknesses[2:])

    def scan_fn(s_matrix, args):
        next_layer_solve_result, next_layer_thickness = args
        s_matrix = append_layer(
            s_matrix,
            next_layer_solve_result,
            next_layer_thickness,
            force_x64_solve=force_x64_solve,
        )
        return s_matrix, s_matrix

    _, stacked_remaining_s_matrices = jax.lax.scan(
        scan_fn,
        init=layer_s_matrix_1,
        xs=(stacked_remaining_solve_results, stacked_remaining_thicknesses),
    )

    # Unstack to return a tuple of scattering matrices.
    remaining_s_matrices = tuple(
        tree_util.tree_map(lambda x: x[i, ...], stacked_remaining_s_matrices)
        for i in range(len(layer_solve_results) - 2)
    )
    return (layer_s_matrix_0, layer_s_matrix_1) + remaining_s_matrices


def stack_s_matrix_scan(
    layer_solve_results: fmm.LayerSolveResult,
    layer_thicknesses: jnp.ndarray,
    force_x64_solve: bool = False,
) -> ScatteringMatrix:
    """Computes the scattering matrix for a stack of layers by a scan operation.

    Unlike ``stack_s_matrix``, this function uses a scan operation rather than a python
    for loop, which can lead to significantly smaller programs and shorter compile
    times. However, it requires that the layer solve results and thicknesses for be
    represented in the leading batch dimension of the arguments.

    This function is best used when the eigensolve for each layer is also done in a
    batched manner.

    Args:
        layer_solve_results: The layer solve results for all layers in the stack.
        layer_thicknesses: The scalar layer thicknesses for all layers in the stack.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The scattering matrix for the stack.
    """
    assert layer_thicknesses.ndim == 1
    if layer_solve_results.batch_shape[0] != len(layer_thicknesses):
        raise ValueError(
            f"`layer_solve_results` and `layer_thicknesses` should be compatible (i.e. "
            f"correspond to the same number of layers) but inferred layer numbers of "
            f"{layer_solve_results.batch_shape[0]} and {layer_thicknesses.shape[0]}."
        )

    eye = misc.diag(
        jnp.ones(
            layer_solve_results.eigenvalues.shape[1:],
            dtype=layer_solve_results.eigenvalues.dtype,
        )
    )
    start_solve_result = tree_util.tree_map(lambda x: x[0, ...], layer_solve_results)
    s_matrix = ScatteringMatrix(
        s11=eye,
        s12=jnp.zeros_like(eye),
        s21=jnp.zeros_like(eye),
        s22=eye,
        start_layer_solve_result=start_solve_result,
        start_layer_thickness=layer_thicknesses[0],
        end_layer_solve_result=start_solve_result,
        end_layer_thickness=layer_thicknesses[0],
    )

    def scan_fn(s_matrix, x):
        next_layer_solve_result, next_layer_thickness = x
        s_matrix = append_layer(
            s_matrix,
            next_layer_solve_result,
            next_layer_thickness,
            force_x64_solve=force_x64_solve,
        )
        return s_matrix, s_matrix

    s_matrix, _ = jax.lax.scan(
        scan_fn,
        init=s_matrix,
        xs=(
            tree_util.tree_map(lambda x: x[1:], layer_solve_results),
            layer_thicknesses[1:],
        ),
    )
    return s_matrix


def redheffer_star_product(
    a: ScatteringMatrix,
    b: ScatteringMatrix,
    force_x64_solve: bool = False,
) -> ScatteringMatrix:
    """Compute the Redheffer star product of two scattering matrices."""
    a_extended = append_layer(a, b.start_layer_solve_result, b.start_layer_thickness)
    a11, a12, a21, a22 = a_extended.s11, a_extended.s12, a_extended.s21, a_extended.s22
    b11, b12, b21, b22 = b.s11, b.s12, b.s21, b.s22

    solve = functools.partial(utils.solve, force_x64_solve=force_x64_solve)

    # See https://en.wikipedia.org/wiki/Redheffer_star_product
    eye = misc.diag(jnp.ones_like(a11[..., 0]))
    s11 = b11 @ solve(eye - a12 @ b21, a11)
    s12 = b12 + b11 @ solve(eye - a12 @ b21, a12 @ b22)
    s21 = a21 + a22 @ solve(eye - b21 @ a12, b21 @ a11)
    s22 = a22 @ solve(eye - b21 @ a12, b22)
    return ScatteringMatrix(
        s11=s11,
        s12=s12,
        s21=s21,
        s22=s22,
        start_layer_solve_result=a.start_layer_solve_result,
        start_layer_thickness=a.start_layer_thickness,
        end_layer_solve_result=b.end_layer_solve_result,
        end_layer_thickness=b.end_layer_thickness,
    )


def _pair_s_matrix(
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
    force_x64_solve: bool,
) -> ScatteringMatrix:
    """Generate the scattering matrix for a pair of layers."""
    # Alias for brevity: eigenvalues, eigenvectors, and omega-k matrix.
    q = layer_solve_result.eigenvalues
    phi = layer_solve_result.eigenvectors
    omega_k = layer_solve_result.omega_script_k_matrix

    next_q = next_layer_solve_result.eigenvalues
    next_phi = next_layer_solve_result.eigenvectors
    next_omega_k = next_layer_solve_result.omega_script_k_matrix

    solve = functools.partial(utils.solve, force_x64_solve=force_x64_solve)

    term1 = q[..., jnp.newaxis] * solve(
        omega_k @ phi,
        next_omega_k @ next_phi * (1 / next_q)[..., jnp.newaxis, :],
    )
    term2 = solve(omega_k @ phi, omega_k @ next_phi)
    i11 = i22 = 0.5 * (term1 + term2)
    i12 = i21 = 0.5 * (-term1 + term2)

    fd = jnp.exp(1j * q * layer_thickness)
    fd_next = jnp.exp(1j * next_q * next_layer_thickness)

    # The computation is identical to that in `_extend_s_matrix` with `s11` and `s22`
    # being the identity, and `s12` and `s21` being zero.
    fd_diag = jnp.expand_dims(misc.diag(fd), tuple(range(i11.ndim - fd.ndim - 1)))
    s11 = solve(i11, fd_diag)
    s12 = solve(i11, -i12 * fd_next[..., jnp.newaxis, :])
    s21 = i21 @ s11
    s22 = i21 @ s12 + i22 * fd_next[..., jnp.newaxis, :]

    return ScatteringMatrix(
        s11=s11,
        s12=s12,
        s21=s21,
        s22=s22,
        start_layer_solve_result=layer_solve_result,
        start_layer_thickness=layer_thickness,
        end_layer_solve_result=next_layer_solve_result,
        end_layer_thickness=next_layer_thickness,
    )
