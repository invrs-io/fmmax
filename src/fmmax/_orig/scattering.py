"""Functions related to scattering matrix computation for the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import functools
from typing import Tuple

import jax.numpy as jnp
from jax import tree_util

from fmmax import fmm, utils


@dataclasses.dataclass
class ScatteringMatrix:
    """Stores the scattering matrix for a stack of layers.

    The first layer in a stack is the "start" layer, and the last layer in the
    stack is the "end" layer.

    The scattering matrix relates the forward-going and backward-going waves
    on the two sides of a layer stack, which are labeled ``a`` and ``b`` respectively.

    Note that forward going fields are defined at the *start* of a layer while
    backward-going fields are defined at the *end* of a layer, as depicted below.
    This is discussed near equation 4.1 in [1999 Whittaker].
    ::

                |             |           |         |           |
                |   layer 0   |  layer 1  |   ...   |  layer N  |
                | start layer |           |         | end layer |
                |             |           |         |           |
                 -> a_0                              -> a_N
                        b_0 <-                            b_N <-

    Following the convention of [1999 Whittaker], the terms a_N and b_0 are
    obtained from,
    ::


                    a_N = s11 @ a_0 + s12 @ b_N
                    b_0 = s21 @ a_0 + s22 @ b_N

    Besides the actual scattering matrix element, the ``ScatteringMatrix`` stores
    information about the start and end layers, which are needed to extend the
    scattering matrix to include more layers.

    Attributes:
        s11: Relates forward-going fields at start to forward-going fields at end.
        s12: Relates backward-going fields at end to forward-going fields at end.
        s21: Relates forward-going fields at start to backward-going fields at start.
        s22: Relates backward-going fields at end to backward-going fields at start.
        start_layer_solve_result: The eigensolve result for the start layer.
        start_layer_thickness: The start layer thickness.
        end_layer_solve_result: The eigensolve result for the end layer.
        end_layer_thickness: The end layer thickness.
    """

    s11: jnp.ndarray
    s12: jnp.ndarray
    s21: jnp.ndarray
    s22: jnp.ndarray

    start_layer_solve_result: fmm.LayerSolveResult
    start_layer_thickness: jnp.ndarray

    end_layer_solve_result: fmm.LayerSolveResult
    end_layer_thickness: jnp.ndarray


def append_layer(
    s_matrix: ScatteringMatrix,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
    force_x64_solve: bool = False,
) -> ScatteringMatrix:
    """Returns new scattering matrix for the stack with an appended layer.

    Args:
        s_matrix: The existing scattering matrix.
        next_layer_solve_result: The eigensolve result for the layer to append.
        next_layer_thickness: The thickness for the layer to append.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The new ``ScatteringMatrix``.
    """
    s11_next, s12_next, s21_next, s22_next = _extend_s_matrix(
        s_matrix_blocks=(s_matrix.s11, s_matrix.s12, s_matrix.s21, s_matrix.s22),
        layer_solve_result=s_matrix.end_layer_solve_result,
        layer_thickness=s_matrix.end_layer_thickness,
        next_layer_solve_result=next_layer_solve_result,
        next_layer_thickness=next_layer_thickness,
        force_x64_solve=force_x64_solve,
    )
    return ScatteringMatrix(
        s11=s11_next,
        s12=s12_next,
        s21=s21_next,
        s22=s22_next,
        start_layer_solve_result=s_matrix.start_layer_solve_result,
        start_layer_thickness=s_matrix.start_layer_thickness,
        end_layer_solve_result=next_layer_solve_result,
        end_layer_thickness=next_layer_thickness,
    )


def prepend_layer(
    s_matrix: ScatteringMatrix,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
    force_x64_solve: bool = False,
) -> ScatteringMatrix:
    """Returns new scattering matrix for the stack with a prepended layer.

    Args:
        s_matrix: The existing scattering matrix.
        next_layer_solve_result: The eigensolve result for the layer to append.
        next_layer_thickness: The thickness for the layer to append.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The new ``ScatteringMatrix``.
    """
    # To prepend a layer, we compute the scattering matrix that results from
    # appending the layer to the reversed stack. The scattering matrix for
    # the reversed stack is simply the re-ordering of the matrix blocks, i.e.
    # s11 become s22, s12 becomes s21, etc.
    s22_next, s21_next, s12_next, s11_next = _extend_s_matrix(
        s_matrix_blocks=(s_matrix.s22, s_matrix.s21, s_matrix.s12, s_matrix.s11),
        layer_solve_result=s_matrix.start_layer_solve_result,
        layer_thickness=s_matrix.start_layer_thickness,
        next_layer_solve_result=next_layer_solve_result,
        next_layer_thickness=next_layer_thickness,
        force_x64_solve=force_x64_solve,
    )
    return ScatteringMatrix(
        s11=s11_next,
        s12=s12_next,
        s21=s21_next,
        s22=s22_next,
        start_layer_solve_result=next_layer_solve_result,
        start_layer_thickness=next_layer_thickness,
        end_layer_solve_result=s_matrix.end_layer_solve_result,
        end_layer_thickness=s_matrix.end_layer_thickness,
    )


def _extend_s_matrix(
    s_matrix_blocks: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    layer_solve_result: fmm.LayerSolveResult,
    layer_thickness: jnp.ndarray,
    next_layer_solve_result: fmm.LayerSolveResult,
    next_layer_thickness: jnp.ndarray,
    force_x64_solve: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extends the scattering matrix, adding a layer to the end.

    The approach here follows section 5 of [1999 Whittaker].

    Args:
        s_matrix_blocks: The elements ``(s11, s12, s21, s22)``.
        layer_solve_result: The eigensolve result of the ending layer.
        layer_thickness: The thickness of the ending layer.
        next_layer_solve_result: The eigensolve result for the layer to append.
        next_layer_thickness: The thickness for the layer to append.
        force_x64_solve: If ``True``, matrix solves will be done with 64 bit precision.

    Returns:
        The new ``ScatteringMatrix``.
    """
    # Alias for brevity: eigenvalues, eigenvectors, and omega-k matrix.
    q = layer_solve_result.eigenvalues
    phi = layer_solve_result.eigenvectors
    omega_k = layer_solve_result.omega_script_k_matrix

    next_q = next_layer_solve_result.eigenvalues
    next_phi = next_layer_solve_result.eigenvectors
    next_omega_k = next_layer_solve_result.omega_script_k_matrix

    solve = functools.partial(utils.solve, force_x64_solve=force_x64_solve)

    # Compute the interface matrices following equation 5.3 of [1999 Whittaker].
    # These make use the matrix form of the orthogonality relation of equation 3.9
    # in [1999 Whittaker], i.e. `phi_T @ omega_k @ phi = 1`, to compute `phi_T`.
    # Throughout, we use optimized expressions which avoid matrix inversion and
    # matrix-matrix multiplications. More straightforward implementations are in
    # comments.
    #
    # phi_T = jnp.linalg.inv(omega_k @ phi)
    # term1 = diag(q) @ phi_T @ next_omega_k @ next_phi @ diag(1 / next_q)
    term1 = q[..., jnp.newaxis] * solve(
        omega_k @ phi,
        next_omega_k @ next_phi * (1 / next_q)[..., jnp.newaxis, :],
    )
    # term2 = phi_T @ omega_k @ next_phi
    term2 = solve(omega_k @ phi, omega_k @ next_phi)
    i11 = i22 = 0.5 * (term1 + term2)
    i12 = i21 = 0.5 * (-term1 + term2)

    # Phase terms \hat{f}(d) defined near equation 4.2 of [1999 Whittaker]. These
    # describe phase accumulated by propagating across a layer for each eigenmode.
    fd = jnp.exp(1j * q * layer_thickness)
    fd_next = jnp.exp(1j * next_q * next_layer_thickness)

    # Update the s-matrix to include the present layer, following the recipe
    # given in equation 5.4 of [1999 Whittaker].
    s11, s12, s21, s22 = s_matrix_blocks

    # s11_next = inv(i11 - diag(fd) @ s12 @ i21) @ diag(fd) @ s11
    term3 = i11 - fd[..., jnp.newaxis] * s12 @ i21
    s11_next = solve(term3, fd[..., jnp.newaxis] * s11)
    # s12_next = inv(i11 - diag(fd) @ s12 @ i21)
    #            @ (diag(fd) @ s12 @ i22 - i12) @ diag(fd_next)
    s12_next = solve(
        term3,
        (fd[..., jnp.newaxis] * s12 @ i22 - i12) * fd_next[..., jnp.newaxis, :],
    )
    s21_next = s22 @ i21 @ s11_next + s21
    # s22_next = s22 @ i21 @ s12_next + s22 @ i22 @ diag(fd_next)
    s22_next = s22 @ i21 @ s12_next + s22 @ i22 * fd_next[..., jnp.newaxis, :]

    return (s11_next, s12_next, s21_next, s22_next)


def set_end_layer_thickness(
    s_matrix: ScatteringMatrix,
    thickness: jnp.ndarray,
) -> ScatteringMatrix:
    """Returns a new ``ScatteringMatrix`` with a modified end layer thickness.

    Args:
        s_matrix: The initial ``ScatteringMatrix``.
        thickness: The desired thickness of the layer.

    Returns:
        The new ``ScatteringMatrix``.
    """
    q = s_matrix.end_layer_solve_result.eigenvalues
    fd = jnp.exp(1j * q * (thickness - s_matrix.end_layer_thickness))
    return ScatteringMatrix(
        s11=s_matrix.s11,
        s12=s_matrix.s12 * fd[..., jnp.newaxis, :],
        s21=s_matrix.s21,
        s22=s_matrix.s22 * fd[..., jnp.newaxis, :],
        start_layer_solve_result=s_matrix.start_layer_solve_result,
        start_layer_thickness=s_matrix.start_layer_thickness,
        end_layer_solve_result=s_matrix.end_layer_solve_result,
        end_layer_thickness=thickness,
    )


def set_start_layer_thickness(
    s_matrix: ScatteringMatrix,
    thickness: jnp.ndarray,
) -> ScatteringMatrix:
    """Returns a new ``ScatteringMatrix`` with a modified start layer thickness.

    Args:
        s_matrix: The initial ``ScatteringMatrix``.
        thickness: The desired thickness of the layer.

    Returns:
        The new ``ScatteringMatrix``.
    """
    q = s_matrix.start_layer_solve_result.eigenvalues
    fd = jnp.exp(1j * q * (thickness - s_matrix.start_layer_thickness))
    return ScatteringMatrix(
        s11=s_matrix.s11 * fd[..., jnp.newaxis, :],
        s12=s_matrix.s12,
        s21=s_matrix.s21 * fd[..., jnp.newaxis, :],
        s22=s_matrix.s22,
        start_layer_solve_result=s_matrix.start_layer_solve_result,
        start_layer_thickness=thickness,
        end_layer_solve_result=s_matrix.end_layer_solve_result,
        end_layer_thickness=s_matrix.end_layer_thickness,
    )


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


tree_util.register_pytree_node(
    ScatteringMatrix,
    lambda x: (
        (
            x.s11,
            x.s12,
            x.s21,
            x.s22,
            x.start_layer_solve_result,
            x.start_layer_thickness,
            x.end_layer_solve_result,
            x.end_layer_thickness,
        ),
        None,
    ),
    lambda _, x: ScatteringMatrix(*x),
)
