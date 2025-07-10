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
"""Defines a jax-differentiable eigendecomposition.

Copyright (c) Martin F. Schubert
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from fmmax import misc

# The `jeig` package offers several jax-wrapped implementations of eigendecomposition,
# some of which have performance benefits. However, since `jeig` has a dependency on
# pytorch, we make its use optional. If `jeig` is not available, we fall back on a
# pure-jax implementation of the eigendecomposition.
try:
    import jeig

    _JEIG_AVAILABLE = True
except ModuleNotFoundError:
    _JEIG_AVAILABLE = False


_EIG_EPS_RELATIVE = 1e-12
_EIG_EPS_MINIMUM = 1e-24


@jax.custom_vjp
def eig(
    matrix: jnp.ndarray,
    eps_relative: float = _EIG_EPS_RELATIVE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

    The custom vjp allows gradients with resepct to the eigenvectors, unlike the
    standard jax implementation of `eig`. We use an expression for the gradient
    given in [2019 Boeddeker] along with a regularization scheme that applies
    a Lorentzian broadening to a term containing the inverse difference of
    eigenvalues. The broadening is related to the maximum magnitude of the
    eigenvalues.

    [2019 Boeddeker] https://arxiv.org/abs/1701.00392

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps_relative: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps_relative
    return _eig(matrix)


def _eig_jax(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jax.lax.linalg.eig`."""
    eigenvalues, eigenvectors = jax.lax.linalg.eig(
        matrix,
        compute_left_eigenvectors=False,
        use_magma=False,
    )
    return eigenvalues, eigenvectors


def _eig(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition using `jeig` if available, and `_eig_jax` if not."""
    if _JEIG_AVAILABLE:
        return jeig.eig(matrix)
    else:
        return _eig_jax(matrix)


def _eig_fwd(
    matrix: jnp.ndarray,
    eps_relative: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = _eig(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps_relative)


def _eig_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, float],
    grads: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None]:
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps_relative = res
    grad_eigenvalues, grad_eigenvectors = grads

    # The expression for gradient of matrix eigenvectors with respect to matrix values
    # contains a difference between eigenvalues in the denominator. This causes
    # numerical problems in situations where eigenvalues are degenerate or nearly
    # degenerate. This problem is addressed in a few different ways in existing rcwa
    # codes. Defining `delta = eigval_i - eigval_j`,
    #
    # - torcwa uses Lorentzian broadening of the form
    #      1 / delta -> delta.conj / (abs(delta)**2 + eps)
    #   with an eps value of 1e-10.
    #   (https://github.com/kch3782/torcwa/blob/main/torcwa/torch_eig.py#L29)
    #
    # - tf_rcwa uses an expression similar to torcwa, but with apparently different
    #   conjugation. Their value of eps is 1e-6. (We tested their implementation and
    #   found it fails some of our gradient validation tests.)
    #   (https://github.com/scolburn54/rcwa_tf/blob/master/src/tensor_utils.py#L89)
    #
    # - grcwa uses `1 / delta -> 1 / (delta + eps)` with an eps value of 1e-10.
    #   (https://github.com/weiliangjinca/grcwa/blob/master/grcwa/primitives.py#L44)
    #
    # Also, from the wider literature:
    #
    # - Liao et al. (https://arxiv.org/pdf/1903.09650 sec III.A.1) uses Lorentzian
    #   broadening with an eps value of 1e-12.
    #
    # One issue with Lorentzian broadening and a fixed eps is the independence from
    # matrix scale. While a given eps may be appropriate for a matrix, it would be
    # inappropriate for the same matrix scaled by e.g. 1e-6. (The gradient should
    # be identical, except scaled by 1 / 1e-6.)
    #
    # Therefore, we use Lorentzian broadening similar to torcwa, but with an eps
    # value that is computed in a way that considers the eigenvalue range.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    delta_eig = eigenvalues_i - eigenvalues_j
    eig_range_sq = jnp.amax(jnp.abs(delta_eig) ** 2, axis=(-2, -1), keepdims=True)
    eps = jnp.maximum(eps_relative * eig_range_sq, _EIG_EPS_MINIMUM)
    f_broadened = delta_eig.conj() / (jnp.abs(delta_eig) ** 2 + eps)

    # Manually set the diagonal elements to zero, as we do not use broadening here.
    i = jnp.arange(f_broadened.shape[-1])
    f_broadened = f_broadened.at[..., i, i].set(0)

    # By jax convention, gradients are with respect to the complex parameters, not with
    # respect to their conjugates. Take the conjugates.
    grad_eigenvalues_conj = jnp.conj(grad_eigenvalues)
    grad_eigenvectors_conj = jnp.conj(grad_eigenvectors)

    eigenvectors_adj = misc.matrix_adjoint(eigenvectors)
    dim = eigenvalues.shape[-1]
    eye_mask = jnp.eye(dim, dtype=bool)
    eye_mask = eye_mask.reshape((1,) * (eigenvalues.ndim - 1) + (dim, dim))

    # Then, the gradient is found by equation 4.77 of [2019 Boeddeker].
    rhs = (
        misc.diag(grad_eigenvalues_conj)
        + jnp.conj(f_broadened) * (eigenvectors_adj @ grad_eigenvectors_conj)
        - jnp.conj(f_broadened)
        * (eigenvectors_adj @ eigenvectors)
        @ jnp.where(eye_mask, jnp.real(eigenvectors_adj @ grad_eigenvectors_conj), 0.0)
    ) @ eigenvectors_adj
    grad_matrix = jnp.linalg.solve(eigenvectors_adj, rhs)

    # Take the conjugate of the gradient, reverting to the jax convention
    # where gradients are with respect to complex parameters.
    grad_matrix = jnp.conj(grad_matrix)

    # Return `grad_matrix`, and `None` for the gradient with respect to `eps`.
    return grad_matrix, None


eig.defvjp(_eig_fwd, _eig_bwd)
