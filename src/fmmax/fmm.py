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
"""Functions related to layer eigenmode calculation for the FMM algorithm.

Copyright (c) Martin F. Schubert
"""

from typing import Tuple

import jax.numpy as jnp

from fmmax import basis

# ruff: noqa: F401
from fmmax._orig.fmm import (
    Formulation,
    LayerSolveResult,
    _eigensolve_patterned_general_anisotropic_media,
    _eigensolve_patterned_isotropic_media,
    _eigensolve_uniform_general_anisotropic_media,
    _eigensolve_uniform_isotropic_media,
    _fourier_matrices_patterned_anisotropic_media,
    _fourier_matrices_patterned_isotropic_media,
    _numerical_eigensolve,
    _select_eigenvalues_sign,
    _validate_and_broadcast,
    eigensolve_anisotropic_media,
    eigensolve_general_anisotropic_media,
    eigensolve_isotropic_media,
)


def broadcast_result(
    layer_solve_result: LayerSolveResult,
    shape: Tuple[int, ...],
) -> LayerSolveResult:
    """Broadcast ``layer_solve_result`` attributes to have specified batch shape."""
    lsr = layer_solve_result  # Alias for brevity.
    n = lsr.expansion.num_terms
    return LayerSolveResult(
        wavelength=jnp.broadcast_to(lsr.wavelength, shape),
        in_plane_wavevector=jnp.broadcast_to(lsr.in_plane_wavevector, shape + (2,)),
        primitive_lattice_vectors=basis.LatticeVectors(
            u=jnp.broadcast_to(lsr.primitive_lattice_vectors.u, shape + (2,)),
            v=jnp.broadcast_to(lsr.primitive_lattice_vectors.v, shape + (2,)),
        ),
        expansion=lsr.expansion,
        eigenvalues=jnp.broadcast_to(lsr.eigenvalues, shape + (2 * n,)),
        eigenvectors=jnp.broadcast_to(lsr.eigenvectors, shape + (2 * n, 2 * n)),
        omega_script_k_matrix=jnp.broadcast_to(
            lsr.omega_script_k_matrix, shape + (2 * n, 2 * n)
        ),
        z_permittivity_matrix=jnp.broadcast_to(
            lsr.z_permittivity_matrix, shape + (n, n)
        ),
        inverse_z_permittivity_matrix=jnp.broadcast_to(
            lsr.inverse_z_permittivity_matrix, shape + (n, n)
        ),
        z_permeability_matrix=jnp.broadcast_to(
            lsr.z_permeability_matrix, shape + (n, n)
        ),
        inverse_z_permeability_matrix=jnp.broadcast_to(
            lsr.inverse_z_permeability_matrix, shape + (n, n)
        ),
        transverse_permeability_matrix=jnp.broadcast_to(
            lsr.transverse_permeability_matrix,
            shape + (2 * n, 2 * n),
        ),
        tangent_vector_field=lsr.tangent_vector_field,
    )
