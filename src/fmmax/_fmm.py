"""Functions related to layer eigenmode calculation for the FMM algorithm.

Copyright (c) 2025 invrs.io LLC
"""

import enum
from typing import Callable, Tuple

import jax.numpy as jnp

from fmmax import basis, vector
from fmmax._fmm_result import LayerSolveResult

# xx, xy, yx, yy, and zz components of permittivity or permeability.
TensorComponents = Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]
VectorFn = Callable[
    [jnp.ndarray, basis.Expansion, basis.LatticeVectors],
    Tuple[jnp.ndarray, jnp.ndarray],
]


@enum.unique
class Formulation(enum.Enum):
    """Enumerates supported Fourier modal method formulations.

    Each formulation specifies an algorithm to compute the transverse permittivity
    matrix used in the Fourier modal method. The simplest formulation is ``FFT``, in
    which the blocks of the transverse permittivity matrix are simply the Fourier
    convolution matrices for their respective permittivity tensor components.

    The remaining formulations are so-called vector formulations, which make use of a
    vector field generated in the unit cell of the FMM calculation. The vector field
    defines a local coordinate system that tangent and normal to the interfaces of
    features in the unit cell, allowing improved convergence through independent
    treatment of field components that are tangent and normal to the interfaces.
    """

    #: The simplest formulation, which does not consider the orientation of
    #: interfaces of features in a permittivity array.
    FFT = "fft"

    #: Generates a complex linear vector field which has maximum magnitude ``1`` and
    #: a null in the interior of features. In the objective, the gradient of the
    #: vector field on the real-space grid is computed; a penalty term discourages
    #: non-smooth fields (i.e. fields whose gradient is large).
    POL = vector.POL

    #: Takes the field computed by ``POL`` and normalizes so the magnitude is ``1``
    #: evereywhere in the unit cell. Where ``POL`` has zeros, ``NORMAL`` has
    #: discontinuities.
    NORMAL = vector.NORMAL

    #: Takes the field computed by ``POL``, and converts it to a complex
    #: elliptical field which has magnitude ``1`` everywhere and lacks discontinuities.
    JONES = vector.JONES

    #: Directly computes a complex elliptical vector field without first finding
    #: a linear vector field. Smoothness is evaluated on the real-space grid.
    JONES_DIRECT = vector.JONES_DIRECT

    #: Generates a complex linear vector field, but with an alternate method of
    #: penalizing non-smoothness. Specifically, the Fourier components corresponding to
    #: high spatial frequencies are penalized. Compared to ``POL``, ``POL_FOURIER``
    #: can be computed more efficiently.
    POL_FOURIER = vector.POL_FOURIER

    #: Takes the field computed by ``POL_FOURIER``and normalizes so the magnitude is
    #: ``1`` evereywhere in the unit cell.
    NORMAL_FOURIER = vector.NORMAL_FOURIER

    #: Takes the field computed by ``POL_FOURIER`` and converts it to a complex
    #: elliptical field.
    JONES_FOURIER = vector.JONES_FOURIER

    #: Directly computes a complex elliptical vector field, using Fourier coefficients
    #: to penalize non-smoothness.
    JONES_DIRECT_FOURIER = vector.JONES_DIRECT_FOURIER


def broadcast_result(
    layer_solve_result: "LayerSolveResult",
    shape: Tuple[int, ...],
) -> "LayerSolveResult":
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
        transverse_permittivity_matrix=jnp.broadcast_to(
            lsr.transverse_permittivity_matrix,
            shape + (2 * n, 2 * n),
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
