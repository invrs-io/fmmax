"""Functions related to layer eigenmode calculation for the FMM algorithm.

Copyright (c) 2023 Meta Platforms, Inc. and affiliates.
"""

import dataclasses
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as onp

from fmmax import basis, misc


@dataclasses.dataclass
class LayerSolveResult:
    """Stores the result of a layer eigensolve.

    This eigenvalue problem is specified in equation 28 of [2012 Liu].

    Attributes:
        wavelength: The wavelength for the solve.
        in_plane_wavevector: The in-plane wavevector for the solve.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        expansion: The expansion used for the eigensolve.
        eigenvalues: The layer eigenvalues.
        eigenvectors: The layer eigenvectors.
        omega_script_k_matrix: The omega-script-k matrix from equation 26 of [2012 Liu].
        z_permittivity_matrix: The fourier-transformed zz-component of permittivity.
        inverse_z_permittivity_matrix: The fourier-transformed inverse of zz-component
            of permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix which relates
            the electric field and electric displacement fields.
        z_permeability_matrix: The fourier-transformed zz-component of permeability.
        inverse_z_permeability_matrix: The fourier-transformed inverse of zz-component
            of permeability.
        transverse_permeability_matrix: The transverse permeability matrix, needed to
            calculate the omega-script-k matrix from equation 26 of [2012 Liu]. This
            is needed to generate the layer scattering matrix.
        tangent_vector_field: The tangent vector field ``(tx, ty)`` used to compute the
            transverse permittivity matrix, if a vector FMM formulation is used. If
            the ``FFT`` formulation is used, the vector field is ``None``.
    """

    wavelength: jnp.ndarray
    in_plane_wavevector: jnp.ndarray
    primitive_lattice_vectors: basis.LatticeVectors
    expansion: basis.Expansion
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    omega_script_k_matrix: jnp.ndarray
    z_permittivity_matrix: jnp.ndarray
    inverse_z_permittivity_matrix: jnp.ndarray
    transverse_permittivity_matrix: jnp.ndarray
    z_permeability_matrix: jnp.ndarray
    inverse_z_permeability_matrix: jnp.ndarray
    transverse_permeability_matrix: jnp.ndarray
    tangent_vector_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]]

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self.eigenvectors.shape[:-2]

    def __post_init__(self) -> None:
        """Validates shapes of the ``LayerSolveResult`` attributes."""
        # Avoid validation when attributes are e.g. tracers.
        if not isinstance(self.eigenvalues, (jnp.ndarray, onp.ndarray)):
            return

        required_dtype = self.eigenvalues.dtype
        if self.eigenvectors.dtype != required_dtype:
            raise ValueError(
                f"`eigenvectors` should have dtype {required_dtype} but got "
                f"{self.eigenvectors.dtype}"
            )
        if self.z_permittivity_matrix.dtype != required_dtype:
            raise ValueError(
                f"`z_permittivity_matrix` should have dtype {required_dtype} but got "
                f"{self.z_permittivity_matrix.dtype}"
            )

        if self.inverse_z_permittivity_matrix.dtype != required_dtype:
            raise ValueError(
                f"`inverse_z_permittivity_matrix` should have dtype {required_dtype} "
                f"but got {self.inverse_z_permittivity_matrix.dtype}"
            )
        if self.z_permeability_matrix.dtype != required_dtype:
            raise ValueError(
                f"`z_permeability_matrix` should have dtype {required_dtype} but got "
                f"{self.z_permeability_matrix.dtype}"
            )
        if self.transverse_permeability_matrix.dtype != required_dtype:
            raise ValueError(
                f"`transverse_permeability_matrix` should have dtype {required_dtype} "
                f"but got {self.transverse_permeability_matrix.dtype}"
            )

        def _incompatible(arr: jnp.ndarray, reference_shape: Tuple[int, ...]) -> bool:
            ndim_mismatch = arr.ndim != len(reference_shape)
            batch_compatible = misc.batch_compatible_shapes(arr.shape, reference_shape)
            return ndim_mismatch or not batch_compatible

        if _incompatible(self.wavelength, self.batch_shape):
            raise ValueError(
                f"`wavelength` must have compatible batch shape, but got shape "
                f"{self.wavelength.shape} when `eigenvectors` shape is "
                f"{self.eigenvectors.shape}."
            )
        if _incompatible(self.in_plane_wavevector, self.batch_shape + (2,)):
            raise ValueError(
                f"`in_plane_wavevector` must have compatible batch shape, but got "
                f"shape {self.in_plane_wavevector.shape} when `eigenvectors` shape is "
                f"{self.eigenvectors.shape}."
            )
        if _incompatible(self.primitive_lattice_vectors.u, self.batch_shape + (2,)):
            raise ValueError(
                f"`primitive_lattice_vectors.u` must have compatible batch shape, but "
                f"got shape {self.primitive_lattice_vectors.u.shape} when "
                f"`eigenvectors` shape is {self.eigenvectors.shape}."
            )
        if _incompatible(self.primitive_lattice_vectors.v, self.batch_shape + (2,)):
            raise ValueError(
                f"`primitive_lattice_vectors.v` must have compatible batch shape, but "
                f"got shape {self.primitive_lattice_vectors.v.shape} when "
                f"`eigenvectors` shape is {self.eigenvectors.shape}."
            )
        if self.expansion.num_terms * 2 != self.eigenvectors.shape[-1]:
            raise ValueError(
                f"`eigenvectors` must have shape compatible with `expansion.num_terms`,"
                f" but got shape {self.eigenvectors.shape} when `num_terms` shape is "
                f"{self.expansion.num_terms}."
            )
        if self.eigenvalues.shape != self.eigenvectors.shape[:-1]:
            raise ValueError(
                f"`eigenvalues` must have compatible shape, but got shape "
                f"{self.eigenvalues.shape} when `eigenvectors` shape is "
                f"{self.eigenvectors.shape}."
            )
        if self.omega_script_k_matrix.shape != self.eigenvectors.shape:
            raise ValueError(
                f"`omega_script_k_matrix` must have shape matching `eigenvectors` "
                f"shape, but got {self.eigenvalues.shape} when `eigenvectors` shape "
                f"is {self.eigenvectors.shape}."
            )

        expected_matrix_shape = self.batch_shape + (self.expansion.num_terms,) * 2
        if _incompatible(self.inverse_z_permittivity_matrix, expected_matrix_shape):
            raise ValueError(
                f"`inverse_z_permittivity_matrix` must have shape compatible with "
                f"`eigenvectors`, but got shapes "
                f"{self.inverse_z_permittivity_matrix.shape} "
                f"and {self.eigenvectors.shape}."
            )
        if _incompatible(self.z_permittivity_matrix, expected_matrix_shape):
            raise ValueError(
                f"`z_permittivity_matrix` must have shape compatible with "
                f"`eigenvectors`, but got shapes {self.z_permittivity_matrix.shape} "
                f"and {self.eigenvectors.shape}."
            )
        if _incompatible(self.transverse_permittivity_matrix, self.eigenvectors.shape):
            raise ValueError(
                f"`transverse_permittivity_matrix` must have shape compatible with "
                f"`eigenvectors`, but got shapes "
                f"{self.transverse_permittivity_matrix.shape} and "
                f"{self.eigenvectors.shape}."
            )
        if _incompatible(self.inverse_z_permeability_matrix, expected_matrix_shape):
            raise ValueError(
                f"`inverse_z_permeability_matrix` must have shape compatible with "
                f"`eigenvectors`, but got shapes "
                f"{self.inverse_z_permeability_matrix.shape} "
                f"and {self.eigenvectors.shape}."
            )
        if _incompatible(self.z_permeability_matrix, expected_matrix_shape):
            raise ValueError(
                f"`z_permeability_matrix` must have shape compatible with "
                f"`eigenvectors`, but got shapes {self.z_permeability_matrix.shape} "
                f"and {self.eigenvectors.shape}."
            )
        if _incompatible(self.transverse_permeability_matrix, self.eigenvectors.shape):
            raise ValueError(
                f"`transverse_permeability_matrix` must have shape compatible with "
                f"`eigenvectors`, but got shapes "
                f"{self.transverse_permeability_matrix.shape} and "
                f"{self.eigenvectors.shape}."
            )

        if self.tangent_vector_field is not None and (
            self.tangent_vector_field[0].ndim != self.eigenvectors.ndim
        ):
            raise ValueError(
                f"`tangent_vector_field` must have ndim compatible with "
                f"`eigenvectors`, but got shapes {self.tangent_vector_field[0]} and "
                f"{self.eigenvectors}."
            )
