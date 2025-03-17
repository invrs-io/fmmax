# Copyright (c) 2025 Martin F. Schubert

__version__ = "v1.1.0"

# ruff: noqa: F401
from fmmax.basis import (
    Expansion,
    LatticeVectors,
    Truncation,
)
from fmmax.basis import X as _X
from fmmax.basis import Y as _Y
from fmmax.basis import (
    brillouin_zone_in_plane_wavevector,
    generate_expansion,
    min_array_shape_for_expansion,
    plane_wave_in_plane_wavevector,
    transverse_wavevectors,
    unit_cell_coordinates,
)
from fmmax.beams import shifted_rotated_fields
from fmmax.farfield import farfield_integrated_flux, farfield_profile
from fmmax.fields import (
    amplitude_poynting_flux,
    colocate_amplitudes,
    directional_poynting_flux,
    eigenmode_poynting_flux,
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
    time_average_z_poynting_flux,
)
from fmmax.fmm import (
    Formulation,
    LayerSolveResult,
    eigensolve_anisotropic_media,
    eigensolve_general_anisotropic_media,
    eigensolve_isotropic_media,
)
from fmmax.pml import PMLParams, apply_uniaxial_pml
from fmmax.scattering import (
    ScatteringMatrix,
    append_layer,
    prepend_layer,
    redheffer_star_product,
    stack_s_matrices_interior,
    stack_s_matrix,
    stack_s_matrix_scan,
)
from fmmax.sources import (
    amplitudes_for_fields,
    amplitudes_for_source,
    dirac_delta_source,
    gaussian_source,
)
from fmmax.translate import translate_layer_solve_result
from fmmax.utils import angular_frequency_for_wavelength, interpolate_permittivity

X = _X  # basis.X
"""Unit vector pointing in the x direction."""

Y = _Y  # basis.Y
"""Unit vector pointing in the y direction."""
