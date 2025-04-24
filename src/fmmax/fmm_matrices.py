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
"""Functions that generate various matrices for the FMM problem.

Copyright (c) Martin F. Schubert
"""

# ruff: noqa: F401
from fmmax._orig.fmm_matrices import (
    _rotation_matrices,
    _tangent_terms,
    k_matrix_patterned,
    omega_script_k_matrix_patterned,
    script_k_matrix_patterned,
    script_k_matrix_uniform,
    transverse_permeability_fft_anisotropic,
    transverse_permeability_vector_anisotropic,
    transverse_permittivity_fft,
    transverse_permittivity_fft_anisotropic,
    transverse_permittivity_vector,
    transverse_permittivity_vector_anisotropic,
)
