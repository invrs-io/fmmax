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
"""Functions related to vectors and field expansion in the FMM scheme.

Copyright (c) Martin F. Schubert
"""

# ruff: noqa: F401
from fmmax._orig.basis import (
    Expansion,
    LatticeVectors,
    Truncation,
    X,
    Y,
    _basis_coefficients_circular,
    _basis_coefficients_parallelogramic,
    _cross_product,
    _HashableArray,
    _reciprocal,
    brillouin_zone_in_plane_wavevector,
    generate_expansion,
    min_array_shape_for_expansion,
    plane_wave_in_plane_wavevector,
    transverse_wavevectors,
    unit_cell_coordinates,
    validate_shape_for_expansion,
)
