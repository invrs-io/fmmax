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
"""Functions related to perfectly matched layers.

Copyright (c) Martin F. Schubert
"""

# ruff: noqa: F401
from fmmax._orig.pml import (
    PMLParams,
    _crop_and_edge_pad_pml_region,
    _normalized_distance_into_pml,
    apply_uniaxial_pml,
)
