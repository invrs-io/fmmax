# Change log

## Unreleased
- Add permittivity shape validation to eigensolve functions.
- Make small changes throughout to acommodate primitive lattice vectors with batch dimensions.
- Update vector field calculation to scan over batch, replacing prior usage of vmap. This seems to compile faster and also actually run faster.
- Update implementation of `scattering._stack_s_matrices` to use scan rather than a Python `for` loop.
- Scalar thickness validation in scattering module.
- Make `omega_script_k_matrix` an attribute of the `LayerSolveResult` rather than a computed property. This avoids redundant computation in cases where jit compilation is not used.
- Provide a private `_fields_on_grid` function which does not require a full layer solve result as input.

## 1.2.0 (April 2, 2025)
- In fields module, allow number of unit cells to be specified independently when Brillouin zone integration is used. Retain ability to infer number of unit cells from Brillouin grid axes in when number of unit cells is not specified.
- Make the `basis.Expansion` hashable.
- Update logic for Brillouin zone integration, so that flux computed from the amplitudes and from the fields are consistent, and source amplitudes computed from fields are consistent from the original amplitudes that yielded the fields. This changes the regression values for the crystal example test, by a factor equal to the the number of points in the Brillouin zone grid.

## 1.1.2 (March 31, 2025)
- Update calculation of `basis.brillouin_zone_in_plane_wavevector` so that wavevectors at the center of the Brillouin zone are exactly at the center. Previously, the manner in which this was calculated could lead to small nonzero values due to floating point calculations. These small values have no effect in virtually all cases.

## 1.1.1 (March 19, 2025)
- Add missing `packaging` dependency (thanks @SamDuffield).

## 1.1.0 (March 17, 2025)
- Enable automatic Brillouin zone integration in functions in the `fields` module.
- Add new `test_fields_bz` module for BZ-related tests.
- Update examples and notebooks to use this, rather than manually carrying out BZ integration.
- Correct docstring for dipole sources in `sources` module; position `(0, 0)` is at the corner of the unit cell, not the center.

## 1.0.0 (March 17, 2025)
- Make some additional functions and modules private, including the `vector` module.
- Update docstrings to improve formatting of sphinx auto-docs.
- Rename args to `time_average_z_poynting_flux` from `*_fields` to `*_field` to be consistent with other functions in the `fields` module.
- Add a notebook that demonstrates calculation of the farfield for a dipole in vacuum.
- Improve detection of 1D permittivity distributions, particularly the case of complex permittivity.
- Add a basic notebook illustrating diffraction calculation.
- API changes
    - Export key functions so they can directly be imported from `fmmax`, e.g. `fmmax.eigensolve_isotropic_media(...)`.
    - Update examples so they import `fmmax` and use exported functions, rather than importing individual modules.
    - Update API docs to include only the exported functions/classes/constants.
    - Update docstrings for enums so they render properly.
    - Rename `amplitudes_interior` to `layer_amplitudes_interior`, for consistency with other functions in the `fields` module.

## 0.14.1 (March 4, 2025)
- Set `Formulation.JONES_DIRECT_FOURIER` as the default formulation for all eigensolve functions in `fmm` module.
- Remove redundant definitions of Poynting flux calculations and use `fields.time_average_z_poynting_flux` instead.
- Include a missing factor of 0.5 in the flux returned by `amplitude_poynting_flux`, `directional_poynting_flux`, `eigenmode_poynting_flux`, and `time_average_z_poynting_flux` functions. The result is that the time-average Poynting flux is computed from the fields by, `sz = 0.5 * real(ex * hy.conj() - ey * hx.conj())`, as it should be. The factor of 0.5 is also missing from S4, GRCWA, and the [S4 reference paper](https://web.stanford.edu/group/fan/publication/Liu_ComputerPhysicsCommunications_183_2233_2012.pdf#page=3.75) (section 5.1), which is how it came to exist in this code.

## 0.14.0 (February 25, 2025)
- Set `Formulation.JONES_DIRECT_FOURIER` as the default formulation for `fmm.eigensolve_isotropic_media`.
- Set `Truncation.CIRCULAR` as the default truncation for `basis.generate_expansion`.
- Expand documentation for `fmm.Formulation` enum, describing each formulation.

## 0.13.4 (January 30, 2025)
- Add a `fields.time_average_z_poynting_flux` function to compute the real-space z-oriented Poynting flux from the real-space electromagnetic fields.

## 0.13.3 (January 25, 2025)
- Avoid post-init validation for `fmm.LayerSolveResult` when the attributes are not arrays, e.g. tracer objects. This allows the `LayerSolveResult` to be returned by a jit-ed function.

## 0.13.2 (January 24, 2025)
- Adjust shapes of transverse and z permeability matrices for uniform isotropic media to ensure they have shapes matching those obtained when performing a patterned layer eigensolve. This allows solve reuslts to be concatenated.

## 0.13.1 (January 15, 2025)
- Relax minimum jax dependency to `>= 0.4.27`. There is a bug in some newer jax versions which causes hanging with the CPU backend (https://github.com/jax-ml/jax/issues/24219).

## 0.13.0 (January 14, 2025)
- Reorganize the project to make the `_fmm_matrices` and `_fft` modules are made private.
- Update the build CI workflow to test older, known-good jax versions.
