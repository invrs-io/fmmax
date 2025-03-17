# Change log

## Unreleased

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
