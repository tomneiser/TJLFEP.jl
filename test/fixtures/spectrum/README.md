# process_in=3 (spectrum mode) Fortran golden

Reference `out.eigenvalue_m{1,2,4}_r040` for the DIII-D 202017C42 case
(`examples/DIIID_202017C42_500ms_v3.1/input.gacode`), single radius `ir=40`, fixed
`width=1.5`, `ky_model=0`, `nbasis=32`, `nky=30`, `ky=0.15`. Used by
`test/runtests_regression_spectrum.jl`.

`input_spectrum.TGLFEP` is the Julia/Fortran input (SCAN_N=1, IRS=40).
`input_fortran_scan5.TGLFEP` is the SCAN_N=5 (IRS=20 -> radii 20,40,60,80,101) input
actually used to generate the golden in one job; radius 40 matches `input_spectrum.TGLFEP`.

## How the golden was generated

The public reference binary `$CFS/m3739/gacode_add_d3d/TGLF-EP/TGLFEP_driver` has a bug on
the `process_in=3` + `INPUT_PROFILE_METHOD=2` (EXPRO/gacode) path: `q_scale` (and
`scan_method`, `pprime_method`) are only read for `process_in in {4,5,6}`, so for
`process_in=3` they stay uninitialized. `q(:) = q_scale*EXPRO_q(:)` then zeroes `q`,
giving singular geometry and `DSYEV failed in ave_inv0` at every radius.

A patched binary was built (`/pscratch/sd/t/tneiser/tglfep_build`, from a copy of the
reference source) with three one-line initializers in `TGLFEP_interface.f90`:

```
integer :: scan_method = 0      ! no factor-driven EP scaling
integer :: pprime_method = 0    ! -> fixed p_prime (default branch)
real    :: q_scale = 1.0         ! no q rescale
```

(plus removal of an erroneous extra `STATUS` arg in five `MPI_SEND` calls so it compiles
under nvfortran's F08 MPI module; those calls are not on the process_in=3 path). These
defaults match what TJLFEP's `readTGLFEP` now sets for a `process_in=3` input, so the
comparison is apples-to-apples.

Build:

```bash
export GACODE_ROOT=/pscratch/sd/t/tneiser/gacode_cpu/gacode
export GACODE_PLATFORM=PERLMUTTER_CPU
source $GACODE_ROOT/platform/env/env.PERLMUTTER_CPU
cd /pscratch/sd/t/tneiser/tglfep_build && make
```

Run (5 radii x 3 spectrum colors x 4 ranks/color):

```bash
# input.TGLFEP = input_fortran_scan5.TGLFEP, input.gacode present in cwd
srun -n 60 /pscratch/sd/t/tneiser/tglfep_build/TGLFEP_driver
```

## Agreement

Julia (`run_gacode_scan_task(..., 1)` on `input_spectrum.TGLFEP`) reproduces these spectra
to ~6-7 significant figures across all 4 modes and 30 ky (e.g. mode 1, ky=0.02:
Julia gamma=0.0170061 vs Fortran 0.0170059; factor 1.4139483 vs 1.413948309666269).
