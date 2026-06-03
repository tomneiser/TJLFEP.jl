# examples/

Canonical TGLF-EP cases used for verification, validation, and benchmarking.
Run everything from the repo root with the project active
(`module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia`).

## DIIID_202017C42_500ms_v3.1/

DIII-D discharge 202017C42 at 500 ms — the primary Fortran-vs-Julia verification
and GPU-benchmark case.

Inputs:
- `input.gacode` — equilibrium + profiles (GACODE EXPRO format).
- `input.TGLFEP` — single-radius scan-control input.
- `input_singleradius_nb6.TGLFEP` — `N_BASIS=6`, `SCAN_N=1` (quick check).
- `input_scan20_nb{6,8,16,32}.TGLFEP` — `SCAN_N=20` scans at four basis sizes
  (the inputs swept by the timing-vs-N_BASIS benchmark).
- `dump.gacode`, `dump.profile`, `fileInput/` — preprocessed file-based inputs.

Fortran golden references (for trust-building comparisons):
- `out.TGLFEP` — reference `SFmin` profile.
- `out.scalefactor_r*` — per-radius scale-factor references.
- `alpha_dndr_crit.input`, `alpha_dpdr_crit.input` — critical-gradient references.

Case scripts:
- `DIIID_juliaValidation.jl` — IMAS-path validation driver for this case.
- `compare_fortran_julia.jl`, `diagnose_crit_grad.jl` — compare Julia output dirs
  against the Fortran references (`out.TGLFEP`, `alpha_*_crit.input`).
- `plotGrads.jl` — critical-gradient plots.
- `batch_TGLF-EP.sl`, `batchRun.sh`, `submit_sweep.sh` — case run wrappers.

Verification and GPU database-generation runs that use this case live in
`build/` (see `build/README.md`).

## ITER/

ITER case driven two ways through the **same** preprocessing/`runTHD` routines:

- `ITERfromFiles.jl` — file-based path from `input.{TGLFEP,MTGLF,EXPRO}`.
- `ITERstructExample.jl` — FUSE-native IMAS `dd` path: builds a `dd` via
  `FUSE.case_parameters(:ITER)` and calls `runTHD(dd, rho, OptionsDict; ...)`,
  exercising the same `expro_bound_deriv` gradient logic as the `input.gacode`
  path. This is the reference example for capability 6 (FUSE/IMAS `dd`).

```bash
julia --project=. examples/ITER/ITERfromFiles.jl       # file-based ITER run
julia --project=. examples/ITER/ITERstructExample.jl   # FUSE/IMAS dd ITER run
```
