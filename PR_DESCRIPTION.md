# Run the autodiff solver on the GPU/MPS production route + benchmark

## Summary
Wires the TGLF-EP autodiff (`solver=:ad`) critical-factor path onto the
production GPU route and benchmarks it against the Fortran-equivalent grid path.
Combined with the new TJLF `Complex{Dual}` GPU eigensolve, `use_gpu=true` now
accelerates the AD eigensolves end-to-end on both the `input.gacode` and IMAS
`dd` routes.

## What changed
- **`src/tjlfep_ad_extensions.jl`** — new `_ad_pmap(inner, team)` dispatcher that
  reuses `kwscale_scan`'s chunked MPS-team helpers (`_inner_team_map`); applied to
  the AD path's independent-eval regions (seed grid, AE-band hull scan, faithful
  keep sweep).
- **`src/mainsub.jl`, `src/run_tjlfep_file.jl`, `ext/TJLFEPIMASExt.jl`** — thread
  `inner`/`team` through `_mainsub_ad → critical_factor_optimize →
  marginal_factor_faithful → _ae_unstable_window` (the SPMD gacode + dd routes
  already forwarded `solver`/`inner`/`team`).
- **`src/TJLFEP.jl`** — export `gamma_grad`, `marginal_factor_faithful`,
  `critical_factor_optimize`, `critical_factor_profile`.
- **`build/timing/`** — AD timing batch + submit scripts (`*_julia_gpu_ad.sh`,
  `*_julia_cpu_ad.sh`, `*_julia_gpu_ad_mps.sh`, `submit_timing_vs_nbasis_ad.sh`);
  collector/plotter extended with the AD series.
- **`build/ad/`** — `validate_gpu_dual_grad.jl` (+ batch) and the AD
  validation/benchmark suite.
- **READMEs / `docs/FORTRAN_JULIA_COMPARISON.md`** — document the grid/ad and
  mps/threads tradeoffs and the AD accuracy note.

## Benchmark (DIII-D, SCAN_N=20)
At production `N_BASIS=32`, the AD solver on **GPU + in-process threads** finds
the full 20-radius critical-factor profile in **~43 s — 5.3× faster than the grid
GPU (MPS) path and 36× faster than the Fortran reference.** The MPS team does
**not** help the AD path (small per-radius parallel regions + a sequential Newton
descent), so it is ~1.6–6× slower and scales the wrong way. **Rule of thumb:
`grid` → MPS, `ad` → threads.**

## Result change (intentional)
The `ad` solver is **grid-independent**: it locates the smooth instability/keep
onset by a Newton root-find with exact AD derivatives rather than quantizing to
the coarse EP-scale-factor grid, so its `SFmin` can differ from — and is more
accurate than — the Fortran/`grid` value. It is **not** expected to match Fortran
bit-for-bit (the `grid` solver remains the verified reference).

## Bug fix
Fixes `invalid redefinition of constant Main.GACODE` on the GPU sysimage route:
the generic sysimage bakes the `GACODE` package module into `Main`, colliding
with the task scripts' top-level `const GACODE = …`. Renamed those locals to
`GACODE_PATH` and `let`-wrapped the precompile workloads so they leak nothing into
`Main`. This also un-breaks the grid GPU+MPS path.
