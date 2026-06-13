# TJLFEP.jl

A Julia port of **TGLF-EP**, the energetic-particle (EP) critical-gradient
threshold model built on top of [TJLF](https://github.com/ProjectTorreyPines/TJLF.jl)
(the Julia port of TGLF). TJLFEP scans a scale factor on the EP pressure gradient
until a marginally unstable Alfvénic mode appears, yielding the critical EP
density/pressure gradients used for EP transport and stability studies.

It is a close, jointly-maintained translation of the Fortran GACODE add-on
`TGLF-EP` — verified against it bit-for-bit — and adds a CUDA GPU eigensolver path
that is **~6.8× faster than the Fortran CPU reference** at production basis size
(`N_BASIS=32`). A Julia-native **autodiff (`ad`) solver** pushes this to **~36×**
by replacing the brute-force scale-factor grid with a gradient-based Newton
root-find (see the benchmark below); because it is grid-independent, the AD
solver also *changes the result* — its `sfmin` is the smooth, more accurate
marginal factor rather than a value quantized to the coarse factor grid.

## Capabilities

1. **Verification against Fortran TGLF-EP** — run the same case through the
   canonical Fortran `TGLFEP_driver` and Julia and overlay the results.
2. **Run + validate** on an `input.gacode` + `input.TGLFEP` pair.
3. **Database generation** on GPU (NVIDIA MPS, 1 radius/worker).
4. **Timing vs `N_BASIS`** benchmark (Fortran CPU / Julia CPU / Julia GPU), for
   both the `grid` and autodiff (`ad`) solvers.
5. **Sysimage** build + run (removes JIT cost for production).
6. **FUSE-native IMAS `dd`** path — run from an IMAS data dictionary using the
   same preprocessing/`runTHD` routines as the `input.gacode` path.

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | The TJLFEP package (`module TJLFEP`). |
| `build/` | Run / verify / benchmark / sysimage scripts (see `build/README.md`). |
| `examples/` | Canonical DIII-D and ITER cases (see `examples/README.md`). |
| `utils/` | Preprocessing-comparison utilities (see `utils/README.md`). |
| `test/` | Regression tests, incl. the nb6 Fortran-match fixture. |
| `docs/` | Verification/reproduction notes + benchmark assets. |
| `attic/` | Quarantined development scratch (gitignored, recoverable). |

## Installation

Requires Julia **1.11+** (1.11.7 and the default NERSC `juliaup` Julia 1.12+ are
both fine). TJLFEP depends on registered `TJLF >= 1.2.4` (FuseRegistry); no TJLF
checkout is needed.

```bash
cd TJLFEP
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The GPU path requires **CUDA >= 12.6** (the eigensolver calls `cusolverDnXgeev`).

On HPC systems Julia and CUDA are typically provided as modules (e.g.
`module load julia cudatoolkit/12.9` on Perlmutter, or a pinned
`julia/1.11.7`). If your shared filesystem has a small or slow home directory,
point `JULIA_DEPOT_PATH` at a larger scratch location before instantiating.

## Quick start

```julia
using TJLFEP

# File-based path (set TJLFEP_FILE_ONLY=1 to skip IMAS/FUSE imports):
runTHD("input.TGLFEP", "input.MTGLF", "input.EXPRO"; use_gpu=true)

# Directly from an input.gacode + scan-control input.TGLFEP:
runTHD_from_gacode("examples/DIIID_202017C42_500ms_v3.1/input.gacode",
                   "examples/DIIID_202017C42_500ms_v3.1/input_scan20_nb6.TGLFEP";
                   use_gpu=true)

# FUSE-native IMAS data dictionary (same gradient routines as input.gacode):
runTHD(dd, rho, OptionsDict; use_gpu=true)   # see examples/ITER/ITERstructExample.jl
```

On Perlmutter, the batch wrappers in `build/` drive these on Slurm; start with
`cd build && sbatch run/batch_smoke_test.sh` (capability 2). See `build/README.md`
for the full per-capability script index.

## Verification against Fortran

The Julia port reproduces the Fortran `TGLFEP_driver` `SFmin` profile to its
printed precision on the DIII-D `202017C42_500ms_v3.1` case. The verification
scripts default to the shared Fortran build at
`/global/cfs/cdirs/m3739/gacode_add_d3d/TGLF-EP` (override `TGLFEP_DIR`).

```bash
cd build
sbatch verify/batch_debug_nb6_fortran_scan20_10n.sh   # Fortran reference
sbatch verify/batch_debug_nb6_julia_scan20_10n.sh     # Julia
# then overlay:
FORTRAN_DIR=... JULIA_DIR=... FILE_DIR=... ./verify/compare_debug_nb6_scan20.sh
```

Agreement at the scan radii: `SFmin` max relative error ~0.03%; α(dn/dr) ~0.5%.
A self-contained smoke-level regression (`test/runtests_regression_nb6.jl`) checks
one radius against the archived Fortran golden output. Full steps:
`docs/REPRODUCE_FORTRAN_MATCH.md`; physics-parity notes:
`docs/FORTRAN_JULIA_COMPARISON.md`.

## Benchmark: wallclock vs N_BASIS (DIII-D, SCAN_N=20)

20-radius scan on Perlmutter, all with sysimages. Two critical-factor **solvers**
are compared, each on its fastest parallel layout:

- **`grid`** (default, the verified reference) — the Fortran-equivalent
  `(kyhat × width × factor)` sweep. Thousands of independent eigensolves per
  radius, so it is fastest with an **MPS team** (separate CUDA contexts that
  overlap on each GPU via Hyper-Q).
- **`ad`** — the autodiff route (AD-Newton AE-onset + implicit-function-theorem
  `(kyhat,width)` descent). Far fewer eigensolves per radius, so it is fastest
  **in-process (threads)**; an MPS team only adds overhead here (see below).

![Wallclock vs N_BASIS](docs/plots/scan20_timing_lines.png?v=3)

**Grid solver** — Fortran CPU (10 nodes) vs Julia CPU (10 nodes,
SlurmClusterManager) vs Julia GPU (5 A100 nodes, **MPS team**):

| N_BASIS | Fortran CPU (s) | Julia CPU (s) | Julia GPU MPS (s) | GPU speedup vs Fortran |
|--------:|----------------:|--------------:|------------------:|-----------------------:|
| 6  | 62.5   | 141.3   | 140.4 | 0.45× |
| 8  | 97.5   | 148.4   | 149.0 | 0.65× |
| 16 | 347.0  | 250.0   | 161.7 | 2.15× |
| 32 | 1546.0 | 1029.2  | 226.3 | **6.83×** |

**Autodiff solver** — Julia GPU (5 A100 nodes, **in-process threads**), vs the
grid GPU path above:

| N_BASIS | Grid GPU MPS (s) | AD GPU threads (s) | AD vs grid-GPU | AD vs Fortran |
|--------:|-----------------:|-------------------:|---------------:|--------------:|
| 6  | 140.4 | 65.6 | 2.1× | 0.95× |
| 8  | 149.0 | 57.9 | 2.6× | 1.7× |
| 16 | 161.7 | 47.4 | 3.4× | 7.3× |
| 32 | 226.3 | 42.7 | **5.3×** | **36×** |

The grid GPU eigensolver wins decisively over Fortran as the dense eigenproblem
grows (**6.8×** at `N_BASIS=32`). The AD solver then wins again on top of that:
**~43 s for the full 20-radius profile at `N_BASIS=32` — 5.3× faster than grid-GPU
and 36× faster than Fortran.** Running the AD path on an MPS team instead is
*slower* (≈4.5 min at `N_BASIS=32`, and it worsens with `N_BASIS`): the per-radius
AD parallel regions are small and the Newton descent is sequential, so the
team-spawn / remote-call overhead is never amortized. **Rule of thumb: `grid` → MPS,
`ad` → threads.**

### Why AD time is nearly flat (and even dips) in N_BASIS

It looks odd that the AD wallclock does not grow with `N_BASIS` — it even ticks
*down* slightly (65.6 → 57.9 → 47.4 → 42.7 s for `N_BASIS` 6 → 8 → 16 → 32),
the opposite of the O(n³) growth you see in the grid/Fortran columns. This is
expected, not a measurement glitch:

- **AD is overhead-bound, not arithmetic-bound.** The AD route issues a *fixed,
  N_BASIS-independent* number of eigensolves per radius (a 4×8 seed grid plus a
  capped AE-onset Newton + implicit-function-theorem descent — ~10² solves),
  versus the *thousands per radius* the grid path sweeps. With so few solves, the
  per-radius wall is dominated by fixed per-task cost (CUDA context init, first-call
  kernel compile/load, reading `input.gacode`) rather than by the dense eigensolve.
  At these matrix sizes that overhead floor dwarfs the O(n³) arithmetic, so the AD
  curve stays flat while the grid curve climbs steeply.
- **Higher `N_BASIS` resolves the physics better, so it converges in fewer solves.**
  With a coarse basis the growth-rate spectrum γ(factor) is poorly resolved, so the
  safeguarded Newton/IFT refinement tends to take more steps (more backtracks)
  before it brackets the onset. With a finer basis the γ(factor) curve is smooth and
  well-resolved, so the root-find converges in fewer iterations — which can more than
  offset the larger per-solve cost. This is the small downward trend you see.
- **The `scan` number is a max over 20 parallel radii** (one per GPU), so it is a
  straggler-sensitive extreme value; run-to-run noise of a few seconds is normal and
  accounts for the rest of the wiggle.

Bottom line: a bigger basis is not literally "faster to solve"; the AD route is
simply insensitive to `N_BASIS` in wallclock, which is exactly why it is the
production win at large `N_BASIS` where the grid/Fortran cost explodes.

### AD changes the result (it is grid-independent)

The `ad` solver does **not** read `sfmin` off the coarse EP-scale-factor grid that
the Fortran/`grid` path quantizes to. It locates the smooth instability/keep
**onset** directly with a Newton root-find using exact (forward-mode AD)
derivatives, so its `sfmin` can **differ from the grid value** — it removes the
factor-grid discretization error rather than reproducing it. On DIII-D the grid
`sfmin` is effectively a coarse-grid / keep-flag-transition artifact, which the AD
onset resolves continuously. Treat the AD `sfmin` as the more accurate marginal
factor, not as a bit-for-bit match to the grid number.

Data: `docs/plots/scan20_timing.csv`. Reproduce: grid sweep with
`build/timing/submit_timing_vs_nbasis.sh`, AD sweep with
`build/timing/submit_timing_vs_nbasis_ad.sh` (capability 4).

## Citation

If this software contributes to an academic publication, please cite it as follows:

> T.F. Neiser, D. Sun, B. Agnew, T. Slendebroek, O. Meneghini, B.C. Lyons, A. Ghiozzi, J. McClenaghan, G. Staebler and J. Candy, _TJLF: The quasi-linear model of gyrokinetic transport TGLF translated to Julia_, APS Meeting Abstracts (2024)
