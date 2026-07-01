[![codecov](https://codecov.io/github/projecttorreypines/tjlfep.jl/graph/badge.svg?token=WIeugjkmVB)](https://codecov.io/github/projecttorreypines/tjlfep.jl)
![Docs](https://github.com/ProjectTorreyPines/TJLFEP.jl/actions/workflows/make_docs.yml/badge.svg)

# TJLFEP.jl

A Julia port of **TGLF-EP**, the energetic-particle (EP) critical-gradient
threshold model built on top of [TJLF](https://github.com/ProjectTorreyPines/TJLF.jl)
(the Julia port of TGLF). TJLFEP scans a scale factor on the EP pressure gradient
until a marginally unstable Alfvénic mode appears, yielding the critical EP
density/pressure gradients used for EP transport and stability studies.

It is a close, jointly-maintained translation of the Fortran GACODE add-on
`TGLF-EP` (verified against it bit-for-bit), and adds a **CUDA GPU eigensolver**
plus Julia-native **autodiff (`ad`) solvers**. Together these make TJLFEP much
faster than the Fortran CPU reference and, through the `ad` solvers, *more
accurate*: they resolve the narrow-width (`w<1`) EP-driven Alfvén modes that
Fortran's fixed `w≥1` factor grid cannot.

**Which solver?** Choose along two axes. Do you want to *match* Fortran or
*improve* on it, and do you want the faithful value or a faster approximation?

|                                          | Faithful                                       | Faster approximation                          |
| ---------------------------------------- | ---------------------------------------------- | --------------------------------------------- |
| **Match Fortran** (`w≥1` box)            | **`:grid`** reproduces Fortran (**~13×**)      | **`:ad :only`** approximates `:grid` (**~20×**) |
| **Extend Fortran** (adds `w<1` AE modes) | **`:ad :locate`** *(default)* (**~4.7×**)      | **`:ad :wide`** (**~9×**)                     |

Speedups are **node-hours vs the Fortran CPU reference** at `N_BASIS=32` (the fair,
node-count-normalized cost; see the [benchmark](#benchmark-cost-vs-n_basis-diii-d-scan_n20)
below). `:ad :locate` is the production default (the `ActorTJLFEP` default) and
reports the faithful narrow-width value.

For the full API reference, see the
[online documentation](https://projecttorreypines.github.io/TJLFEP.jl/dev).

## Capabilities

1. **Verification against Fortran TGLF-EP**: run the same case through the
   canonical Fortran `TGLFEP_driver` and Julia and overlay the results.
2. **Run + validate** on an `input.gacode` + `input.TGLFEP` pair.
3. **Database generation** on GPU (NVIDIA MPS, 1 radius/worker).
4. **Timing vs `N_BASIS`** benchmark (Fortran CPU / Julia CPU / Julia GPU), for
   both the `grid` and autodiff (`ad`) solvers.
5. **Sysimage** build + run (removes JIT cost for production).
6. **FUSE-native IMAS `dd`** path: run from an IMAS data dictionary using the
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

# File-based path (loaded standalone, TJLFEP stays light, no IMAS/FUSE):
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

## Benchmark: cost vs N_BASIS (DIII-D, SCAN_N=20)

20-radius scan on Perlmutter, all with sysimages. **Cost is reported in node-hours**
(nodes × wallclock), the fair, layout-independent metric (Fortran runs on 10 CPU
nodes, the GPU tiers on 5). Each solver runs on its fastest parallel layout
(**rule of thumb: `grid` → MPS team, `ad` → in-process threads**; an MPS team only
adds overhead to `ad`, see below).

The solvers, with node-hours vs Fortran at `N_BASIS=32`:

| Solver | vs Fortran | What it computes |
| --- | ---: | --- |
| **`:ad :only`** | **~20×** | Approximates `:grid`: a smooth, de-quantized version of the `w≥1` result (median `:only/grid ≈ 0.9`). Fast iteration only; misses the `w<1` edge modes. |
| **`:grid`** | **~13×** | The verified Fortran-equivalent `(kyhat × width × factor)` sweep (thousands of eigensolves/radius). |
| **`:ad :wide`** | **~9×** | Adds the narrow `w<1` AE modes in one log-seeded pass; conservative (within ~1–2× of `:ad :locate`, never below it). Bulk NN-DB generation. |
| **`:ad :locate`** *(default)* | **~4.7×** | Adds the narrow `w<1` AE modes; the faithful narrow-width value. Production. |

The `ad` solvers are **grid-independent**: rather than reading `sfmin` off the coarse
factor grid, they locate the instability onset directly (Newton root-find with exact
forward-mode AD derivatives). `:grid`/`:ad :only` stay in the Fortran `w≥1` box;
`:ad :locate`/`:ad :wide` additionally resolve the narrow-width (`w<1`)
EP-driven modes it excludes (the *entire* `sfmin` reduction below grid, up to ~12× at
the edge). Higher-fidelity internal reference tiers (`robust_ad`, and the
`nbasis`-converged `truth`) sit above `:ad :locate`, which matches them essentially
bit-for-bit; they are documented for reference in
[`docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md`](https://github.com/ProjectTorreyPines/TJLFEP.jl/blob/master/docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md).

**Accuracy**: `sfmin(IR)` for all solvers at `N_BASIS=32`:

![sfmin vs radius: Fortran vs grid vs :ad :only vs :ad :locate vs :ad :wide](https://raw.githubusercontent.com/ProjectTorreyPines/TJLFEP.jl/master/docs/plots/ad_wide_accuracy_nb32.png?v=5)

`:ad :locate` (green) drops well below `grid` (grey, which reproduces the blue Fortran
reference bit-for-bit) and `:ad :only` (orange) into the narrow-width modes at the outer
radii (IR ≳ 65; up to ~12× below grid). `:ad :wide` (dark red) recovers nearly all of that
in a single pass, within ~1–2× of `:ad :locate` (a mild, conservative over-prediction,
e.g. ~1.7× at IR=64) and never below it. Colors and legend order match the node-hours
plot below for quick cross-reference.

![Node-hours vs N_BASIS](https://raw.githubusercontent.com/ProjectTorreyPines/TJLFEP.jl/master/docs/plots/scan20_timing_wide_lines.png?v=6)

Absolute node-hours at `N_BASIS=32` (1-node-backfill layout, 4 GPU workers draining a
20-radius queue): Fortran ≈4.3; `grid` ≈0.31, `:ad :only` ≈0.21, `:ad :wide` ≈0.46,
`:ad :locate` ≈0.92. `:ad :wide` is ~2× cheaper than `:locate`
(the gap widens at higher `N_BASIS`, where the faithful confirms cost more). The
per-backend tables below give the raw wallclock seconds.

**Best-throughput layout depends on the solver** (for these `SCAN_N=20` runs):
- **`:grid` and `:ad :only` → 5 GPU nodes.** Per-radius cost is uniform, so
  spreading the 20 radii across 5 nodes (~4 radii/node) minimizes wallclock with
  no wasted node-hours.
- **`:ad :locate` and `:ad :wide` → 1 GPU node, backfill.** Their edge radii take
  much longer (the narrow-width `w<1` locate is triggered there), so a fixed
  multi-node split would leave nodes idle waiting on the straggler edge radii.
  Running a single node with 4 workers draining a shared 20-radius claim queue
  keeps every GPU busy and gives the lowest node-hours.

**Grid solver**: Fortran CPU (10 nodes) vs Julia CPU (10 nodes,
SlurmClusterManager) vs Julia GPU (5 A100 nodes, **MPS team**):

| N_BASIS | Fortran CPU (s) | Julia CPU (s) | Julia GPU MPS (s) | GPU speedup vs Fortran (wallclock) |
|--------:|----------------:|--------------:|------------------:|-----------------------------------:|
| 6  | 62.5   | 141.3   | 140.4 | 0.45× |
| 8  | 97.5   | 148.4   | 149.0 | 0.65× |
| 16 | 347.0  | 250.0   | 161.7 | 2.15× |
| 32 | 1546.0 | 1029.2  | 226.3 | **6.83×** |

(Wallclock here compares a 5-node GPU run to a 10-node Fortran run; the fair
node-hours comparison is in the node-hours plot/section above.)

**`:ad :only` (bare `w≥1` AD, no faithful confirm)**: Julia GPU (5 A100 nodes,
**in-process threads**), vs the grid GPU path (both 5-node, so this wallclock ratio is
apples-to-apples):

| N_BASIS | Grid GPU MPS (s) | `:ad :only` GPU threads (s) | `:only` vs grid-GPU (wallclock) |
|--------:|-----------------:|----------------------------:|--------------------------------:|
| 6  | 140.4 | 69.3  | 2.0× |
| 8  | 149.0 | 69.5  | 2.1× |
| 16 | 161.7 | 103.8 | 1.6× |
| 32 | 226.3 | 153.9 | **1.5×** |

The vs-grid wallclock advantage *shrinks* with `N_BASIS` (2.0× → 1.5×) because
`:ad :only`'s wallclock grows faster with `N_BASIS` than the grid MPS path's. Note this
is the *same* `ad` solver run two ways: on an **MPS team** instead of threads it is
*slower* (≈4.5 min at `N_BASIS=32`, worsening with `N_BASIS`): the per-radius AD regions
are small and the descent is sequential, so team-spawn/remote-call overhead never
amortizes. **Rule of thumb: `grid` → MPS, `ad` → threads.**

The `nbasis`-converged `truth` reference tier (validation only, ~34 min for the full
profile at `N_BASIS=32`) is documented in
[`docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md`](https://github.com/ProjectTorreyPines/TJLFEP.jl/blob/master/docs/AD_SOLVERS_AND_SEARCH_BOUNDS.md).

Data: `docs/plots/scan20_timing.csv`. Reproduce: grid sweep with
`build/timing/submit_timing_vs_nbasis.sh`, AD sweep with
`build/timing/submit_timing_vs_nbasis_ad.sh` (capability 4).

## Spectrum diagnostic (`PROCESS_IN=3`)

Besides the critical-gradient **threshold** scans (`PROCESS_IN=4/5`), TJLFEP supports
the TGLF-EP **spectrum diagnostic** (`PROCESS_IN=3`). It does *not* scan or search for a
critical factor; it just computes the linear growth-rate / real-frequency spectra
γ(ky), ω(ky) at fixed plasma conditions, by running the full TGLF transport model over a
fixed `ky` grid (`nky=30`, `ky=0.15`, `KY_MODEL=0`, `NBASIS=32`) at three drives:

| `mode_in` | file | drive |
|-----------|------|-------|
| 1 | `out.eigenvalue_m1` | background plasma **+** EPs (thermal gradients retained) |
| 2 | `out.eigenvalue_m2` | EP drive only |
| 4 | `out.eigenvalue_m4` | EP-only with TAE/EPM filtered (typically leaves ITG/TEM) |

Comparing the three shows which part of the spectrum is EP-driven (AE/EPM) vs thermal
(ITG/TEM). With `WIDTH_IN_FLAG=true` the fixed `WIDTH_IN` is used; with
`WIDTH_IN_FLAG=false` a `TGLFEP_ky_widthscan` (EP-only) first picks the
max-growth-rate width (and marks `kymark`), then the spectrum is computed at that width.

Because it yields spectra rather than an `SFmin`/critical gradient, this mode is **not**
routed through `ActorTJLFEP` (which would error); call TJLFEP directly. A single radius
runs in a few minutes on a CPU login node; a full radial scan, a GPU run, or any long
job should go to Slurm (NERSC login nodes are shared and kill heavy/long processes).

A ready example input is `examples/DIIID_202017C42_500ms_v3.1/input_spectrum.TGLFEP`
(see also the manual `test/smoke_spectrum.jl`).

### Quick login-node run (one radius, CPU)

```bash
module load julia/1.11.7
export JULIA_DEPOT_PATH=$PSCRATCH/.julia
cd TJLFEP
julia --project=. -t 8 --startup-file=no
```

```julia
using TJLFEP   # loaded standalone, TJLFEP stays light (no IMAS/FUSE), CPU-only here

CASE   = "examples/DIIID_202017C42_500ms_v3.1"
gacode = joinpath(CASE, "input.gacode")
tglfep = joinpath(CASE, "input_spectrum.TGLFEP")   # PROCESS_IN=3

# run ONE radius (scan_index=1); writes out.eigenvalue_m{1,2,4} into out_dir
r = run_gacode_scan_task(gacode, tglfep, 1; out_dir="spectrum_out", printout=true)

r.spectra[1]   # (ky, gamma, freq) for mode_in=1; [2]=EP-only, [4]=ITG/TEM
```

`gamma`/`freq` are `nky × nmodes` matrices (row `i` = `ky[i]`). Useful threads scale up
to ~`nky=30` (the `ky` loop is threaded); keep it modest on a login node. For a few
radii, loop `scan_index` over `1:SCAN_N`.

### Slurm (full scan, or GPU)

Use a job when running many radii, on GPU, or unattended:

```bash
#!/bin/bash -l
#SBATCH -N 1 -n 1 -c 128 -C cpu -q debug -t 00:30:00 -A m3739 -J spectrum
module load julia/1.11.7
export JULIA_DEPOT_PATH=$PSCRATCH/.julia
cd TJLFEP
julia --project=. -t 32 --startup-file=no -e '
  using TJLFEP
  CASE="examples/DIIID_202017C42_500ms_v3.1"
  for i in 1:SCAN_N    # set to your input’s SCAN_N
      run_gacode_scan_task(joinpath(CASE,"input.gacode"), joinpath(CASE,"input_spectrum.TGLFEP"), i;
                           out_dir="spectrum_out", printout=true)
  end'
```

For GPU, use `-C gpu`, load `cudatoolkit`, and pass `use_gpu=true`. At `NBASIS=32` the
per-`ky` eigenmatrix is `(NS−NS0+1)·15·NBASIS = 1440²` (complex), so the GPU eigensolver
helps each solve, but the `ky` solves *serialize* on one GPU (no cross-`ky` batching),
so the real GPU payoff is across many radii via the same `inner=:threads`/`:mps_team`
multi-GPU layout used by the `PROCESS_IN=5` scans.

### Validation

`test/runtests_regression_spectrum.jl` compares the Julia spectra to a Fortran golden
(`test/fixtures/spectrum/out.eigenvalue_m{1,2,4}_r040`); agreement is ~3e-5 relative.
See `test/fixtures/spectrum/README.md` for how the golden was generated. Note the public
Fortran `TGLFEP_driver` has a bug on the `PROCESS_IN=3` + EXPRO path (`q_scale`
uninitialized → zeroed `q` → `DSYEV` failure), so a one-line-patched reference binary was
used.

## Citation

If this software contributes to an academic publication, please cite it as follows:

> T.F. Neiser, D. Sun, B. Agnew, T. Slendebroek, O. Meneghini, B.C. Lyons, A. Ghiozzi, J. McClenaghan, G. Staebler and J. Candy, _TJLF: The quasi-linear model of gyrokinetic transport TGLF translated to Julia_, APS Meeting Abstracts (2024)
