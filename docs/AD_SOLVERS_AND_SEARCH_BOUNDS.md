# TGLF-EP critical-factor solvers, accuracy, and the `(kyhat, width)` search bounds

This document records the outcome of a study of the autodiff (AD) critical-factor solvers in
`TJLFEP` and — more importantly — what we learned about the **physical/numerical search bounds**
(`kyhat`, `width`, `nbasis`) while validating them on the DIII-D `n_scan=20`, `nbasis=32` case
(`examples/DIIID_202017C42_500ms_v3.1`). All GPU runs used 1×A100 with an MPS worker team of 4.

The bottom line up front:

- For the **fast production solver**, `adf1` (pinned-aware seed → `:ad` descent → faithful confirm)
  is exact-and-cheap on *clean* and *floor-pinned* radii (most of the profile). The grid-zoom
  `robust_ad` is the trustworthy reference on the *hard* (multimodal/sparse) radii.
- For the **hard near-marginal radii (e.g. IR=48, IR=95)** the *true* critical factor lies at
  **narrow width (`w ≈ 0.1`), below the default `WIDTH_MIN = 1.0`**, and is a real, EP-driven,
  **numerically converged** mode — it is **~10× more unstable** than the in-box value (IR=95:
  `sfmin ≈ 0.21` vs the `w ≥ 1` grid's `~2.64`). See §4.
- The **default search box (`width ∈ [WIDTH_MIN, WIDTH_MAX] = [1,2]`) is a *modeling/faithfulness*
  choice** that matches Fortran TGLF-EP's width floor — **not** a numerical necessity. It keeps the
  scan in a well-conditioned regime, but at near-marginal radii it **biases `sfmin` high by ~10×** by
  excluding genuine narrow-width AEs.

---

## 1. The solver family

All solvers minimise the **faithful** marginal EP scale factor `sfmin` — the factor at which the
leading AE-band growth crosses `γ*` and passes the TGLF-EP keep filters — over `(kyhat, width)`.
`kwscale_scan`/`grid` is the Fortran-equivalent reference.

| Solver | Strategy | Notes |
|--------|----------|-------|
| `:grid` | Fortran `kwscale_scan` `(kyhat × width × factor)` sweep | reference; bit-faithful to Fortran |
| `:ad` (`critical_factor_optimize`) | 1 seed → projected-gradient/IFT descent on the cheap AE-onset surface | fastest; **blind to floor-pinned basins**, single-basin fragile |
| `:robust_ad` (`critical_factor_robust`) | grid-zoom over `(kyhat,width)` with faithful evals + adaptive refinement | robust everywhere; never returns `Inf` |
| `:confirm` (`critical_factor_confirm`) | cheap eigenvalue-only `f1` grid search + early-stop few-confirm | provably exact over the grid; fewer `IFLUX=true` evals |
| `adf1` (`critical_factor_ad_f1seed`) *(core)* | pinned-aware `f1` seed grid → `:ad` descent on interior basins (+ grid-floor guard) → early-stop confirm | fixes `:ad`'s pinned-blindness; fast canonical pass |
| **`:truth`** (`critical_factor_truth`) *(core)* | **extended** log-width `(ky,w)` locate (`w` down to ~0.05) → `:ad` polish → faithful confirm + **separable nbasis convergence** | finds the true narrow-width minimum the `w≥1` box misses; **NOT Fortran-faithful** (see §5) |
| `critical_factor_triggered` *(core)* | fast `adf1` canonical pass + width-floor/trust trigger → escalate to `:truth`, keep `min` | production policy wrapper |
| `critical_factor_direct` *(experiment)* | NLopt `GN_DIRECT_L` global search on cheap AE-edge + early-stop confirm | most accurate on **dense** surfaces; **fails on sparse** ones |
| `critical_factor_ad_escalate` *(experiment)* | `adf1` default + trust gate → escalate to `:direct` or `:grid` | see §3 |

`adf1`, `critical_factor_truth`, and `critical_factor_triggered` are **promoted to core**
(`src/tjlfep_ad_extensions.jl`, exported) and `:truth` is selectable from `mainsub` /
`run_gacode_scan_task` / the `solver` toggle like `:grid`/`:ad`/`:robust_ad`. The remaining
experiment-only solvers (`critical_factor_direct`, `critical_factor_ad_escalate`) live in
`build/ad/direct_solver.jl` and depend on `NLopt` (in `Project.toml` but **not** imported by the
module, so the production package / sysimage stay NLopt-free).

---

## 2. Accuracy comparison (canonical `kyhat ≥ 0.25`, `nbasis=32`)

From the escalation validation (job 54547632), all methods faithful-confirmed, MPS team=4:

| IR | surface | `adf1` | DIRECT-40 | `robust_ad` (grid-zoom) |
|----|---------|--------|-----------|--------------------------|
| 22 | clean        | 0.16182 | 0.16015 | 0.15675 |
| 38 | floor-pinned | 0.019531 | 0.019531 | 0.019531 |
| 48 | dense off-node | 0.054455 | **0.026761** | 0.054455 |
| 95 | sparse       | 5.4907 | **Inf (no_onset)** | 3.8793 |

Reading:

- **Clean / pinned radii (22, 38):** all three agree; `adf1` is fastest (IR=38 ~45 s vs DIRECT
  ~175 s) and is exact on the floor-pinned radius that plain/guarded `:ad` *miss* (their gradient
  objective is blind to floor-pinned instabilities).
- **Dense off-node (48):** DIRECT-40 finds a genuinely lower basin (0.0268, ≈ the dense-grid value
  0.0286) that both `adf1` and `robust_ad` miss (both basin-lock at 0.0545). DIRECT's adaptive
  global sampling is the only method that escapes the wrong basin here.
- **Sparse (95):** DIRECT-40 **fails** — in 40 evals it never lands a confirmable unstable sample
  (the unstable region is ~16% of the box), returning `no_onset`. `robust_ad`'s systematic
  grid+zoom always returns a finite value.

### Is `robust_ad` better than DIRECT-40?

**Neither dominates.** It is a robustness/accuracy trade:

- **`robust_ad`** never fails (always finite), is the production default, but can miss off-node
  minima on dense surfaces (IR=48: +100% vs DIRECT).
- **DIRECT-40** is more accurate on dense surfaces (finds off-node basins) but is **fragile**: on
  sparse surfaces its space-partitioning can return `no_onset` (IR=95), and it costs ~25–50% more
  wallclock. It is also **not** a viable universal escalation target for that reason.

---

## 3. Escalation policy (`critical_factor_ad_escalate`)

`adf1` is the fast default; a cheap **trust gate** escalates only flagged radii:

- `cheap_gap = sfmin / cheap_f1(winner) > 1.5` — keep filters flipped at the descended basin
  (faithful ≫ cheap).
- `feasible_frac < 0.25` — sparse unstable seed grid (multimodal/under-bracketed surface).
- `no_onset` / `cap` — nothing trustworthy found.

Flagged → run the escalation **target** (`:direct` = DIRECT-40, or `:grid` = `robust_ad`) and keep
the lower faithful `sfmin`. Validation (job 54547632) showed two limits worth recording:

1. **DIRECT-40 fails the sparse case (IR=95 → Inf)**, so `:direct` cannot be the universal target;
   `:grid` is required for sparse radii.
2. The cheap gate **does not flag IR=48**: `adf1`'s answer there is locally self-consistent
   (`cheap_gap≈1.0`, `feasible_frac=0.81`) — a *missed basin* with no cheap signal. Detecting it
   cheaply is not possible with a coarse static grid; only adaptive global sampling (DIRECT) finds
   it. This is a genuine limitation, not a tuning issue.

**Practical default:** `adf1` + escalate-to-`:grid` on flagged (sparse/keep-divergent) radii. This
gives `:ad`-class speed on the clean/pinned majority and grid-zoom robustness on the hard radii.

---

## 4. The `(kyhat, width)` search bounds — physics and numerics

While reconciling solver disagreements on the hard radii, we found the disagreements trace to the
**search bounds**, not the optimisers. Key facts:

### 4.1 `kyhat` is physical; its grid "floor" is just sampling
`TJLF_map` unconditionally sets `KY_MODEL=3` (`tjlfep_read_inputs.jl:889`), so for **both** the grid
and AD paths `KY = KYHAT_IN · Z/√(m·T)`. The scan domain is `kyhat ∈ [0,1]`; the grid merely samples
it at `{0.25, 0.5, 0.75, 1.0}` (for `nkyhat=4`). So `kyhat=0.25` is **not** a physical floor.
However, an extended faithful mesh (job 54553260) shows the AE onset **self-limits in `kyhat`**:
it vanishes (stable) below `ky≈0.05` at IR=48 and below `ky≈0.01` at IR=95 — there is no runaway
toward `ky→0`. (An earlier DIRECT result of `sfmin≈1.38` at `ky=0.006` was a sub-`0.01` point that
disappears once the onset is tracked properly.)

### 4.2 `WIDTH_MIN=1.0` truncates more-unstable narrow-width AEs…
Extended faithful meshes (jobs 54553260, 54557433) stepping **below** `WIDTH_MIN=1` find the true
`(ky,w)` minimum at `width < 1` on the hard radii:

| IR | grid box (w≥1) `sfmin` | extended-box min `sfmin` | at (kyhat, width) |
|----|------------------------|--------------------------|-------------------|
| 48 | ~0.039–0.0545 | **0.0195** (scan floor) | (0.25, 0.6) — interior |
| 95 | ~2.64–3.88 | **≈0.21** | (0.8, 0.10–0.125) — interior bowl |

An EP-drive check (`γ_AE(factor)` from `factor→0` to nominal, job 54557433) confirms these
narrow-width modes are **genuinely EP-driven** (`γ_AE ≤ γ*` at `factor→0`, growing with EP drive),
not background micro-instabilities. So the grid box *does* exclude real, more-unstable modes, and the
production `scan20` `sfmin` is **biased high** at near-marginal radii.

### 4.3 The narrow-width minimum is real and numerically converged
A finer sweep in **both** width and `nbasis` (job 54569487, `ad/extbox5_experiment.jl`) shows the
narrow-width minimum is a genuine, finite, converged value — **not** the `width→0` runaway that the
earlier coarse sweeps (jobs 54561168, 54563549) appeared to suggest.

**(a) Width is a bowl, not a runaway.** `sfmin(width)` at IR=95, `nb=32`, turns around below
`w ≈ 0.1` (it *rises* again at `w = 0.05`), giving an **interior minimum**:

| `w`           | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.2 | 0.5 |
|---------------|------|-------|------|-------|------|-----|-----|
| `sfmin` (ky=0.8) | 0.531 | 0.280 | 0.227 | **0.212** | 0.234 | 0.886 | 6.76 |
| `sfmin` (ky=0.5) | 0.394 | 0.264 | **0.257** | 0.263 | 0.292 | 0.687 | 4.47 |

The earlier runs only used width floors of 1.0/0.5/0.2/0.1, so they always sat on the *descending
outer wall* and pinned at the floor; sampling to `w = 0.05` exposes the floor of the bowl.

**(b) `nbasis` converges geometrically.** At the optimum `(ky=0.8, w=0.1)` the per-step change
**halves every step** — a converging geometric series, reaching a stable value by `nb ≈ 48–56`:

| nb | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
|----|----|----|----|----|----|----|----|
| `sfmin` | 0.477 | 0.290 | 0.248 | 0.227 | 0.216 | 0.2115 | **0.2114** |
| Δ | — | −0.187 | −0.043 | −0.021 | −0.011 | −0.0047 | **−0.0001** |

The convergence limit is reached **before** the rank ceiling, so the `nbasis ≥ 64` singularity
(below) is irrelevant to this point — we never need `nb = 64`.

**On the `nbasis ≥ 64` singularity.** It is real but a *separate* issue. `inv(ave.p0)`/`inv(ave.bp)`
in `get_matrix` (`TJLF/src/tjlf_matrix.jl:46–60`) is the inverse of the Hermite **overlap matrix**;
at `N ≳ 64` the basis becomes genuinely rank-deficient (singular at *every* width, incl. `w=1.5`),
so a pseudo-inverse would only null the dependent directions — it adds no information. `nb = 64` is
simply past the usable rank of this basis, and there is no `nb→∞` limit to chase. It does **not**
prevent convergence at the narrow-width optimum, which is already achieved at `nb ≈ 48`.

### 4.4 Consequence
IR=95 has a real, finite, numerically converged minimum at **`sfmin ≈ 0.21`, `(ky≈0.8, w≈0.1)`** — a
genuine EP-driven narrow-width AE that is **~10× more unstable** than the `w ≥ 1` grid value
(`~2.64`). The DIRECT-40 `1.38` (at `w≈1.1`) was just a point on the descending wall, not the true
minimum. Therefore `WIDTH_MIN = 1.0` is a **modeling/faithfulness** choice (match Fortran TGLF-EP),
**not** a numerical necessity: it excludes a converged, more-unstable mode and biases the production
`scan20` `sfmin` **high by ~10×** at near-marginal radii. Whether to admit `w < 1` modes is a
physics-modeling decision (how localized a ballooning envelope is considered physical), not a
solver-accuracy limitation.

### 4.5 The validated physical-truth protocol (`:truth`)

Once §4 established that a real narrow-width minimum exists, the open question was how to find it
**robustly** (don't miss it) and then **fast**. `critical_factor_truth` implements a three-stage
protocol, validated against the brute extended-box sweeps above:

1. **Locate** — coarse seed grid over an **extended** log-spaced width range (`w` down to ~0.05,
   with `WIDTH_MIN`/`WIDTH_MAX` explicitly anchored into the mesh so the canonical `w∈[1,2]` box is
   always covered), ranked by the cheap AE-edge objective, then faithful-confirmed at the top
   candidates at `nb=32`.
2. **Polish** — `critical_factor_optimize` (`:ad`) descent from the best confirmed seed.
3. **Converge `nbasis`** — a **separable** 1-D sweep at the *fixed* located `(ky*, w*)` over
   `nb ∈ {32,40,48}` with geometric/Aitken extrapolation. (Measuring at fixed `(ky*,w*)` — not
   re-optimizing per `nb` — is what keeps the sequence monotone; an earlier `repolish_top` corrupted
   it by relocating the optimum at higher `nb`.) `nb ≥ 64` is skipped: it is past the usable rank of
   the Hermite basis (§4.3) and the optimum is already converged by `nb ≈ 48`.

`critical_factor_triggered` is the **production policy**: run the fast canonical `adf1` pass on the
`w∈[1,2]` box first, and **only escalate** to the full `:truth` protocol when the canonical optimum
pins at `WIDTH_MIN` or the trust diagnostics (`feasible_frac`, `cheap_gap`) flag it, then report
`min(canonical, truth)`. This pays the extended-box cost only at the near-marginal radii that need
it, leaving clean/pinned radii on the fast path.

Two reported quantities make the modeling choice explicit:
- **`critical_factor_truth`** — the production value `min(robust_ad, narrow-width truth)`: the
  extended-`w` narrow-width minimum, floored by `robust_ad` (the refined `w≥1` grid-zoom, kwarg
  `robust_floor=true`) so it never reports above the best refined-faithful threshold. The floor wins
  in the deep core where the narrow-width sequence is still climbing; truth wins outward. Pass
  `robust_floor=false` for the raw, unfloored narrow-width minimum.
- **`critical_factor_triggered`** — the fast production wrapper: canonical `adf1` on `w≥1`, escalating
  to the (floored) `critical_factor_truth` only at width-floor-pinned / trust-flagged radii.

### 4.6 Validated full-profile result (DIII-D SCAN_N=20, nbasis=32)

The full 20-radius `:truth` profile (job 54580579, 5 nodes × 4 GPUs, `MPS_TEAM=8`, JIT) reproduces
the `extbox5` ground truth — most notably **IR=95: `sfmin=0.2147` at `(ky=0.81, w=0.113)`** — and the
production `min(robust_ad, truth)` is **~2× below the grid on the median radius, up to ~12× at IR=95**:

![grid vs robust_ad vs physical-truth sfmin profile](plots/sfmin_grid_vs_truth_nb32.png)

Gray dotted = Fortran-faithful `kwscale_scan` on `w∈[1,2]` (reference); blue = `robust_ad`, the
refined `w≥1` grid-zoom of the faithful onset (the production **floor**); orange =
`critical_factor_truth` (raw, extended `w`); green dashed = the production `min(robust_ad, truth)`
(`:truth`/`:triggered`). The min tracks truth at the outer radii and falls back to the `robust_ad`
floor at the 6 deep-core radii where the raw narrow-width `:truth` sequence is still climbing and
overshoots. (Job 54580579 predates the `robust_floor` baking, so here the floor is applied to that
profile post-hoc from the `robust_ad` r1 run; the rebuilt sysimage applies it inline.)

Per-radius optimum found by `critical_factor_truth` (`ky*`, `width*` = the located `(ky,w)`; `sfmin`
at the nbasis-converged limit over `{32,40,48}`; `binding` = the keep filter that sets the threshold):

| IR | ky* | width* | binding | sfmin (truth) | nb-converged | grid (w≥1) | grid/truth |
|----|-----|--------|---------|---------------|--------------|------------|------------|
| 2 | 0.43 | 0.881 | i_pinch | 1.3995 | false | 0.9374 | 0.7× |
| 7 | 0.81 | 0.881 | i_pinch | 0.9229 | false | 0.6249 | 0.7× |
| 12 | 0.43 | 0.585 | i_pinch | 0.2633 | false | 0.3125 | 1.2× |
| 17 | 0.43 | 0.585 | ae_band_growth | 0.2050 | false | 0.2344 | 1.1× |
| 22 | 0.24 | 0.585 | ae_band_growth | 0.1346 | false | 0.1758 | 1.3× |
| 28 | 0.43 | 0.388 | ae_band_growth | 0.2397 | true | 0.1562 | 0.7× |
| 33 | 0.31 | 0.438 | ae_band_growth | 0.0570 | false | 0.1172 | 2.1× |
| 38 | 0.24 | 0.881 | ae_band_growth | 0.0195 | false | 0.0195 | 1.0× |
| 43 | 0.24 | 0.585 | ae_band_growth | 0.0195 | false | 0.0195 | 1.0× |
| 48 | 0.43 | 0.388 | ae_band_growth | 0.0195 | false | 0.0391 | 2.0× |
| 54 | 0.43 | 0.388 | ae_band_growth | 0.0591 | false | 0.1172 | 2.0× |
| 59 | 0.43 | 0.258 | ae_band_growth | 0.0575 | true | 0.1758 | 3.1× |
| 64 | 0.43 | 0.258 | ae_band_growth | 0.0361 | false | 0.2344 | 6.5× |
| 69 | 0.62 | 0.258 | ae_band_growth | 0.1368 | false | 0.4687 | 3.4× |
| 74 | 0.81 | 0.258 | ae_band_growth | 0.2412 | false | 0.5273 | 2.2× |
| 80 | 0.81 | 0.258 | ae_band_growth | 0.2993 | false | 1.2498 | 4.2× |
| 85 | 1.00 | 0.171 | ae_band_growth | 0.2313 | false | 1.2497 | 5.4× |
| 90 | 1.00 | 0.171 | ae_band_growth | 0.3223 | false | 1.8749 | 5.8× |
| 95 | 0.81 | 0.113 | ae_band_growth | 0.2147 | false | 2.6367 | 12.3× |
| 101 | 0.43 | 0.258 | ae_band_growth | 0.1763 | false | 0.2344 | 1.3× |

Reading:
- **All optima are at `width < 1`** (0.11–0.88), i.e. the threshold-setting AE is the narrow-width mode
  the canonical `w≥1` box excludes — hence the systematic `grid/truth > 1` across the outer half.
- **`nb-converged=false` at most radii** means the `sfmin({32,40,48})` sequence had not flattened to
  tolerance, so the conservative finest-`nb` (extrapolated) limit is reported. At the **deep-core radii
  IR=2/7/28** this overshoots (ratio < 1, truth *higher*); the `robust_floor` inside
  `critical_factor_truth` takes `min(robust_ad, truth)` there, so those radii report the refined
  `robust_ad` value instead (e.g. IR=2 → 0.890, IR=7 → 0.607).
- **`binding`** is `ae_band_growth` (the EP-driven AE growth crossing) everywhere except the innermost
  IR=2/7/12, where `i_pinch` (the ion-pinch keep filter) sets the threshold.

---

## 5. Production recommendation

- Keep **`WIDTH_MIN=1.0`, `WIDTH_MAX=2.0`, `nbasis=32`** as the default operating box **to stay
  faithful to Fortran TGLF-EP** (which floors width at 1.0) and in the well-conditioned regime.
  This is a modeling choice, not a numerical one.
- Fast solver: **`adf1`** (or `:ad`) for clean/pinned radii; **`robust_ad`** as the trustworthy
  reference and as the escalation target for flagged hard radii. Do **not** use DIRECT as a universal
  fallback (sparse-surface failure).
- **Be explicit that the `w ≥ 1` box reports a *conservative-by-construction, Fortran-faithful*
  critical factor.** At near-marginal radii (steep edge of the `sfmin` profile, e.g. IR≈48, 95) a
  real, converged, narrow-width EP-driven AE exists at `w ≈ 0.1` with `sfmin` **~10× lower** (IR=95:
  ≈0.21 vs ≈2.64). If the application needs the true most-unstable EP threshold rather than
  Fortran-equivalence, **lower `WIDTH_MIN` toward ~0.1** (and use `nb ≥ 48` at those radii) — the
  result is numerically trustworthy there.
- **To get the true threshold without hand-tuning `WIDTH_MIN`, run `solver=:truth`** (the
  `critical_factor_triggered` policy of §4.5): it stays on the fast canonical path everywhere except
  the flagged near-marginal radii, where it escalates to the extended-width + separable-`nbasis`
  protocol and reports `min(Fortran-faithful, physical-truth)`. On GPU/MPS use `INNER=mps_team
  MPS_TEAM=8` (same team as `:grid`; the ~66-point extended seed grid is embarrassingly parallel and
  underfills the A100 per-eigensolve at `nb≤48`).

---

## 6. Reproduction

Experiment harnesses (run from `build/`, premium GPU, MPS team=4):

| Question | Script | Batch |
|----------|--------|-------|
| `adf1` vs lean DIRECT head-to-head | `ad/headtohead_experiment.jl` | `ad/batch_headtohead_experiment.sh` |
| escalation: `adf1` + DIRECT-40 + `robust_ad` | `ad/escalation_experiment.jl` | `ad/batch_escalation_experiment.sh` |
| extended `(ky,w)` box (faithful) | `ad/extended_box_experiment.jl` | `ad/batch_extended_box.sh` |
| narrow width + EP-drive check | `ad/extbox2_experiment.jl` | `ad/batch_extbox2.sh` |
| IR=95 corner + `nbasis` convergence | `ad/extbox3_experiment.jl` | `ad/batch_extbox3.sh` |
| high-`nbasis` {32,48,64,96} convergence | `ad/extbox4_experiment.jl` | `ad/batch_extbox4.sh` |
| IR=95 width-bowl + fine `nbasis` {8..56} (ground truth) | `ad/extbox5_experiment.jl` | `ad/batch_extbox5.sh` |
| `:truth` protocol validation (locate + separable `nbasis`) | `ad/truth_experiment.jl` | `ad/batch_truth_experiment.sh` |

Solver definitions for the experiment-only methods: `build/ad/direct_solver.jl`. The core `:truth`
path (`critical_factor_truth` / `critical_factor_triggered` / `adf1`) lives in
`src/tjlfep_ad_extensions.jl`.

Bounds are set via env (`KY_LO`, mesh arrays in the scripts); `INNER=mps_team MPS_TEAM=4` selects the
GPU MPS path for the experiments. The **production `:truth` profile** uses `MPS_TEAM=8`
(`build/timing/batch_scan20_truth.sh`) to match the `:grid` team size.
