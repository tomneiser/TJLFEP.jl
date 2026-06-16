"""
    mainsub(inputsEP, inputsPR, printout=true; solver=:grid, kwargs...)

Single-radius TGLF-EP driver. `solver` selects the critical-factor engine:

  - `:grid` (default) — the Fortran-equivalent `kwscale_scan` `(kyhat × width ×
    factor)` grid sweep (full `IFLUX=true` per combo). The trusted reference path.
  - `:ad` — the fast autodiff route: [`critical_factor_optimize`](@ref) finds the
    critical EP scale factor by AD-Newton onset solves (cheap `IFLUX=false`
    eigenvalue passes) with implicit-function-theorem `(kyhat,width)` descent,
    then confirms the all-filter onset with [`marginal_factor_faithful`](@ref).
    Cheapest, but the descent can land in a shallower local basin on the bumpy
    `f★(ky,w)` surface (so it can disagree with the grid at some radii).
  - `:robust_ad` — the robust autodiff route: [`critical_factor_robust`](@ref)
    takes the global minimum of the all-filter onset over a `(kyhat,width)` grid
    (plus `refine_rounds` of `(ky,w)` window narrowing that mirror the Fortran
    refinement), so it reproduces the grid `sfmin` to within the continuous-vs-
    discrete-factor difference. `refine_rounds` is the accuracy/speed knob: `0` is
    the cheapest coarse-grid min, larger values better resolve radii whose binding
    `(ky,w)` lies between coarse nodes (e.g. the plasma edge) at proportionally
    higher cost.
  - `:truth` — the physical-truth route: [`critical_factor_truth`](@ref) extends the
    `width` search **below** `WIDTH_MIN` (log-spaced down to ~0.05) to capture the
    genuine narrow-width EP-driven AEs that the Fortran-faithful `w≥1` box excludes,
    then converges the value in `nbasis` at the located optimum. NOT Fortran-faithful:
    at near-marginal radii it can return a `sfmin` up to ~25× lower (more unstable)
    than `:grid`/`:robust_ad`. Use it for the most-unstable physical threshold rather
    than Fortran equivalence.

All AD solvers set `FACTOR_IN`/`KYMARK`/`WIDTH_IN` like the grid path and return
the same `((growthrate, inputsEP, inputsPR, marginal_ql), (scalefactor_buffer,
wavebuffer_all))` shape, with placeholder `growthrate`/`marginal_ql` (the AD paths
do not assemble the full QL buffers).
"""
function mainsub(inputsEP::Options, inputsPR::profile, printout::Bool = true; use_gpu::Bool = false,
                 inner::Symbol = :threads, team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 ql_flux_scan::Bool = false, solver::Symbol = :grid, refine_rounds::Int = 1)
    solver in (:grid, :ad, :robust_ad, :truth) ||
        error("mainsub: solver must be :grid, :ad, :robust_ad, or :truth (got $solver)")
    x = inputsEP.PROCESS_IN
    if (x == 1)
        msg = "No"
        return msg
    elseif (x == 2)
        msg = "No"
        return msg
    elseif (x == 3)
        msg = "No"
        return msg
    elseif (x == 4)
        msg = "No"
        return msg
    elseif (x == 5)
        inputsEP.WIDTH_IN_FLAG = false
        inputsEP.MODE_IN = 2
        inputsEP.KY_MODEL = 3
        dbgmsg("mainsub ir=", inputsEP.IR, " suffix=", inputsEP.SUFFIX,
            " SCAN_N=", inputsEP.SCAN_N, " N_BASIS=", inputsEP.N_BASIS)

        if solver == :ad
            return _mainsub_ad(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team)
        elseif solver == :robust_ad
            return _mainsub_robust_ad(inputsEP, inputsPR, printout; use_gpu=use_gpu,
                                      inner=inner, team=team, refine_rounds=refine_rounds)
        elseif solver == :truth
            return _mainsub_truth(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team)
        end

        growthrate, inputsEP, inputsPR, marginal_ql, scalefactor_buffer, wavebuffer_all =
            kwscale_scan(inputsEP, inputsPR, printout; use_gpu=use_gpu, inner=inner, team=team,
                         ql_flux_scan=ql_flux_scan)
        return (growthrate, inputsEP, inputsPR, marginal_ql), (scalefactor_buffer, wavebuffer_all)
    elseif (x == 6)
        msg = "No"
        return msg 
    end
end

"""
    _mainsub_ad(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads, team=nothing) -> ((growthrate, ep, pr, marginal_ql), (sf_buf, wf_buf))

AD-solver branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_optimize`](@ref) (faithful-confirmed) to obtain the critical EP
scale factor and its marking `(kymark, width)`, writes them onto `inputsEP`
(`FACTOR_IN`, `KYMARK`, `WIDTH_IN`) exactly like the grid path, and returns the
`mainsub` tuple shape so the radial drivers/output writers are unchanged. The AD
path does not assemble the full QL/growthrate buffers, so `growthrate` is a NaN
placeholder and `marginal_ql` is `nothing`; the scalefactor buffer carries a short
AD summary when `printout`.

`inner`/`team` select the within-radius parallelism of the AD path's independent-
eval regions (seed grid, eigenvalue hull scan, faithful sweep): `:mps_team`
distributes them across the MPS team (the production GPU route, separate CUDA
contexts overlapping via Hyper-Q), else they run in-process threaded.
"""
function _mainsub_ad(inputsEP::Options, inputsPR::profile, printout::Bool; use_gpu::Bool = false,
                     inner::Symbol = :threads,
                     team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    res = critical_factor_optimize(inputsEP, inputsPR; use_gpu=use_gpu, faithful_confirm=true,
                                   inner=inner, team=team)

    # Prefer the all-filter (faithful) onset when a mode actually binds; otherwise
    # fall back to the AE-band optimum from the descent.
    sfmin = res.sfmin
    if res.faithful !== nothing && res.faithful.binding != :none &&
       isfinite(res.faithful.factor_faithful)
        sfmin = res.faithful.factor_faithful
    end
    kymark = res.kyhat
    wmark  = res.width

    inputsEP.FACTOR_IN = isfinite(sfmin) ? sfmin : inputsEP.FACTOR_IN
    inputsEP.KYMARK    = kymark
    if isfinite(wmark)
        inputsEP.WIDTH_IN = wmark
    end

    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing

    sf_buf = String[]
    if printout
        bind = res.faithful === nothing ? :none : res.faithful.binding
        push!(sf_buf, "# TJLFEP AD solver (critical_factor_optimize + marginal_factor_faithful)")
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(bind)")
        push!(sf_buf, "iters = $(res.iters)  evals = $(res.evals)  converged = $(res.converged)")
    end

    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, nothing)
end

"""
    _mainsub_robust_ad(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads,
                       team=nothing, refine_rounds=1) -> ((growthrate, ep, pr, marginal_ql), (sf_buf, wf_buf))

Robust AD-solver branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_robust`](@ref) — the global minimum of the all-filter
marginal factor over a `(kyhat,width)` grid with `refine_rounds` of window
narrowing — to obtain the critical EP scale factor and its marking
`(kymark, width)`, writes them onto `inputsEP` (`FACTOR_IN`, `KYMARK`, `WIDTH_IN`)
exactly like the grid path, and returns the `mainsub` tuple shape. `refine_rounds`
is the user-facing accuracy/speed knob (see [`critical_factor_robust`](@ref)).

If the reduction reports no genuine interior onset (`status ∈ (:no_onset, :cap)`),
`FACTOR_IN` is left at its incoming value (the grid path remains the fallback for
that radius) and the status is recorded in the scalefactor buffer.

`inner`/`team` select the within-radius parallelism (the `(ky,w)` grid points are
distributed across the MPS team or in-process threads); each point's inner
faithful factor sweep runs serially to avoid nested oversubscription.
"""
function _mainsub_robust_ad(inputsEP::Options, inputsPR::profile, printout::Bool;
                            use_gpu::Bool = false, inner::Symbol = :threads,
                            team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                            refine_rounds::Int = 1)
    res = critical_factor_robust(inputsEP, inputsPR; use_gpu=use_gpu,
                                 inner=inner, team=team, refine_rounds=refine_rounds)

    sfmin  = res.sfmin
    kymark = res.kyhat
    wmark  = res.width
    genuine = res.status === :ok && isfinite(sfmin)

    # Only adopt the AD onset when a genuine interior onset was found; otherwise leave
    # FACTOR_IN untouched so the radius can fall back to the grid path / its prior value.
    if genuine
        inputsEP.FACTOR_IN = sfmin
        inputsEP.KYMARK    = kymark
        if isfinite(wmark)
            inputsEP.WIDTH_IN = wmark
        end
    end

    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing

    sf_buf = String[]
    if printout
        push!(sf_buf, "# TJLFEP AD solver (critical_factor_robust, refine_rounds=$(refine_rounds))")
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(res.binding)")
        push!(sf_buf, "status = $(res.status)")
        push!(sf_buf, "refine_done = $(res.refine_done)  (budget=$(refine_rounds), adaptive)")
        push!(sf_buf, "n_feasible_coarse = $(res.n_feasible_coarse) / $(res.npts_coarse)")
        push!(sf_buf, "npts = $(res.npts)  evals_full = $(res.total_evals_full)  evals_eig = $(res.total_evals_eig)")
    end

    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, nothing)
end

"""
    _mainsub_truth(inputsEP, inputsPR, printout; use_gpu=false, inner=:threads,
                   team=nothing) -> ((growthrate, ep, pr, marginal_ql), (sf_buf, wf_buf))

Physical-truth AD-solver branch of [`mainsub`](@ref) (PROCESS_IN==5). Runs
[`critical_factor_truth`](@ref) — locates the global `(kyhat,width)` minimum of the
all-filter marginal factor over a width box **extended below `WIDTH_MIN`** (log-spaced
down to ~0.05) and converges the value in `nbasis` at that optimum — then writes the
critical EP scale factor and its marking `(kymark, width)` onto `inputsEP`
(`FACTOR_IN`, `KYMARK`, `WIDTH_IN`) like the grid path. The reported `sfmin` is the
nbasis-converged estimate; `sfmin_work` (the value at the working basis) is recorded too.

NOT Fortran-faithful: the narrow-width modes it captures are excluded by the canonical
`w≥1` box, so at near-marginal radii `sfmin` can be much lower (more unstable). If no
genuine onset is found (`status ∈ (:no_onset,:cap)`), `FACTOR_IN` is left untouched.
`inner`/`team` select within-radius parallelism over the seed grid / nbasis evals.
"""
function _mainsub_truth(inputsEP::Options, inputsPR::profile, printout::Bool;
                        use_gpu::Bool = false, inner::Symbol = :threads,
                        team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    res = critical_factor_truth(inputsEP, inputsPR; use_gpu=use_gpu, inner=inner, team=team)

    sfmin  = res.sfmin
    kymark = res.kyhat
    wmark  = res.width
    genuine = res.status === :ok && isfinite(sfmin)

    if genuine
        inputsEP.FACTOR_IN = sfmin
        inputsEP.KYMARK    = kymark
        if isfinite(wmark)
            inputsEP.WIDTH_IN = wmark
        end
    end

    growthrate  = fill(NaN, (5, 10, 10, inputsEP.NMODES))
    marginal_ql = nothing

    sf_buf = String[]
    if printout
        nbs, vals = res.nb_table
        push!(sf_buf, "# TJLFEP AD solver (critical_factor_truth, extended-width + nbasis-converged)")
        push!(sf_buf, "ir = $(inputsEP.IR)")
        push!(sf_buf, "sfmin = $(sfmin)   # nbasis-converged")
        push!(sf_buf, "sfmin_work = $(res.sfmin_work)   # at working nbasis")
        push!(sf_buf, "kymark = $(kymark)")
        push!(sf_buf, "width = $(wmark)")
        push!(sf_buf, "binding = $(res.binding)")
        push!(sf_buf, "status = $(res.status)")
        push!(sf_buf, "nbasis = $(collect(nbs)) -> $(collect(vals))  (converged=$(res.nb_converged), limit=$(res.nb_limit))")
        push!(sf_buf, "feasible_frac = $(res.feasible_frac)")
        push!(sf_buf, "n_confirm = $(res.n_confirm)  evals_full = $(res.total_evals_full)  evals_eig = $(res.total_evals_eig)")
    end

    return (growthrate, inputsEP, inputsPR, marginal_ql), (sf_buf, nothing)
end