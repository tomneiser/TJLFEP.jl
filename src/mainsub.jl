"""
    mainsub(inputsEP, inputsPR, printout=true; solver=:grid, kwargs...)

Single-radius TGLF-EP driver. `solver` selects the critical-factor engine:

  - `:grid` (default) — the Fortran-equivalent `kwscale_scan` `(kyhat × width ×
    factor)` grid sweep (full `IFLUX=true` per combo). The trusted reference path.
  - `:ad` — the autodiff route: [`critical_factor_optimize`](@ref) finds the
    critical EP scale factor by AD-Newton onset solves (cheap `IFLUX=false`
    eigenvalue passes) with implicit-function-theorem `(kyhat,width)` descent,
    then confirms the all-filter onset with [`marginal_factor_faithful`](@ref).
    Sets `FACTOR_IN`/`KYMARK`/`WIDTH_IN` like the grid path. Returns the same
    `((growthrate, inputsEP, inputsPR, marginal_ql), (scalefactor_buffer,
    wavebuffer_all))` shape, with placeholder `growthrate`/`marginal_ql` (the AD
    path does not assemble the full QL buffers).
"""
function mainsub(inputsEP::Options, inputsPR::profile, printout::Bool = true; use_gpu::Bool = false,
                 inner::Symbol = :threads, team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                 ql_flux_scan::Bool = false, solver::Symbol = :grid)
    solver in (:grid, :ad) || error("mainsub: solver must be :grid or :ad (got $solver)")
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