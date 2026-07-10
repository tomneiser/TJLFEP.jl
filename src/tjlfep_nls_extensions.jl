# Derivative-free critical-factor solvers borrowed from the FluxMatcher's nonlinear-optimizer
# playbook. These replace the AD gradients used by `critical_factor_optimize` at BOTH layers:
#
#   inner  (factor root)  : `marginal_factor_df` — bracketing ITP/Brent instead of AD-Newton
#                           (defined in tjlfep_ad_extensions.jl, used here for f★(ky,w)).
#   outer  ((ky,w) search): `critical_factor_dfsane` — SimpleDFSane on the finite-difference
#                           stationarity of f★(ky,w), globalized by a cheap AE-onset multistart;
#                           `critical_factor_nlopt`  — NLopt derivative-free global (GN_DIRECT_L /
#                           GN_CRS2_LM) + optional local polish (LN_BOBYQA / LN_COBYLA).
#
# Both search the width box EXTENDED below WIDTH_MIN (down to `w_lo`≈0.05, like :ad :locate /
# :robust_ad) to capture the narrow EP-driven AE modes the `w≥1` box misses, then faithful-confirm
# (IFLUX=true) the candidate optima with an early-stop bound. They return the SAME contract as
# `critical_factor_robust`/`critical_factor_direct` so the `_mainsub_*` branches consume them
# uniformly: `(; sfmin, kyhat, width, binding, status, converged, faithful, n_confirm, n_samples,
# total_evals_full, total_evals_eig)` with `status ∈ (:ok, :no_onset, :cap)`.

# ── shared: cheap AE-onset (ky,w) ranking (IFLUX=false), narrow-width extended ──
# Log-spaced widths (down to w_lo) so the multistart sees the narrow band, linear kyhat.
# Returns feasible seeds `(ky, w, f1, pinned)` sorted ascending by cheap onset f1, plus eig count.
function _cheap_onset_seeds(ep0::Options{Float64}, prof::profile{Float64}, gth::Float64;
                            slo::Float64, shi::Float64, ky_lo::Float64, w_lo::Float64, w_hi::Float64,
                            nseed_ky::Int, nseed_w::Int, n_eig::Int,
                            inner::Symbol, team, use_gpu::Bool)
    kyhats = nseed_ky <= 1 ? [0.5 * (ky_lo + 1.0)] :
             [ky_lo + (1.0 - ky_lo) / (nseed_ky - 1) * (i - 1) for i in 1:nseed_ky]
    # Widths span two regimes: LOG-spaced from w_lo (resolves the narrow EP-driven AE band that
    # the `w≥1` box misses) AND LINEAR across [WIDTH_MIN, WIDTH_MAX] (the canonical box where
    # kwscale_scan finds the dense core modes — log spacing alone under-samples it). Union both so
    # the multistart brackets both the narrow-edge and wide-core basins.
    wmn = Float64(ep0.WIDTH_MIN); wmx = Float64(ep0.WIDTH_MAX)
    logw = nseed_w <= 1 ? [sqrt(w_lo * w_hi)] : collect(exp.(range(log(w_lo), log(w_hi), length = nseed_w)))
    nlin = max(2, cld(nseed_w, 2))
    linw = [wmn + (wmx - wmn) * (i - 1) / (nlin - 1) for i in 1:nlin]
    widths = sort(unique(vcat(logw, linw, wmn, wmx)))
    pts = [(ky, w) for ky in kyhats for w in widths]
    cheap = _ad_pmap(idx -> begin
            ky, w = pts[idx]; ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            win = _ae_unstable_window(ep, prof, gth; scan_lo = slo, scan_hi = shi,
                                      n_eig = n_eig, threaded = false, use_gpu = use_gpu)
            (; ky = ky, w = w, f = (win.unstable ? win.f1 : Inf), pinned = win.pinned_lo, evals = win.evals)
        end, length(pts); inner = inner, team = team)
    eig = 0
    seeds = NamedTuple[]
    for c in cheap
        eig += c.evals
        isfinite(c.f) && push!(seeds, (; ky = c.ky, w = c.w, f = c.f, pinned = c.pinned))
    end
    sort!(seeds, by = s -> s.f)
    return (; seeds = seeds, npts = length(pts), eig = eig)
end

# ── shared: faithful (IFLUX=true) early-stop confirm of candidate (ky,w) optima ──
# `cands` :: Vector of (ky, w, cheap_f), sorted ascending by cheap_f. The cheap AE-band onset
# lower-bounds the faithful keep onset, so once a candidate's cheap_f ≥ the running faithful best
# it (and all later) cannot win — an exact prune that bounds the confirm count. Returns the winning
# faithful onset, marking, binding, the `marginal_factor_faithful` result, and eval tallies.
function _confirm_candidates(ep0::Options{Float64}, prof::profile{Float64}, gth::Float64,
                             cands::Vector{Tuple{Float64,Float64,Float64}};
                             slo::Float64, shi::Float64, inner::Symbol, team, use_gpu::Bool)
    best_f = Inf; best_ky = NaN; best_w = NaN; best_bind = :none; best_faithful = nothing
    n_confirm = 0; total_full = 0; total_eig = 0
    for c in cands
        isfinite(c[3]) || continue
        stop_bound = isfinite(best_f) ? best_f : shi
        c[3] >= stop_bound && break
        r = marginal_factor_faithful(ep0, prof; kyhat = c[1], width = c[2], gamma_thresh = gth,
                                     scan_lo = slo, scan_hi = shi, threaded = true,
                                     inner = inner, team = team, use_gpu = use_gpu)
        n_confirm += 1; total_full += r.evals_full; total_eig += r.evals_eig
        if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
            best_f = r.factor_faithful; best_ky = c[1]; best_w = c[2]
            best_bind = r.binding; best_faithful = r
        end
    end
    return (; sfmin = best_f, kyhat = best_ky, width = best_w, binding = best_bind,
            faithful = best_faithful, n_confirm = n_confirm,
            total_evals_full = total_full, total_evals_eig = total_eig)
end

_nls_status(best_f, best_bind, shi) =
    (best_bind === :none || !isfinite(best_f)) ? :no_onset : (best_f >= 0.999 * shi ? :cap : :ok)

"""
    critical_factor_dfsane(inputsEP, inputsPR; kwargs...) -> NamedTuple

Derivative-free `(kyhat, width)` critical-factor search using **SimpleDFSane** (the
derivative-free spectral-residual solver the FUSE FluxMatcher defaults to) as the per-seed local
driver. The bumpy AE-onset surface `f★(ky,w)` is multimodal, so global-ness comes from a cheap
AE-onset multistart (`_cheap_onset_seeds`), not from DFSane itself:

  1. Cheap eigenvalue-only (`IFLUX=false`) AE-onset rank over a log-width (down to `w_lo`) × kyhat
     grid → feasible seeds sorted by onset.
  2. For the top `k_descend` well-separated non-pinned seeds, run `SimpleDFSane` on the 2D
     stationarity residual `F(ky,w) = ∇f★(ky,w) = 0`, where the gradient is a forward finite
     difference of `marginal_factor_df` (fully derivative-free) and `(ky,w)` is clamped to the box.
     The lowest `f★` seen along each trajectory is tracked (DFSane can wander on the bumpy surface),
     and both the DFSane optimum and the raw seed are kept as candidates (floor guard).
  3. Faithful-confirm (`IFLUX=true`) the pooled candidates in ascending cheap order with an
     early-stop bound (`_confirm_candidates`) → the critical factor.

Keywords: `gamma_thresh=nothing`, `scan_lo=nothing`(→`shi/512`), `scan_hi=nothing`(→`FACTOR_IN`),
`ky_lo=0.05`, `w_lo=0.05`, `nseed_ky=6`, `nseed_w=10`, `n_eig_seed=12`, `k_descend=3`,
`dfsane_maxiters=25`, `rootsolver=:itp`, `inner=:threads`, `team=nothing`, `use_gpu=false`,
`verbose=false`.

Returns `(; sfmin, kyhat, width, binding, status, converged, faithful, n_confirm, n_samples,
total_evals_full, total_evals_eig)`, `status ∈ (:ok,:no_onset,:cap)`.
"""
function critical_factor_dfsane(ep0::Options{Float64}, prof::profile{Float64};
                                gamma_thresh::Union{Nothing,Float64} = nothing,
                                scan_lo::Union{Nothing,Float64} = nothing,
                                scan_hi::Union{Nothing,Float64} = nothing,
                                ky_lo::Float64 = 0.05, w_lo::Float64 = 0.05,
                                nseed_ky::Int = 6, nseed_w::Int = 10, n_eig_seed::Int = 12,
                                k_descend::Int = 3, dfsane_maxiters::Int = 25,
                                rootsolver::Symbol = :itp,
                                inner::Symbol = :threads, team = nothing,
                                use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? shi / 512.0 : scan_lo
    kylo, kyhi = ky_lo, 1.0
    wlo, whi = w_lo, Float64(ep0.WIDTH_MAX)

    # (1) cheap AE-onset multistart seeds (narrow-width extended)
    cg = _cheap_onset_seeds(ep0, prof, gth; slo = slo, shi = shi, ky_lo = ky_lo, w_lo = w_lo,
                            w_hi = whi, nseed_ky = nseed_ky, nseed_w = nseed_w, n_eig = n_eig_seed,
                            inner = inner, team = team, use_gpu = use_gpu)
    total_eig = cg.eig
    if isempty(cg.seeds)
        return (; sfmin = Inf, kyhat = NaN, width = NaN, binding = :none, status = :no_onset,
                converged = false, faithful = nothing, n_confirm = 0, n_samples = cg.npts,
                total_evals_full = 0, total_evals_eig = total_eig)
    end

    # derivative-free f★(ky,w) via the bracketing inner solve; tracks the running minimum.
    best = Ref((Inf, NaN, NaN))   # (f★, ky, w)
    evals = Ref(0)
    fstar = function (ky::Float64, w::Float64; f_start = nothing)
        ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        mf = marginal_factor_df(ep, prof; gamma_thresh = gth, ae_band = true, f_start = f_start,
                                scan_lo = slo, scan_hi = shi, rootsolver = rootsolver, use_gpu = use_gpu)
        evals[] += mf.evals
        f = (mf.converged && isfinite(mf.factor)) ? mf.factor : Inf
        (isfinite(f) && f < best[][1]) && (best[] = (f, ky, w))
        return f
    end

    # greedily pick well-separated non-pinned seeds (distinct basins)
    picks = NamedTuple[]
    for s in cg.seeds
        s.pinned && continue
        if all(p -> abs(log(s.w) - log(p.w)) > 0.5 || abs(s.ky - p.ky) > 0.2, picks)
            push!(picks, s)
        end
        length(picks) >= k_descend && break
    end
    isempty(picks) && (picks = [cg.seeds[1]])

    # Seed the candidate pool with EVERY feasible cheap onset (floor guards), like `_locate_extended`,
    # so the faithful early-stop confirm considers all cheap basins — not just the DFSane-descended
    # ones. DFSane optima for the top `k_descend` basins are appended below. Without this the winning
    # basin can be pruned before it is confirmed.
    cands = Tuple{Float64,Float64,Float64}[(s.ky, s.w, s.f) for s in cg.seeds]
    for s in picks
        # 2D stationarity residual F(x)=∇f★ via forward differences, x clamped to the box.
        resid! = function (F, u, _p)
            ky = _clamp_to(u[1], kylo, kyhi); w = _clamp_to(u[2], wlo, whi)
            f0 = fstar(ky, w)
            if !isfinite(f0)
                F[1] = 0.0; F[2] = 0.0; return F
            end
            hk = max(1.0e-3, 1.0e-2 * (kyhi - kylo))
            hw = max(1.0e-3, 1.0e-2 * w)
            kyp = ky + hk <= kyhi ? ky + hk : ky - hk        # one-sided step that stays in-box
            wp  = w + hw  <= whi  ? w + hw  : w - hw
            fpk = fstar(kyp, w; f_start = isfinite(f0) ? f0 : nothing)
            fpw = fstar(ky, wp; f_start = isfinite(f0) ? f0 : nothing)
            F[1] = isfinite(fpk) ? (fpk - f0) / (kyp - ky) : 0.0
            F[2] = isfinite(fpw) ? (fpw - f0) / (wp - w) : 0.0
            return F
        end
        u0 = [s.ky, s.w]
        try
            prob = SimpleNonlinearSolve.NonlinearProblem(resid!, u0)
            SimpleNonlinearSolve.solve(prob, SimpleNonlinearSolve.SimpleDFSane();
                                       maxiters = dfsane_maxiters, abstol = 1.0e-6)
        catch e
            e isa InterruptException && rethrow(e)
            verbose && @warn "critical_factor_dfsane: SimpleDFSane failed at seed ($(s.ky),$(s.w)); using tracked best" exception=e
        end
        # append the DFSane-tracked minimum for this basin (raw seed already in `cands`)
        bf, bky, bw = best[]
        isfinite(bf) && push!(cands, (bky, bw, bf))
        best[] = (Inf, NaN, NaN)   # reset per-seed so each basin contributes its own tracked min
    end
    total_eig += evals[]
    # de-dup + sort ascending by cheap/tracked onset for the early-stop confirm
    sort!(cands, by = c -> c[3])

    conf = _confirm_candidates(ep0, prof, gth, cands; slo = slo, shi = shi,
                               inner = inner, team = team, use_gpu = use_gpu)
    total_eig += conf.total_evals_eig
    status = _nls_status(conf.sfmin, conf.binding, shi)
    verbose && @info "critical_factor_dfsane" sfmin=conf.sfmin ky=conf.kyhat w=conf.width binding=conf.binding status=status n_confirm=conf.n_confirm
    return (; sfmin = conf.sfmin, kyhat = conf.kyhat, width = conf.width, binding = conf.binding,
            status = status, converged = (status === :ok), faithful = conf.faithful,
            n_confirm = conf.n_confirm, n_samples = cg.npts,
            total_evals_full = conf.total_evals_full, total_evals_eig = total_eig)
end

"""
    critical_factor_nlopt(inputsEP, inputsPR; kwargs...) -> NamedTuple

Derivative-free `(kyhat, width)` critical-factor search using **NLopt** global derivative-free
optimization on the cheap AE-band onset surface, then a faithful early-stop confirm. This is the
core promotion of the offline `critical_factor_direct` prototype, generalized to a selectable
algorithm and an optional local polish:

  1. `NLopt` global search (`algo`, default `:GN_DIRECT_L`; also `:GN_CRS2_LM`, `:GN_AGS`, ...) over
     `(ky,w)` minimizing the cheap AE onset `f1` (`_ae_unstable_window`, `IFLUX=false`; stable points
     penalized to `FACTOR_IN`), across the narrow-width-extended box `ky∈[ky_lo,1]`, `w∈[w_lo,WIDTH_MAX]`.
  2. Optional local polish (`local_algo`, e.g. `:LN_BOBYQA`/`:LN_COBYLA`) seeded at the global best.
  3. Faithful-confirm (`IFLUX=true`) the sampled points in ascending cheap order with early stop.

Keywords: `algo=:GN_DIRECT_L`, `local_algo=nothing`, `max_evals=40`, `local_evals=20`,
`gamma_thresh=nothing`, `scan_lo=nothing`, `scan_hi=nothing`, `ky_lo=0.05`, `w_lo=0.05`,
`n_eig=16`, `inner=:threads`, `team=nothing`, `use_gpu=false`, `verbose=false`.

Returns the same contract as [`critical_factor_dfsane`](@ref).
"""
function critical_factor_nlopt(ep0::Options{Float64}, prof::profile{Float64};
                               algo::Symbol = :GN_DIRECT_L, local_algo::Union{Nothing,Symbol} = nothing,
                               max_evals::Int = 40, local_evals::Int = 20,
                               gamma_thresh::Union{Nothing,Float64} = nothing,
                               scan_lo::Union{Nothing,Float64} = nothing,
                               scan_hi::Union{Nothing,Float64} = nothing,
                               ky_lo::Float64 = 0.05, w_lo::Float64 = 0.05, n_eig::Int = 16,
                               inner::Symbol = :threads, team = nothing,
                               use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? shi / 512.0 : scan_lo
    kylo, kyhi = ky_lo, 1.0
    wlo, whi = w_lo, Float64(ep0.WIDTH_MAX)
    PENALTY = shi

    samples = NamedTuple[]
    eig = Ref(0)
    cheap_onset = function (ky, w)
        ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        win = _ae_unstable_window(ep, prof, gth; scan_lo = slo, scan_hi = shi,
                                  n_eig = n_eig, threaded = true, use_gpu = use_gpu, inner = inner, team = team)
        eig[] += win.evals
        f = win.unstable ? win.f1 : PENALTY
        push!(samples, (; ky = ky, w = w, f = f, unstable = win.unstable, pinned = win.pinned_lo))
        return f
    end

    # (1) global derivative-free search
    opt = NLopt.Opt(algo, 2)
    NLopt.lower_bounds!(opt, [kylo, wlo])
    NLopt.upper_bounds!(opt, [kyhi, whi])
    NLopt.maxeval!(opt, max_evals)
    NLopt.min_objective!(opt, (x, grad) -> cheap_onset(x[1], x[2]))
    (_minf, minx, ret) = NLopt.optimize(opt, [0.5 * (kylo + kyhi), sqrt(wlo * whi)])

    # (2) optional local polish seeded at the global best
    if local_algo !== nothing
        lopt = NLopt.Opt(local_algo, 2)
        NLopt.lower_bounds!(lopt, [kylo, wlo])
        NLopt.upper_bounds!(lopt, [kyhi, whi])
        NLopt.maxeval!(lopt, local_evals)
        NLopt.min_objective!(lopt, (x, grad) -> cheap_onset(x[1], x[2]))
        try
            NLopt.optimize(lopt, [minx[1], minx[2]])
        catch e
            e isa InterruptException && rethrow(e)
            verbose && @warn "critical_factor_nlopt: local polish ($local_algo) failed" exception=e
        end
    end

    # (3) faithful confirm the sampled unstable points, ascending cheap order, early stop
    unstable = [(s.ky, s.w, Float64(s.f)) for s in samples if s.unstable]
    sort!(unstable, by = c -> c[3])
    conf = _confirm_candidates(ep0, prof, gth, unstable; slo = slo, shi = shi,
                               inner = inner, team = team, use_gpu = use_gpu)
    total_eig = eig[] + conf.total_evals_eig
    status = _nls_status(conf.sfmin, conf.binding, shi)
    verbose && @info "critical_factor_nlopt" algo=algo nlopt_ret=ret sfmin=conf.sfmin ky=conf.kyhat w=conf.width binding=conf.binding status=status
    return (; sfmin = conf.sfmin, kyhat = conf.kyhat, width = conf.width, binding = conf.binding,
            status = status, converged = (status === :ok), faithful = conf.faithful,
            n_confirm = conf.n_confirm, n_samples = length(samples),
            total_evals_full = conf.total_evals_full, total_evals_eig = total_eig)
end
