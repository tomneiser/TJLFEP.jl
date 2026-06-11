# ─────────────────────────────────────────────────────────────────────────────
# TJLFEP AD helpers for ForwardDiff.Dual element types.
#
# The AD-compatible eigen rules (real-symmetric, complex, and the generalized
# complex eigenproblem A r = λ B r) now live in TJLF's TJLFForwardDiffExt and are
# dispatched on TJLF-owned functions (_sym_eigen / _herm_eigen /
# _generalized_eigenvalues) rather than pirating LinearAlgebra.eigen. TJLFEP no
# longer defines any eigen methods here; it only provides the Dual-safe run
# wrapper and a Float64 projection used by the wavefunction path.
# ─────────────────────────────────────────────────────────────────────────────

import ForwardDiff
import TJLF

# ── AD-safe run for non-Float64 element types (Dual, etc.) ───────────────────
#
# TJLF.run's single-ky branch hardcodes zeros(Float64,...)/zeros(ComplexF64,...)
# for result arrays, which fails for Dual. This function is identical to that
# branch but allocates with zeros(T,...). All internal TJLF calls are unchanged.
# Called from TJLFEP_ky via:  T === Float64 ? TJLF.run(x) : _tjlf_run_dual(x, T)
function _tjlf_run_dual(inputTJLF, ::Type{T}; use_gpu::Bool = false) where {T}
    TJLF.checkInput(inputTJLF)
    outputHermite = TJLF.gauss_hermite(inputTJLF)
    satParams     = TJLF.get_sat_params(inputTJLF)
    inputTJLF.KY_SPECTRUM .= TJLF.get_ky_spectrum(inputTJLF, satParams.grad_r0)

    ns     = inputTJLF.NS
    nmodes = inputTJLF.NMODES
    nbasis = inputTJLF.NBASIS_MAX

    nmodes_out, gamma_out, freq_out,
        particle_QL_out, energy_QL_out, stress_tor_QL_out, stress_par_QL_out, exchange_QL_out,
        _ft, field_weight_out_3d, _phi = TJLF.tjlf_LS(inputTJLF, satParams, outputHermite,
                                                       inputTJLF.KY, nbasis, inputTJLF.VEXB_SHEAR, 1;
                                                       use_gpu=use_gpu)

    eigenvalue       = zeros(T,          nmodes, 1, 2)
    QL_weights       = zeros(T,          3, ns, nmodes, 1, 5)
    field_weight_out = zeros(Complex{T}, 3, nbasis, nmodes, 1)

    eigenvalue[:, 1, 1]                      .= gamma_out
    eigenvalue[:, 1, 2]                      .= freq_out
    QL_weights[:, :, 1:nmodes_out, 1, 1]    .= particle_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 2]    .= energy_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 3]    .= stress_tor_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 4]    .= stress_par_QL_out[:, :, 1:nmodes_out]
    QL_weights[:, :, 1:nmodes_out, 1, 5]    .= exchange_QL_out[:, :, 1:nmodes_out]
    field_weight_out[:, :, 1:nmodes_out, 1] .= field_weight_out_3d[:, :, 1:nmodes_out]

    QL_flux_out   = Array{T}(undef, 0, 0, 0)
    flux_spectrum = Array{T}(undef, 0, 0, 0)
    return (QL_weights=QL_weights, eigenvalue=eigenvalue, QL_flux_out=QL_flux_out,
            flux_spectrum=flux_spectrum, field_weight_out=field_weight_out)
end

# ── Strip Dual partials from InputTJLF → InputTJLF{Float64} ─────────────────
# Used before calling get_wavefunction, which has Float64-typed geometry arrays
# internally. The wavefunction shape is independent of the differentiated
# parameters, so extracting .value is exact (no information lost).
#
# NOTE: `src` may be either TJLFEP.InputTJLF or TJLF.InputTJLF; no type
# annotation so both are accepted. Runtime field copy works for either.
function _to_float64_input(src)
    dst = TJLF.InputTJLF{Float64}(src.NS, src.NKY)
    for fn in fieldnames(typeof(src))
        v = getfield(src, fn)
        if v isa ForwardDiff.Dual
            setfield!(dst, fn, ForwardDiff.value(v))
        elseif v isa AbstractArray && eltype(v) <: ForwardDiff.Dual
            setfield!(dst, fn, ForwardDiff.value.(v))
        elseif v isa AbstractArray && eltype(v) <: Complex && real(eltype(v)) <: ForwardDiff.Dual
            setfield!(dst, fn, map(x -> ComplexF64(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x))), v))
        else
            try; setfield!(dst, fn, v); catch; end
        end
    end
    return dst
end

# ── Promote a Float64 InputTJLF → InputTJLF{D} (D <: ForwardDiff.Dual) ────────
# Inverse of `_to_float64_input`: real scalars/arrays are widened to the Dual
# element type with zero partials (they are constants w.r.t. the seeded
# variable), complex arrays become Complex{D}, and Int/Bool/String/Missing
# fields are copied verbatim. The caller then overwrites the handful of fields
# that actually depend on the differentiated parameter with truly Dual values.
function _to_dual_input(src, ::Type{D}) where {D<:ForwardDiff.Dual}
    dst = TJLF.InputTJLF{D}(src.NS, src.NKY)
    for fn in fieldnames(typeof(src))
        v = getfield(src, fn)
        if v isa AbstractFloat
            setfield!(dst, fn, D(v))
        elseif v isa AbstractArray && eltype(v) <: AbstractFloat
            setfield!(dst, fn, D.(v))
        elseif v isa AbstractArray && eltype(v) <: Complex
            setfield!(dst, fn, map(x -> Complex{D}(D(real(x)), D(imag(x))), v))
        else
            try; setfield!(dst, fn, v); catch; end
        end
    end
    return dst
end

# ── Static InputTJLF configuration shared with TJLFEP_ky ─────────────────────
# Mirrors the in-place overrides applied right after `TJLF_map` inside
# `TJLFEP_ky` (single-ky path: transport model off, fixed width/basis, eigen
# solve on, TGLF-EP physics switches). Factored out so the AD building block
# below configures the eigensolve input identically to the production scan
# without re-running the rest of TJLFEP_ky. The validation script asserts that a
# Float64 input built this way reproduces TJLFEP_ky's growth rates exactly.
function _configure_inputTJLF_for_ky!(inputTJLF, inputsEP)
    inputTJLF.USE_TRANSPORT_MODEL = false
    inputTJLF.KYGRID_MODEL        = 0
    inputTJLF.NMODES              = inputsEP.NMODES
    inputTJLF.NBASIS_MIN          = inputsEP.N_BASIS
    inputTJLF.NBASIS_MAX          = inputsEP.N_BASIS
    inputTJLF.NXGRID              = 32
    inputTJLF.WIDTH               = inputsEP.WIDTH_IN
    inputTJLF.FIND_WIDTH          = false
    inputTJLF.USE_AVE_ION_GRID    = false
    inputTJLF.WIDTH_SPECTRUM     .= inputTJLF.WIDTH
    inputTJLF.FIND_EIGEN          = true
    # AD only needs eigenvalues (growth rate): turn off the QL-flux / eigenvector
    # block in tjlf_LS, whose per-mode inverse-iteration LU (lu!(zmat)) is by far
    # the most expensive Dual operation and is unused by gamma_dgamma_dfactor.
    # The eigenvalue solve runs before that block, so γ is unchanged.
    inputTJLF.IFLUX               = false
    inputTJLF.RLNP_CUTOFF         = 18.0
    inputTJLF.BETA_LOC            = 0.0
    inputTJLF.DAMP_PSI            = 0.0
    inputTJLF.DAMP_SIG            = 0.0
    inputTJLF.WDIA_TRAPPED        = 0.0
    if inputTJLF.SAT_RULE == 2 || inputTJLF.SAT_RULE == 3
        inputTJLF.UNITS        = "CGYRO"
        inputTJLF.XNU_MODEL    = 3
        inputTJLF.WDIA_TRAPPED = 1.0
    end
    inputTJLF.KX0_LOC = 0.0
    return inputTJLF
end

# ── Promote a Float64 value/array to the Dual element type (constants) ───────
_to_dual_val(v::AbstractFloat, ::Type{D}) where {D<:ForwardDiff.Dual} = D(v)
_to_dual_val(v::AbstractArray{<:AbstractFloat}, ::Type{D}) where {D<:ForwardDiff.Dual} = D.(v)
_to_dual_val(v, ::Type{D}) where {D<:ForwardDiff.Dual} = v   # Int/Bool/String/Missing/Vector{Int}/…

# Copy every field of `src` into `dst` (same struct, Dual element type), widening
# real scalars/arrays to `D` with zero partials. Fields named in `seed` are taken
# verbatim from `seed` instead (used to inject the truly-differentiated input).
function _promote_struct_to_dual!(dst, src, ::Type{D}, seed) where {D<:ForwardDiff.Dual}
    for fn in fieldnames(typeof(src))
        if haskey(seed, fn)
            setfield!(dst, fn, seed[fn])
        else
            setfield!(dst, fn, _to_dual_val(getfield(src, fn), D))
        end
    end
    return dst
end

# Options{Float64} → Options{D}, with FACTOR_IN replaced by the seeded Dual.
function _to_dual_options(src::Options{Float64}, factor_seed::D, nr::Int) where {D<:ForwardDiff.Dual}
    # Constructor dims only size scratch arrays that _promote_struct_to_dual!
    # immediately overwrites, so coalesce any `missing` dims to a safe value.
    dst = Options{D}(src.SCAN_N, src.WIDTH_IN_FLAG, coalesce(src.NN, 1), nr, coalesce(src.JTSCALE_MAX, 1), src.NMODES)
    return _promote_struct_to_dual!(dst, src, D, Dict{Symbol,Any}(:FACTOR_IN => factor_seed))
end

# profile{Float64} → profile{D} (all entries are constants w.r.t. the factor).
function _to_dual_profile(src::profile{Float64}, ::Type{D}) where {D<:ForwardDiff.Dual}
    nr, ns = size(src.AS)
    dst = profile{D}(nr, ns)
    return _promote_struct_to_dual!(dst, src, D, Dict{Symbol,Any}())
end

"""
    gamma_dgamma_dfactor(inputsEP, inputsPR; use_gpu=false)

Single-point AD building block for the EP critical-gradient scan.

Evaluates the TGLF-EP dispersion at one `(KYHAT_IN, WIDTH_IN, FACTOR_IN, IR)`
operating point — exactly as the inner `kwscale_scan` combo does — and returns
both the per-mode growth rate `γ` and its exact derivative `dγ/d(FACTOR_IN)`,
computed in a single forward-mode AD pass by seeding a `ForwardDiff.Dual` on the
EP scale factor.

The derivative is taken *through `TJLF_map`* (by promoting `Options`/`profile` to
a Dual element type and seeding `FACTOR_IN`), so every channel by which the
factor reaches the TGLF input is captured automatically — not only the EP density
gradient `RLNS[is]` / density `AS`, but also the pressure-gradient (`P_PRIME_LOC`,
MHD-α) stabilization term, which for `PPRIME_METHOD ∈ {2,3}` varies with the
scaled EP gradient via the `sum*/sum1` ratio. Hand-replicating the factor→input
map is brittle for exactly this reason.

Returns a `NamedTuple`:
  - `gamma`            :: Vector{Float64}  per-mode growth rate (length NMODES)
  - `dgamma_dfactor`   :: Vector{Float64}  per-mode dγ/d(FACTOR_IN)
  - `freq`             :: Vector{Float64}  per-mode real frequency
  - `dfreq_dfactor`    :: Vector{Float64}  per-mode d(freq)/d(FACTOR_IN)
  - `factor`           :: Float64          the (clamped) factor the result is evaluated at

`inputsEP.IR`, `FACTOR_IN`, `KYHAT_IN`, and `WIDTH_IN` must be set on entry (as
the scan sets them). `inputsEP`/`inputsPR` are not mutated.
"""
function gamma_dgamma_dfactor(inputsEP::Options{Float64}, inputsPR::profile{Float64}; use_gpu::Bool = false)
    f0 = Float64(inputsEP.FACTOR_IN)

    Tag = typeof(ForwardDiff.Tag(gamma_dgamma_dfactor, Float64))
    D   = ForwardDiff.Dual{Tag, Float64, 1}
    s   = ForwardDiff.Dual{Tag}(f0, one(Float64))   # seed: value=f0, ∂/∂factor=1

    nr = size(inputsPR.AS, 1)
    epD = _to_dual_options(inputsEP, s, nr)
    prD = _to_dual_profile(inputsPR, D)

    # Differentiate through the full factor→TGLF-input map.
    inputD = TJLF_map(epD, prD)
    inputD isa Integer && error("gamma_dgamma_dfactor: TJLF_map rejected IR=$(inputsEP.IR) (out of range)")
    _configure_inputTJLF_for_ky!(inputD, epD)

    fclamp = ForwardDiff.value(epD.FACTOR_IN)   # post-clamp factor TJLF_map actually used
    if fclamp <= 0 || fclamp >= ForwardDiff.value(epD.FACTOR_MAX)
        @warn "gamma_dgamma_dfactor: FACTOR_IN=$fclamp is at/over the clamp range [0, $(ForwardDiff.value(epD.FACTOR_MAX))]; dγ/dfactor assumes an interior point"
    end

    res = _tjlf_run_dual(inputD, D; use_gpu = use_gpu)
    g = res.eigenvalue[:, 1, 1]   # [nmodes] growth rate (Dual)
    f = res.eigenvalue[:, 1, 2]   # [nmodes] frequency   (Dual)

    return (
        gamma          = ForwardDiff.value.(g),
        dgamma_dfactor = [ForwardDiff.partials(x, 1) for x in g],
        freq           = ForwardDiff.value.(f),
        dfreq_dfactor  = [ForwardDiff.partials(x, 1) for x in f],
        factor         = fclamp,
    )
end

# Leading growth rate γ_lead = max over modes, and its dγ_lead/d(factor), at a
# given factor. In the EP single-ky path the most-unstable mode is the EP-driven
# AE, so γ_lead(factor) is the instability indicator the marginal scan tracks.
function _gamma_lead_dfactor(inputsEP::Options{Float64}, inputsPR::profile{Float64}, f::Float64; use_gpu::Bool = false)
    ep = deepcopy(inputsEP)
    ep.FACTOR_IN = f
    r = gamma_dgamma_dfactor(ep, inputsPR; use_gpu = use_gpu)
    i = argmax(r.gamma)
    return r.gamma[i], r.dgamma_dfactor[i], r
end

"""
    marginal_factor(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD-accelerated Newton root-find for the marginal EP scale factor at a fixed
`(IR, KYHAT_IN, WIDTH_IN)` operating point: the factor `f★` at which the leading
growth rate crosses the threshold,

    γ_lead(f★) = gamma_thresh .

This is the Julia-native, derivative-based complement to the Fortran 1-D factor
scan: instead of bracketing on an 8-point factor grid and taking a single secant
step, it runs a **safeguarded Newton** (Newton step with a bisection fallback)
using the exact `dγ_lead/d(factor)` from `gamma_dgamma_dfactor`, so it converges
quadratically and needs only a handful of TGLF eigensolves.

Keywords:
  - `gamma_thresh = 1e-7`           threshold (matches the default `GAMMA_THRESH`)
  - `bracket = nothing`             `(f_lo, f_hi)` bracketing the sign change of
                                    `g(f)=γ_lead(f)-gamma_thresh`; auto-found if `nothing`
  - `scan_lo=1e-3, scan_hi=10.0, nscan=8`  geometric samples used to auto-bracket
  - `tol = 1e-6`                    converged when `|g| < tol·(1+|γ_thresh|)` or step `< xtol`
  - `xtol = 1e-8`, `maxiter = 40`
  - `use_gpu = false`, `verbose = false`

Returns `(; factor, gamma_lead, gamma, freq, iters, evals, converged, bracket)`.
"""
function marginal_factor(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                         gamma_thresh::Float64 = 1.0e-7,
                         bracket::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                         scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0, nscan::Int = 8,
                         tol::Float64 = 1.0e-6, xtol::Float64 = 1.0e-8, maxiter::Int = 40,
                         use_gpu::Bool = false, verbose::Bool = false)
    evals = Ref(0)
    g(f) = begin
        gl, dgl, r = _gamma_lead_dfactor(inputsEP, inputsPR, f; use_gpu = use_gpu)
        evals[] += 1
        verbose && @info "eval" factor=f gamma_lead=gl gobj=(gl - gamma_thresh) dgdf=dgl
        (gl - gamma_thresh, dgl, gl, r)
    end

    # ── Establish a bracket [a,b] with g(a) < 0 < g(b) ──
    local a, b, ga, gb
    if bracket === nothing
        fs = exp.(range(log(scan_lo), log(scan_hi); length = nscan))
        prev_f = fs[1]; prev_g = g(prev_f)[1]
        a = b = NaN
        for f in fs[2:end]
            gf = g(f)[1]
            if prev_g < 0 && gf >= 0
                a, ga, b, gb = prev_f, prev_g, f, gf
                break
            end
            prev_f, prev_g = f, gf
        end
        if isnan(a)
            @warn "marginal_factor: no sign change of γ_lead-thresh in [$scan_lo, $scan_hi] (always stable or always unstable); returning best endpoint"
            return (; factor = prev_g < 0 ? scan_hi : scan_lo, gamma_lead = prev_g + gamma_thresh,
                    gamma = Float64[], freq = Float64[], iters = 0, evals = evals[], converged = false, bracket = (scan_lo, scan_hi))
        end
    else
        a, b = bracket
        ga = g(a)[1]; gb = g(b)[1]
        ga < 0 < gb || error("marginal_factor: provided bracket ($a,$b) does not satisfy g(a)<0<g(b): g(a)=$ga g(b)=$gb")
    end

    # ── Safeguarded Newton (rtsafe) ──
    f = 0.5 * (a + b)
    gf, dgf, gl, r = g(f)
    iters = 0
    converged = false
    for it in 1:maxiter
        iters = it
        # keep the bracket tight
        if gf < 0; a = f; else; b = f; end
        # Newton step, fall back to bisection if it leaves the bracket or dg≈0
        if abs(dgf) > 1e-30
            fn = f - gf / dgf
        else
            fn = NaN
        end
        if !(a < fn < b) || isnan(fn)
            fn = 0.5 * (a + b)   # bisection fallback
        end
        step = abs(fn - f)
        f = fn
        gf, dgf, gl, r = g(f)
        verbose && @info "newton" iter=it factor=f gobj=gf step=step
        if abs(gf) < tol * (1 + abs(gamma_thresh)) || step < xtol
            converged = true
            break
        end
    end

    return (; factor = f, gamma_lead = gl, gamma = r.gamma, freq = r.freq,
            iters = iters, evals = evals[], converged = converged, bracket = (a, b))
end

"""
    critical_factor_grid(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD analogue of the inner `kwscale_scan`: the critical EP scale factor is the
*minimum over the `(kyhat, width)` grid* of the per-point marginal factor. For
each grid point this calls `marginal_factor` (AD-Newton) instead of the
brute-force 8-factor × 4-round bracket, so it reaches `sfmin` in far fewer TGLF
eigensolves.

The `(kyhat, width)` grid mirrors `kwscale_scan`'s first round:
  - `kyhat[i] = (1/nkyhat)·i`,  i=1..nkyhat        (kyhat∈(0,1])
  - `width[i] = WIDTH_MIN + (WIDTH_MAX-WIDTH_MIN)/(nefwid-1)·(i-1)`

Keywords: `nkyhat=4, nefwid=8`, `gamma_thresh=1e-7`, `scan_lo=1e-3`,
`scan_hi=nothing` (defaults to the incoming `FACTOR_IN`), `threaded=true`,
`use_gpu=false`.

Returns `(; sfmin, kyhat, width, gamma, total_evals, results)` where `results`
is the per-grid-point table.
"""
function critical_factor_grid(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                              nkyhat::Int = 4, nefwid::Int = 8,
                              gamma_thresh::Float64 = 1.0e-7,
                              scan_lo::Float64 = 1.0e-3, scan_hi::Union{Nothing,Float64} = nothing,
                              threaded::Bool = true, use_gpu::Bool = false)
    w0 = Float64(inputsEP.WIDTH_MIN); w1 = Float64(inputsEP.WIDTH_MAX)
    kyhats = [(1.0 / nkyhat) * i for i in 1:nkyhat]
    widths = [w0 + (w1 - w0) / (nefwid - 1) * (i - 1) for i in 1:nefwid]
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi

    pts = [(ky, w) for ky in kyhats for w in widths]
    npts = length(pts)
    out = Vector{Any}(undef, npts)

    work = function (idx)
        ky, w = pts[idx]
        ep = deepcopy(inputsEP); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        r = marginal_factor(ep, inputsPR; gamma_thresh = gamma_thresh,
                            scan_lo = scan_lo, scan_hi = shi, use_gpu = use_gpu)
        out[idx] = (kyhat = ky, width = w, factor = r.factor, gamma = r.gamma,
                    converged = r.converged, evals = r.evals)
    end

    if threaded
        TJLF.with_blas_threads(1) do
            Threads.@threads for idx in 1:npts
                work(idx)
            end
        end
    else
        for idx in 1:npts
            work(idx)
        end
    end

    total_evals = sum(o.evals for o in out)
    best = (factor = Inf, kyhat = NaN, width = NaN, gamma = Float64[])
    for o in out
        if o.converged && o.factor < best.factor
            best = (factor = o.factor, kyhat = o.kyhat, width = o.width, gamma = o.gamma)
        end
    end

    return (; sfmin = best.factor, kyhat = best.kyhat, width = best.width,
            gamma = best.gamma, total_evals = total_evals, results = out)
end
