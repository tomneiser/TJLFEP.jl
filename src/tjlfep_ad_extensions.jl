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

# Options{Float64} → Options{D} with an arbitrary set of seeded Dual fields.
# `seeds` maps EP field names (e.g. :FACTOR_IN, :KYHAT_IN, :WIDTH_IN) to the
# multi-partial Dual to inject; every other field is widened as a constant.
function _to_dual_options_seeded(src::Options{Float64}, seeds::AbstractDict, ::Type{D}, nr::Int) where {D<:ForwardDiff.Dual}
    dst = Options{D}(src.SCAN_N, src.WIDTH_IN_FLAG, coalesce(src.NN, 1), nr, coalesce(src.JTSCALE_MAX, 1), src.NMODES)
    return _promote_struct_to_dual!(dst, src, D, seeds)
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

# Operating-point variables that gamma_grad can seed. Each reaches the eigensolve
# through TJLF_map / _configure_inputTJLF_for_ky!:
#   FACTOR_IN → EP density/gradient + P_PRIME_LOC (MHD-α) channels
#   KYHAT_IN  → inputTJLF.KY  (TJLF_map KY_MODEL=3, line ~1092)
#   WIDTH_IN  → inputTJLF.WIDTH / WIDTH_SPECTRUM (_configure_inputTJLF_for_ky!)
const _GAMMA_GRAD_VARS = (:FACTOR_IN, :KYHAT_IN, :WIDTH_IN)

"""
    gamma_grad(inputsEP, inputsPR; vars=(:FACTOR_IN,:KYHAT_IN,:WIDTH_IN), use_gpu=false)

Multi-variable generalization of [`gamma_dgamma_dfactor`](@ref): one forward-mode
AD pass returning the per-mode growth rate `γ` and real frequency together with
their exact partials w.r.t. each operating-point variable in `vars` — any subset
of `(:FACTOR_IN, :KYHAT_IN, :WIDTH_IN)`. All partials are obtained in a single
N-partial `ForwardDiff.Dual` solve (one eigensolve), which is the building block
for the `(kyhat, width)` optimizer (Phase 3) and the freq/keep-condition root
finds (Phase 2).

Derivatives are taken *through `TJLF_map`* (by promoting `Options`/`profile` to
the Dual element type and seeding the chosen EP fields), so every channel by which
a variable reaches the TGLF input is captured automatically. `IFLUX=false`, so
only the eigenvalue block runs.

Returns a `NamedTuple`:
  - `gamma`  :: Vector{Float64}          per-mode growth rate (length NMODES)
  - `freq`   :: Vector{Float64}          per-mode real frequency
  - `dgamma` :: Matrix{Float64}          `NMODES × length(vars)`, `∂γ_m/∂vars[k]`
  - `dfreq`  :: Matrix{Float64}          `NMODES × length(vars)`, `∂freq_m/∂vars[k]`
  - `vars`   :: NTuple                   the variable order matching the columns

`inputsEP.IR`, `FACTOR_IN`, `KYHAT_IN`, `WIDTH_IN` must be set on entry (as the
scan sets them). `inputsEP`/`inputsPR` are not mutated.
"""
function gamma_grad(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                    vars::NTuple{N,Symbol} = _GAMMA_GRAD_VARS,
                    use_gpu::Bool = false) where {N}
    all(v -> v in _GAMMA_GRAD_VARS, vars) ||
        error("gamma_grad: vars must be a subset of $_GAMMA_GRAD_VARS; got $vars")

    Tag = typeof(ForwardDiff.Tag(gamma_grad, Float64))
    D   = ForwardDiff.Dual{Tag, Float64, N}

    seeds = Dict{Symbol,Any}()
    for (i, v) in enumerate(vars)
        x0 = Float64(getfield(inputsEP, v))
        seeds[v] = ForwardDiff.Dual{Tag}(x0, ntuple(j -> j == i ? 1.0 : 0.0, N)...)
    end

    nr  = size(inputsPR.AS, 1)
    epD = _to_dual_options_seeded(inputsEP, seeds, D, nr)
    prD = _to_dual_profile(inputsPR, D)

    inputD = TJLF_map(epD, prD)
    inputD isa Integer && error("gamma_grad: TJLF_map rejected IR=$(inputsEP.IR) (out of range)")
    _configure_inputTJLF_for_ky!(inputD, epD)

    if :FACTOR_IN in vars
        fclamp = ForwardDiff.value(epD.FACTOR_IN)
        if fclamp <= 0 || fclamp >= ForwardDiff.value(epD.FACTOR_MAX)
            @warn "gamma_grad: FACTOR_IN=$fclamp is at/over the clamp range [0, $(ForwardDiff.value(epD.FACTOR_MAX))]; ∂γ/∂FACTOR_IN assumes an interior point"
        end
    end

    res = _tjlf_run_dual(inputD, D; use_gpu = use_gpu)
    g = res.eigenvalue[:, 1, 1]
    f = res.eigenvalue[:, 1, 2]
    NM = length(g)

    gamma  = ForwardDiff.value.(g)
    freq   = ForwardDiff.value.(f)
    dgamma = [ForwardDiff.partials(g[m], i) for m in 1:NM, i in 1:N]
    dfreq  = [ForwardDiff.partials(f[m], i) for m in 1:NM, i in 1:N]

    return (; gamma, freq, dgamma, dfreq, vars)
end

# AE-band upper frequency cut (FREQ_AE_UPPER = -|ω_GAM|): modes with freq below it
# are in the Alfvén-eigenmode band. Read from inputsEP if set (the standard read
# path sets it), else recomputed from the profile.
function _ae_band_upper(inputsEP::Options{Float64}, inputsPR::profile{Float64})
    fu = inputsEP.FREQ_AE_UPPER
    if !(fu === missing) && fu isa Real && !isnan(fu)
        return Float64(fu)
    end
    return -abs(inputsPR.omegaGAM[inputsEP.IR])
end

# Marginal growth-rate threshold γ* for the current radius, reproducing exactly
# what kwscale_scan/TJLF_map set on inputsEP.GAMMA_THRESH:
#   ROTATIONAL_SUPPRESSION_FLAG==1 : γ* = min(0.15·|γ_E/ŝ|, |γ_p|·2·min(1-r/a,r/a)/Rmaj)
#                                    (Bass PoP 2017 flow-shear AE suppression) — a
#                                    FINITE threshold ⇒ γ_keep(factor)=γ* is a smooth,
#                                    grid-independent root suitable for AD-Newton.
#   otherwise                      : γ* = 1e-7 (the "any positive AE growth" onset).
function _gamma_thresh_for(inputsEP::Options{Float64}, inputsPR::profile{Float64})
    ir = inputsEP.IR
    if coalesce(inputsEP.ROTATIONAL_SUPPRESSION_FLAG, 0) == 1
        r_over_a = inputsPR.RMIN[ir] / inputsPR.RMIN[end]
        gthr_max = abs(inputsPR.gammap[ir]) * 2.0 * (min(1.0 - r_over_a, r_over_a) / inputsPR.RMAJ[ir])
        gthr = 0.15 * abs(inputsPR.gammaE[ir] / inputsPR.SHEAR[ir])
        return min(gthr, gthr_max)
    else
        return 1.0e-7
    end
end

# Leading growth rate γ_lead and its dγ_lead/d(factor) at a given factor.
#
# `ae_band=false` (default): γ_lead = max over all modes — the raw indicator. This
# is dominated by background ITG/TEM modes that are unstable even at tiny EP
# factor and can be non-monotonic in factor, so it is unsuitable for an
# endpoint-only stability test.
#
# `ae_band=true`: γ_lead = max over only the modes whose real frequency lies in
# the AE band (freq < FREQ_AE_UPPER), mirroring the *primary* keep filter in
# TJLFEP_ky. This isolates the EP-driven Alfvén eigenmode (the instability the
# marginal scan actually tracks); if no mode is in-band the point is AE-stable and
# γ_lead is reported as 0. Uses only eigenvalues/frequencies, so it is compatible
# with the IFLUX=false fast path. (The secondary tearing/pinch/QL/θ² rejections
# need the wavefunction/QL weights, i.e. IFLUX=true, and are not applied here.)
function _gamma_lead_dfactor(inputsEP::Options{Float64}, inputsPR::profile{Float64}, f::Float64;
                             use_gpu::Bool = false, ae_band::Bool = false)
    ep = deepcopy(inputsEP)
    ep.FACTOR_IN = f
    r = gamma_dgamma_dfactor(ep, inputsPR; use_gpu = use_gpu)
    if ae_band
        fu = _ae_band_upper(inputsEP, inputsPR)
        cand = findall(<(fu), r.freq)
        isempty(cand) && return 0.0, 0.0, r   # no AE-band mode → AE-stable
        j = cand[argmax(@view r.gamma[cand])]
        return r.gamma[j], r.dgamma_dfactor[j], r
    end
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
  - `gamma_thresh = nothing`        threshold; `nothing` ⇒ the case's own
                                    `GAMMA_THRESH` (rotational γ* if
                                    `ROTATIONAL_SUPPRESSION_FLAG=1`, else 1e-7)
  - `bracket = nothing`             `(f_lo, f_hi)` bracketing the sign change of
                                    `g(f)=γ_lead(f)-gamma_thresh`; auto-found if `nothing`
  - `scan_lo=1e-3, scan_hi=10.0, nscan=8`  geometric samples used to auto-bracket
  - `ae_band = false`               track only the AE-band-filtered γ (freq <
                                    `FREQ_AE_UPPER`) instead of the raw max-over-modes γ
  - `tol = 1e-6`                    converged when `|g| < tol·(1+|γ_thresh|)` or step `< xtol`
  - `xtol = 1e-8`, `maxiter = 40`
  - `use_gpu = false`, `verbose = false`

Returns `(; factor, gamma_lead, gamma, freq, iters, evals, converged, bracket)`.
"""
function marginal_factor(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                         gamma_thresh::Union{Nothing,Float64} = nothing,
                         bracket::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                         f_start::Union{Nothing,Float64} = nothing,
                         scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0, nscan::Int = 8,
                         ae_band::Bool = false,
                         tol::Float64 = 1.0e-6, xtol::Float64 = 1.0e-8, maxiter::Int = 40,
                         use_gpu::Bool = false, verbose::Bool = false)
    # nothing ⇒ use the case's own threshold (rotational γ* or 1e-7), matching kwscale_scan
    gamma_thresh = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    evals = Ref(0)
    g(f) = begin
        gl, dgl, r = _gamma_lead_dfactor(inputsEP, inputsPR, f; use_gpu = use_gpu, ae_band = ae_band)
        evals[] += 1
        verbose && @info "eval" factor=f gamma_lead=gl gobj=(gl - gamma_thresh) dgdf=dgl
        (gl - gamma_thresh, dgl, gl, r)
    end

    # ── Establish a bracket [a,b] with g(a) < 0 < g(b) ──
    local a, b, ga, gb
    a = b = NaN
    # Warm start: build a cheap LOCAL bracket by geometric expansion around f_start
    # (the previous root in a continuation/optimizer), avoiding the nscan global scan.
    if bracket === nothing && f_start !== nothing && scan_lo < f_start < scan_hi
        gs = g(f_start)[1]
        if gs >= 0           # at/above onset → step DOWN for a (g<0)
            b, gb = f_start, gs
            fa = f_start
            for _ in 1:8
                fa = max(scan_lo, fa / 1.5)
                gfa = g(fa)[1]
                if gfa < 0; a, ga = fa, gfa; break end
                fa <= scan_lo && break
            end
        else                 # below onset → step UP for b (g≥0)
            a, ga = f_start, gs
            fb = f_start
            for _ in 1:8
                fb = min(scan_hi, fb * 1.5)
                gfb = g(fb)[1]
                if gfb >= 0; b, gb = fb, gfb; break end
                fb >= scan_hi && break
            end
        end
    end
    if isnan(a) || isnan(b)
        a = b = NaN
        if bracket === nothing
            fs = exp.(range(log(scan_lo), log(scan_hi); length = nscan))
            prev_f = fs[1]; prev_g = g(prev_f)[1]
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
                              gamma_thresh::Union{Nothing,Float64} = nothing,
                              scan_lo::Float64 = 1.0e-3, scan_hi::Union{Nothing,Float64} = nothing,
                              ae_band::Bool = false,
                              threaded::Bool = true, use_gpu::Bool = false)
    # nothing ⇒ use the case's own threshold (rotational γ* or 1e-7); IR-dependent only,
    # so it is identical for every (kyhat,width) grid point.
    gamma_thresh = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
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
                            scan_lo = scan_lo, scan_hi = shi, ae_band = ae_band, use_gpu = use_gpu)
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

# ── Faithful keep evaluation at one operating point ──────────────────────────
# Runs the FULL production single-ky path (TJLFEP_ky, IFLUX=true) so every keep
# filter is applied: band-entry (freq<FREQ_AE_UPPER) and growth (g>GAMMA_THRESH),
# plus the wavefunction/QL secondary rejections (tearing, ion/elec/thermal/EP
# pinch, QL-ratio, θ²). This is the SAME keep boolean kwscale_scan reduces to
# `imark`/`sfmin`, so it is the ground-truth target the AD root must match.
# (`inputsEP` is not mutated.)
function keep_at(inputsEP::Options{Float64}, inputsPR::profile{Float64}, factor::Float64;
                 kyhat::Float64 = Float64(inputsEP.KYHAT_IN),
                 width::Float64 = Float64(inputsEP.WIDTH_IN),
                 use_gpu::Bool = false)
    ep = deepcopy(inputsEP)
    ep.KYHAT_IN  = kyhat
    ep.WIDTH_IN  = width
    ep.FACTOR_IN = factor
    g, f, = TJLFEP_ky(ep, inputsPR, "", 0; eigen_cache = nothing, use_gpu = use_gpu)
    nm = ep.NMODES
    keep = collect(@view ep.LKEEP[1:nm])
    return (; factor, any_keep = any(keep), LKEEP = keep,
            gamma = g[1:nm], freq = f[1:nm],
            FREQ_AE_UPPER = Float64(ep.FREQ_AE_UPPER), GAMMA_THRESH = Float64(ep.GAMMA_THRESH),
            LTEARING = collect(@view ep.LTEARING[1:nm]),
            L_I_PINCH = collect(@view ep.L_I_PINCH[1:nm]),
            L_E_PINCH = collect(@view ep.L_E_PINCH[1:nm]),
            L_TH_PINCH = collect(@view ep.L_TH_PINCH[1:nm]),
            L_EP_PINCH = collect(@view ep.L_EP_PINCH[1:nm]),
            L_QL_RATIO = collect(@view ep.L_QL_RATIO[1:nm]),
            L_THETA_SQ = collect(@view ep.L_THETA_SQ[1:nm]))
end

# Which keep sub-condition rejects mode `n` at a faithful keep evaluation `kf`
# (the FIRST that fails, in TJLFEP_ky's evaluation order). Returns :kept if kept.
function _why_rejected(kf, n::Int)
    kf.LKEEP[n]                      && return :kept
    !(kf.freq[n] < kf.FREQ_AE_UPPER) && return :band          # not in AE band
    !(kf.gamma[n] > kf.GAMMA_THRESH) && return :below_thresh   # growth below γ*
    kf.LTEARING[n]                   && return :tearing
    kf.L_I_PINCH[n]                  && return :i_pinch
    kf.L_E_PINCH[n]                  && return :e_pinch
    kf.L_TH_PINCH[n]                 && return :th_pinch
    kf.L_EP_PINCH[n]                 && return :ep_pinch
    kf.L_QL_RATIO[n]                 && return :ql_ratio
    kf.L_THETA_SQ[n]                 && return :theta_sq
    return :unknown
end

# Parallel-map dispatcher for the AD path's independent-evaluation regions. With an MPS
# team (`inner=:mps_team`, non-empty `team`) it reuses kwscale_scan's chunked remotecall
# helper `_inner_team_map` (one round-trip per worker; each worker is a separate CUDA
# context so GPU solves overlap via Hyper-Q). Otherwise it runs in-process with threads
# (BLAS pinned to 1 so the dense LAPACK/CUSOLVER per-eval doesn't oversubscribe). Returns
# results in index order, `f(i)` for `i in 1:n`.
function _ad_pmap(f, n::Int; inner::Symbol = :threads,
                  team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    if inner === :mps_team && team !== nothing && !isempty(team)
        return _inner_team_map(f, team, n)
    end
    res = Vector{Any}(undef, n)
    TJLF.with_blas_threads(1) do
        Threads.@threads for i in 1:n
            res[i] = f(i)
        end
    end
    return res
end

# AE-band unstable hull [f1,f2]: the factor range over which the leading AE-band
# (freq<FREQ_AE_UPPER) growth rate exceeds γ*, from a cheap eigenvalue-only scan
# (IFLUX=false). The keep window is a SUBSET of this (keep requires g>γ* and
# in-band), so a faithful sweep restricted to [f1,f2] is complete — and the AE
# γ(factor) is a narrow "bump", so an unbounded geometric sweep can step over the
# kept window entirely. `f1` is Newton-refined (smooth, grid-independent onset of
# instability) when the up-crossing is bracketed inside the scan; `f2` is padded
# to the next stable sample above the bump. `pinned_lo=true` flags that γ_AE>γ*
# already at scan_lo (no interior onset — the true edge is below the scan range).
# Returns `(; f1, f2, evals, unstable, pinned_lo)`.
function _ae_unstable_window(ep0::Options{Float64}, inputsPR::profile{Float64}, gth::Float64;
                             scan_lo::Float64, scan_hi::Float64, n_eig::Int,
                             threaded::Bool = true, use_gpu::Bool = false,
                             inner::Symbol = :threads,
                             team::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    fe  = exp.(range(log(scan_lo), log(scan_hi); length = n_eig))
    gae = Vector{Float64}(undef, n_eig)
    if threaded
        vals = _ad_pmap(i -> _gamma_lead_dfactor(ep0, inputsPR, fe[i]; use_gpu = use_gpu, ae_band = true)[1],
                        n_eig; inner = inner, team = team)
        for i in 1:n_eig
            gae[i] = vals[i]::Float64
        end
    else
        for i in 1:n_eig
            gae[i] = _gamma_lead_dfactor(ep0, inputsPR, fe[i]; use_gpu = use_gpu, ae_band = true)[1]
        end
    end
    unst = findall(>(gth), gae)
    isempty(unst) && return (; f1 = NaN, f2 = NaN, evals = n_eig, unstable = false, pinned_lo = false)
    i1 = first(unst); i2 = last(unst)
    evals = n_eig
    # Newton-refine the lower up-crossing of γ_AE(f)=γ* within [fe[i1-1],fe[i1]].
    f1 = fe[i1]
    pinned_lo = (i1 == 1)
    if i1 > 1
        rm = marginal_factor(ep0, inputsPR; gamma_thresh = gth, ae_band = true,
                             bracket = (fe[i1-1], fe[i1]), use_gpu = use_gpu)
        f1 = rm.factor
        evals += rm.evals
    end
    f2 = i2 < n_eig ? fe[i2+1] : fe[i2]   # pad to the next (stable) sample so the window brackets the bump
    return (; f1 = f1, f2 = f2, evals = evals, unstable = true, pinned_lo = pinned_lo)
end

"""
    marginal_factor_faithful(inputsEP, inputsPR; kwargs...) -> NamedTuple

Ground-truth marginal EP scale factor at a fixed `(IR, kyhat, width)` point that
matches the production keep definition (all secondary IFLUX=true filters), using
cheap AD eigensolves to localize the search and full evals only where needed.

Strategy:
  1. Cheap eigenvalue-only scan (`IFLUX=false`) maps the AE-band UNSTABLE hull
     `[f1,f2]` where `γ_AE(f) > γ*`, with `f1` Newton-refined. Because the keep
     window is a subset of this hull (keep ⇒ `g>γ*`), and `γ_AE(f)` is typically
     a narrow bump, restricting the expensive search to `[f1,f2]` is both
     complete and immune to stepping over the window.
  2. Faithful keep sweep (`IFLUX=true`) on a fine grid within `[f1,f2]`: the
     lowest kept sample is bisected against its unkept neighbor to give the lower
     edge of the kept window. The condition that fails just below that edge names
     the binding filter (band-entry/growth for DIII-D; ion/thermal pinch or
     QL-ratio for the rotational ITER case).

Keywords: `kyhat`, `width` (default to the struct's), `gamma_thresh=nothing`
(case γ*), `scan_lo=1e-3`, `scan_hi=10.0`, `n_eig=24` (eigenvalue scan points),
`n_fine=28` (faithful samples within the hull), `xtol=1e-4`, `maxbisect=30`,
`use_gpu=false`, `verbose=false`.

Returns `(; factor_faithful, factor_fast, window, binding, kept_modes,
evals_eig, evals_full, converged, keep)`. `factor_fast` is the cheap AD instability
onset `f1`; `window=(f1,f2)`. `binding` ∈ {`:ae_band_growth`, `:tearing`,
`:i_pinch`, `:e_pinch`, `:th_pinch`, `:ep_pinch`, `:ql_ratio`, `:theta_sq`,
`:none`}.
"""
function marginal_factor_faithful(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                  kyhat::Float64 = Float64(inputsEP.KYHAT_IN),
                                  width::Float64 = Float64(inputsEP.WIDTH_IN),
                                  gamma_thresh::Union{Nothing,Float64} = nothing,
                                  scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0,
                                  n_eig::Int = 24, n_fine::Int = 28,
                                  xtol::Float64 = 1.0e-4, maxbisect::Int = 30,
                                  threaded::Bool = true,
                                  inner::Symbol = :threads,
                                  team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                  use_gpu::Bool = false, verbose::Bool = false)
    ep0 = deepcopy(inputsEP); ep0.KYHAT_IN = kyhat; ep0.WIDTH_IN = width
    gth = gamma_thresh === nothing ? _gamma_thresh_for(ep0, inputsPR) : gamma_thresh

    # ── (1) cheap eigenvalue scan → AE-band unstable hull [f1,f2] ──
    win = _ae_unstable_window(ep0, inputsPR, gth; scan_lo = scan_lo, scan_hi = scan_hi,
                              n_eig = n_eig, threaded = threaded, use_gpu = use_gpu,
                              inner = inner, team = team)
    evals_eig = win.evals
    if !win.unstable
        verbose && @info "marginal_factor_faithful: AE-band γ never reaches γ* in [$scan_lo,$scan_hi]" gth
        return (; factor_faithful = NaN, factor_fast = NaN, window = (NaN, NaN),
                binding = :none, kept_modes = Int[], evals_eig, evals_full = 0,
                converged = false, keep = nothing, pinned_lo = false)
    end
    f1, f2 = win.f1, win.f2
    win.pinned_lo && verbose && @warn "marginal_factor_faithful: AE band already unstable at scan_lo=$scan_lo; the instability onset is below the scan range (true lower edge not bracketed)"
    verbose && @info "AE-band unstable hull" f1 f2 gth pinned_lo=win.pinned_lo

    nfull = Ref(0)
    keepf = f -> (nfull[] += 1; keep_at(ep0, inputsPR, f; kyhat = kyhat, width = width, use_gpu = use_gpu))

    # ── (2) faithful sweep within [f1,f2]: lowest kept sample + its unkept neighbor.
    #        Threaded: evaluate all n_fine samples in parallel (the keep window can sit
    #        anywhere in the hull), then locate the lowest kept index. ──
    fs = collect(range(f1, f2; length = n_fine))
    a = NaN; b = NaN; kept_b = nothing
    if threaded
        kfs = _ad_pmap(i -> keep_at(ep0, inputsPR, fs[i]; kyhat = kyhat, width = width, use_gpu = use_gpu),
                       n_fine; inner = inner, team = team)
        nfull[] += n_fine
        j = findfirst(kf -> kf.any_keep, kfs)
        if j !== nothing
            b = fs[j]; kept_b = kfs[j]
            a = j > 1 ? fs[j-1] : f1
        end
    else
        prev_f = NaN
        for (i, f) in enumerate(fs)
            kf = keepf(f)
            if kf.any_keep
                b = f; kept_b = kf
                a = i > 1 ? prev_f : f1
                break
            end
            prev_f = f
        end
    end
    if isnan(b)
        verbose && @info "marginal_factor_faithful: AE band unstable but NO mode kept in [f1,f2] (all rejected by secondary filters)"
        return (; factor_faithful = NaN, factor_fast = f1, window = (f1, f2),
                binding = :none, kept_modes = Int[], evals_eig, evals_full = nfull[],
                converged = false, keep = nothing, pinned_lo = win.pinned_lo)
    end

    # bisect [a (unkept), b (kept)] on any(LKEEP) for the lower edge
    fa, fb = a, b
    iters = 0
    while (fb - fa) > xtol && iters < maxbisect
        iters += 1
        fm = 0.5 * (fa + fb)
        kf = keepf(fm)
        if kf.any_keep
            fb = fm; kept_b = kf
        else
            fa = fm
        end
    end

    # binding condition = what fails on the unkept side just below the edge
    binding = :ae_band_growth
    ka = keepf(fa)
    for n in findall(kept_b.LKEEP)
        r = _why_rejected(ka, n)
        if r != :kept
            binding = (r == :band || r == :below_thresh) ? :ae_band_growth : r
            break
        end
    end

    return (; factor_faithful = fb, factor_fast = f1, window = (f1, f2), binding = binding,
            kept_modes = findall(kept_b.LKEEP), evals_eig, evals_full = nfull[],
            converged = true, keep = kept_b, pinned_lo = win.pinned_lo)
end

# Per-point marginal factor f★(ky,w) (AE-band growth root γ_AE=γ*) AND its exact
# gradient ∂f★/∂(ky,w) by the implicit function theorem. Differentiating
# γ_AE(f★(ky,w), ky, w) = γ* gives ∂f★/∂ky = -(∂γ/∂ky)/(∂γ/∂f) and likewise for w,
# with all three partials read from a SINGLE gamma_grad pass at (f★,ky,w). The
# binding mode is the AE-band leading mode (max γ among freq<FREQ_AE_UPPER) that
# defines the root. Returns `(; f, converged, grad, evals)`.
function _marginal_and_grad(ep0::Options{Float64}, inputsPR::profile{Float64}, ky::Float64, w::Float64;
                            gth::Float64, scan_lo::Float64, scan_hi::Float64, nscan::Int,
                            f_start::Union{Nothing,Float64} = nothing, use_gpu::Bool)
    ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
    mf = marginal_factor(ep, inputsPR; gamma_thresh = gth, ae_band = true, f_start = f_start,
                         scan_lo = scan_lo, scan_hi = scan_hi, nscan = nscan, use_gpu = use_gpu)
    !mf.converged && return (; f = Inf, converged = false, grad = (0.0, 0.0), evals = mf.evals)
    ep.FACTOR_IN = mf.factor
    fu = _ae_band_upper(ep, inputsPR)
    gg = gamma_grad(ep, inputsPR; vars = (:FACTOR_IN, :KYHAT_IN, :WIDTH_IN), use_gpu = use_gpu)
    cand = findall(<(fu), gg.freq)
    isempty(cand) && return (; f = mf.factor, converged = false, grad = (0.0, 0.0), evals = mf.evals + 1)
    j = cand[argmax(@view gg.gamma[cand])]
    dgdf, dgdky, dgdw = gg.dgamma[j, 1], gg.dgamma[j, 2], gg.dgamma[j, 3]
    if abs(dgdf) < 1e-30
        return (; f = mf.factor, converged = true, grad = (0.0, 0.0), evals = mf.evals + 1)
    end
    return (; f = mf.factor, converged = true, grad = (-dgdky / dgdf, -dgdw / dgdf), evals = mf.evals + 1)
end

_clamp_to(x, lo, hi) = min(max(x, lo), hi)

"""
    critical_factor_optimize(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD analogue of the inner `kwscale_scan`: find the critical EP scale factor
`sfmin = min over (kyhat,width) of the marginal factor f★(kyhat,width)` by
**projected-gradient descent with implicit-function-theorem gradients**.

Each evaluated `(ky,w)` solves the AE-band growth root `f★` by AD-Newton
([`marginal_factor`](@ref)) and gets the exact `∂f★/∂(ky,w)` from one extra
`gamma_grad` pass (the implicit-function-theorem ratio `-(∂γ/∂x)/(∂γ/∂f)`). A
coarse `nseed_ky × nseed_w` seed grid (or a caller-supplied `seed`) locates a
feasible basin; descent then polishes it with backtracking line search,
projecting `(ky,w)` to `ky∈ky_bounds`, `w∈[WIDTH_MIN,WIDTH_MAX]`.

NOTE on cost: this yields the *continuous* optimum (more precise than any fixed
grid), but it is NOT necessarily cheaper than a `(ky,w)` grid scan — the AE-band
onset surface `f★(ky,w)` is bumpy/multimodal (the same narrow-bump physics the
keep window shows in factor), so descent can need comparably many TGLF solves and
warm seeds can land in a worse basin. The large AD win is per-evaluation
(eigenvalue-only `IFLUX=false` Newton vs the production `IFLUX=true` factor grid),
realized in [`marginal_factor`](@ref)/[`marginal_factor_faithful`](@ref), not in
collapsing the `(ky,w)` grid.

Keywords: `gamma_thresh=nothing` (case γ*), `nseed_ky=4`, `nseed_w=4`,
`scan_lo=1e-3`, `scan_hi=nothing`(→FACTOR_IN), `nscan=8`, `ky_bounds=(1e-3,1.0)`,
`w_bounds=nothing`(→struct), `maxiter=25`, `gtol=1e-4`, `xtol=1e-5`,
`step0=0.25`, `nbacktrack=12`, `faithful_confirm=false`, `use_gpu=false`,
`verbose=false`.

Returns `(; sfmin, kyhat, width, f_seedmin, ky_seed, w_seed, iters, evals,
converged, faithful)` where `faithful` is the [`marginal_factor_faithful`](@ref)
result at the optimum (or `nothing`).
"""
function critical_factor_optimize(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                  gamma_thresh::Union{Nothing,Float64} = nothing,
                                  seed::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                                  nseed_ky::Int = 4, nseed_w::Int = 4,
                                  scan_lo::Float64 = 1.0e-3, scan_hi::Union{Nothing,Float64} = nothing,
                                  nscan::Int = 8,
                                  ky_bounds::Tuple{Float64,Float64} = (1.0e-3, 1.0),
                                  w_bounds::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                                  maxiter::Int = 20, gtol::Float64 = 1.0e-4, xtol::Float64 = 1.0e-5,
                                  step0::Float64 = 0.25, nbacktrack::Int = 8,
                                  faithful_confirm::Bool = false,
                                  inner::Symbol = :threads,
                                  team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                  use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi
    wlo, whi = w_bounds === nothing ? (Float64(inputsEP.WIDTH_MIN), Float64(inputsEP.WIDTH_MAX)) : w_bounds
    kylo, kyhi = ky_bounds

    evals = Ref(0)
    fval = function (ky, w; f_start = nothing)
        ep = deepcopy(inputsEP); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        mf = marginal_factor(ep, inputsPR; gamma_thresh = gth, ae_band = true, f_start = f_start,
                             scan_lo = scan_lo, scan_hi = shi, nscan = nscan, use_gpu = use_gpu)
        evals[] += mf.evals
        mf.converged ? mf.factor : Inf
    end
    fgrad = function (ky, w; f_start = nothing)
        r = _marginal_and_grad(inputsEP, inputsPR, ky, w; gth = gth, scan_lo = scan_lo,
                               scan_hi = shi, nscan = nscan, f_start = f_start, use_gpu = use_gpu)
        evals[] += r.evals
        r
    end

    # ── seed: warm start from a caller-provided (ky,w) (e.g. neighbor radius), or
    #    a coarse grid (mirrors kwscale_scan's first round). The grid is the robust
    #    fallback if the warm seed is infeasible. ──
    best_f = Inf; best_ky = NaN; best_w = NaN
    if seed !== nothing
        sky = _clamp_to(seed[1], kylo, kyhi); sw = _clamp_to(seed[2], wlo, whi)
        f = fval(sky, sw)
        if isfinite(f)
            best_f = f; best_ky = sky; best_w = sw
        end
    end
    if !isfinite(best_f)
        kys = [kylo + (kyhi - kylo) * (i - 0.5) / nseed_ky for i in 1:nseed_ky]
        ws  = nseed_w == 1 ? [0.5 * (wlo + whi)] : [wlo + (whi - wlo) * (i - 1) / (nseed_w - 1) for i in 1:nseed_w]
        seedpts = [(ky, w) for ky in kys for w in ws]
        # Each seed solve is independent → saturate the team/threads. The eval-counting
        # `fval` closure mutates a coordinator-local Ref that remote workers can't update,
        # so the seed solver returns (factor, n_evals) and we reduce on the coordinator.
        seedsolve = function (idx)
            ky, w = seedpts[idx]
            ep = deepcopy(inputsEP); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
            mf = marginal_factor(ep, inputsPR; gamma_thresh = gth, ae_band = true,
                                 scan_lo = scan_lo, scan_hi = shi, nscan = nscan, use_gpu = use_gpu)
            (mf.converged ? mf.factor : Inf, mf.evals)
        end
        seedres = _ad_pmap(seedsolve, length(seedpts); inner = inner, team = team)
        for (idx, fr) in enumerate(seedres)
            f, ne = fr
            evals[] += ne
            if f < best_f
                best_f = f; best_ky = seedpts[idx][1]; best_w = seedpts[idx][2]
            end
        end
    end
    f_seedmin = best_f; ky_seed = best_ky; w_seed = best_w
    if !isfinite(best_f)
        verbose && @warn "critical_factor_optimize: no feasible (instability-bearing) seed"
        return (; sfmin = Inf, kyhat = NaN, width = NaN, f_seedmin, ky_seed, w_seed,
                iters = 0, evals = evals[], converged = false, faithful = nothing)
    end

    # ── projected-gradient descent with backtracking line search ──
    ky, w = best_ky, best_w
    f = best_f
    iters = 0
    converged = false
    for it in 1:maxiter
        iters = it
        r = fgrad(ky, w; f_start = isfinite(f) ? f : nothing)
        (!r.converged || !isfinite(r.f)) && break
        gky, gw = r.grad
        # scale step by the box so ky and w move comparably
        sky = (kyhi - kylo); sw = (whi - wlo)
        gnorm = sqrt((gky * sky)^2 + (gw * sw)^2)
        gnorm < gtol && (converged = true; break)
        α = step0
        improved = false
        f_cur = r.f
        for _ in 1:nbacktrack
            dky = -α * gky * sky^2 / max(gnorm, 1e-30)
            dw  = -α * gw  * sw^2  / max(gnorm, 1e-30)
            ky_t = _clamp_to(ky + dky, kylo, kyhi)
            w_t  = _clamp_to(w  + dw,  wlo, whi)
            # warm-start the trial solve from the first-order IFT prediction of f★
            f_pred = f_cur + gky * (ky_t - ky) + gw * (w_t - w)
            f_t = fval(ky_t, w_t; f_start = isfinite(f_pred) && f_pred > 0 ? f_pred : nothing)
            if f_t < f_cur - 1e-10
                if abs(ky_t - ky) + abs(w_t - w) < xtol
                    ky, w, f = ky_t, w_t, f_t; improved = true; converged = true
                    break
                end
                ky, w, f = ky_t, w_t, f_t; improved = true
                break
            end
            α *= 0.5
        end
        verbose && @info "cfo" iter=it ky=ky w=w f=f grad=(gky, gw) gnorm=gnorm improved=improved
        if !improved
            converged = true   # no descent direction within the box → local min
            break
        end
    end

    faithful = nothing
    if faithful_confirm
        faithful = marginal_factor_faithful(inputsEP, inputsPR; kyhat = ky, width = w,
                                            scan_lo = scan_lo, scan_hi = shi, use_gpu = use_gpu,
                                            inner = inner, team = team)
    end

    return (; sfmin = f, kyhat = ky, width = w, f_seedmin, ky_seed, w_seed,
            iters = iters, evals = evals[], converged = converged, faithful = faithful)
end

"""
    critical_factor_faithful_grid(inputsEP, inputsPR; kwargs...) -> NamedTuple

Robust, grid-faithful critical EP scale factor: the **global minimum over a
`(kyhat, width)` grid of the FAITHFUL (all-filter) marginal factor**, mirroring
exactly what the Fortran/`grid` `kwscale_scan` reduces to (`sfmin = min over
(kyhat,width) of the kept-window lower edge`). Unlike [`critical_factor_optimize`](@ref),
which gradient-descends the *bumpy/multimodal* AE-band onset surface `f★(ky,w)`
and can land in a shallower local basin (or on a mode the keep filters later
reject), this evaluates [`marginal_factor_faithful`](@ref) at **every** grid point
and takes the global min — so it reproduces the grid `sfmin` to within the
continuous-vs-discrete-factor difference, at the cost of scanning the full
`(ky,w)` grid (no `(ky,w)` collapse). The inner *factor* root is still the cheap
AD-Newton + faithful-bisect (`IFLUX=false` hull localization, `IFLUX=true` only
inside the AE band), so each grid point is far cheaper than the production
8-factor × 4-round brute force.

The `(kyhat, width)` grid mirrors `kwscale_scan`'s first round:
  - `kyhat[i] = (1/nkyhat)·i`, i=1..nkyhat
  - `width[i] = WIDTH_MIN + (WIDTH_MAX-WIDTH_MIN)/(nefwid-1)·(i-1)`

Keywords: `nkyhat=4, nefwid=8`, `gamma_thresh=nothing` (case γ*), `scan_lo=1e-3`,
`scan_hi=nothing`(→`FACTOR_IN`), `inner=:threads`, `team=nothing`, `use_gpu=false`,
`verbose=false`. The `(ky,w)` points are parallelized over the team/threads; each
point's inner faithful sweep runs serially to avoid nested oversubscription.

Returns `(; sfmin, kyhat, width, binding, total_evals_full, total_evals_eig,
npts, results)`.
"""
function critical_factor_faithful_grid(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                       nkyhat::Int = 4, nefwid::Int = 8,
                                       refine_rounds::Int = 1,
                                       gamma_thresh::Union{Nothing,Float64} = nothing,
                                       scan_lo::Union{Nothing,Float64} = nothing, scan_hi::Union{Nothing,Float64} = nothing,
                                       inner::Symbol = :threads,
                                       team::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                       use_gpu::Bool = false, verbose::Bool = false)
    gth = gamma_thresh === nothing ? _gamma_thresh_for(inputsEP, inputsPR) : gamma_thresh
    shi = scan_hi === nothing ? Float64(inputsEP.FACTOR_IN) : scan_hi
    # Default scan_lo to the GRID's refinement floor so AD floors the same way the Fortran
    # kwscale_scan does: its k_max=4 rounds of nfactor=8-point bracketing (f1←2·fmark, f0←0)
    # bottom out at FACTOR_IN/(nfactor·4^(k_max-1)) = FACTOR_IN/512. A (ky,w) that is unstable
    # to arbitrarily small factor is reported by the grid at that floor (a discretization
    # artifact), so using the same floor keeps AD comparable instead of resolving below it.
    slo = scan_lo === nothing ? shi / 512.0 : scan_lo
    wlo = Float64(inputsEP.WIDTH_MIN); whi = Float64(inputsEP.WIDTH_MAX)
    kylo = 0.0; kyhi = 1.0

    total_full = 0; total_eig = 0; npts_total = 0
    all_res = Any[]
    best_f = Inf; best_ky = NaN; best_w = NaN; best_bind = :none

    # Evaluate the faithful onset on a (kyhat, width) grid spanning [kya,kyb]×[wa,wb] and
    # fold the global min into the running best. Parallelize the points over the team/
    # threads; each point's inner faithful sweep runs serially (`threaded=false`) to avoid
    # nesting Threads.@threads / re-spawning the team.
    function eval_grid!(kya, kyb, wa, wb)
        kyhats = nkyhat == 1 ? [0.5 * (kya + kyb)] :
                 (kya <= 0.0 ? [(kyb / nkyhat) * i for i in 1:nkyhat] :
                  [kya + (kyb - kya) / (nkyhat - 1) * (i - 1) for i in 1:nkyhat])
        widths = nefwid == 1 ? [0.5 * (wa + wb)] : [wa + (wb - wa) / (nefwid - 1) * (i - 1) for i in 1:nefwid]
        pts = [(ky, w) for ky in kyhats for w in widths]
        np = length(pts)
        res = _ad_pmap(idx -> begin
                ky, w = pts[idx]
                marginal_factor_faithful(inputsEP, inputsPR; kyhat = ky, width = w,
                                         gamma_thresh = gth, scan_lo = slo, scan_hi = shi,
                                         threaded = false, use_gpu = use_gpu)
            end, np; inner = inner, team = team)
        for (idx, r) in enumerate(res)
            total_full += r.evals_full; total_eig += r.evals_eig
            push!(all_res, r)
            if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
                best_f = r.factor_faithful; best_ky = pts[idx][1]; best_w = pts[idx][2]; best_bind = r.binding
            end
            verbose && @info "faithful grid pt" ky=pts[idx][1] w=pts[idx][2] factor=r.factor_faithful binding=r.binding
        end
        npts_total += np
        return nothing
    end

    # Coarse pass mirrors kwscale_scan's first round: kyhat∈(0,1], width∈[WIDTH_MIN,WIDTH_MAX].
    eval_grid!(kylo, kyhi, wlo, whi)

    # Refinement rounds mirror the Fortran (ky,w) window narrowing: re-grid a shrinking box
    # around the current best (ky,w) so the binding point can be resolved off the coarse
    # nodes (the grid path refines (ky,w) across its k rounds; a fixed coarse grid otherwise
    # overestimates radii whose optimum lies between nodes, e.g. the plasma edge).
    dky = nkyhat > 1 ? (kyhi - kylo) / (nkyhat - 1) : (kyhi - kylo)
    dw  = nefwid > 1 ? (whi - wlo) / (nefwid - 1) : (whi - wlo)
    for _ in 1:max(0, refine_rounds)
        (best_bind === :none || !isfinite(best_f)) && break   # nothing to refine around
        kya = max(kylo, best_ky - dky); kyb = min(kyhi, best_ky + dky)
        wa  = max(wlo,  best_w  - dw);  wb  = min(whi,  best_w  + dw)
        eval_grid!(kya, kyb, wa, wb)
        dky *= 2.0 / max(nkyhat - 1, 1)   # window already spans ±dky; tighten node spacing
        dw  *= 2.0 / max(nefwid - 1, 1)
    end

    # status: :ok (genuine interior onset), :no_onset (no (ky,w) had a kept window in
    # [slo,shi]), or :cap (best onset sits at the search ceiling → likely no real onset
    # below FACTOR_IN). Callers can fall back to the grid solver on :no_onset/:cap.
    status = (best_bind === :none || !isfinite(best_f)) ? :no_onset :
             (best_f >= 0.999 * shi ? :cap : :ok)

    return (; sfmin = best_f, kyhat = best_ky, width = best_w, binding = best_bind,
            status = status, scan_lo = slo, scan_hi = shi,
            total_evals_full = total_full, total_evals_eig = total_eig, npts = npts_total, results = all_res)
end

"""
    critical_factor_profile(inputsEP, inputsPR; kwargs...) -> NamedTuple

AD analogue of the radial driver loop (`tjlfep_driver` / `mainsub` over the
`SCAN_N` radii): the critical EP scale factor `sfmin(IR)` versus radius, computed
by [`critical_factor_optimize`](@ref) at each radius with **cross-radius
continuation** — the optimum `(kyhat,width)` of one radius warm-starts the next.
(Continuation skips the per-radius seed grid, but on the bumpy `f★(ky,w)` surface
it does not reliably reduce total solves and can land in a worse local basin; see
the cost note on `critical_factor_optimize`. It is most useful when the optimum
moves smoothly with radius.)

Keywords: `radii=nothing` (→ `Int.(IR_EXP[1:SCAN_N])`), `scan_lo=1e-3`,
`scan_hi=10.0`, `faithful=false` (confirm each optimum with
[`marginal_factor_faithful`](@ref)), `use_gpu=false`, `verbose=false`; remaining
kwargs forwarded to `critical_factor_optimize`.

Returns `(; radii, sfmin, kyhat, width, evals, converged, binding, total_evals)`.
When `faithful=true`, `sfmin` holds the faithful (all-filter) onset and `binding`
the per-radius binding condition; otherwise `sfmin` is the AE-band onset.
"""
function critical_factor_profile(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                 radii::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                 scan_lo::Float64 = 1.0e-3, scan_hi::Float64 = 10.0,
                                 faithful::Bool = false,
                                 use_gpu::Bool = false, verbose::Bool = false, kwargs...)
    rs = radii === nothing ? Int.(inputsEP.IR_EXP[1:inputsEP.SCAN_N]) : collect(radii)
    n = length(rs)
    sfmin = fill(Inf, n); kyhat = fill(NaN, n); width = fill(NaN, n)
    evals = zeros(Int, n); conv = falses(n); binding = fill(:none, n)
    prev_seed = nothing
    for (i, ir) in enumerate(rs)
        ep = deepcopy(inputsEP); ep.IR = ir
        r = critical_factor_optimize(ep, inputsPR; seed = prev_seed, scan_lo = scan_lo, scan_hi = scan_hi,
                                     faithful_confirm = faithful, use_gpu = use_gpu, verbose = verbose, kwargs...)
        if faithful && r.faithful !== nothing
            sfmin[i] = r.faithful.factor_faithful
            binding[i] = r.faithful.binding
        else
            sfmin[i] = r.sfmin
        end
        kyhat[i] = r.kyhat; width[i] = r.width; evals[i] = r.evals; conv[i] = r.converged
        verbose && @info "radius" i ir sfmin=sfmin[i] kyhat=kyhat[i] width=width[i] evals=evals[i] converged=conv[i] binding=binding[i]
        # continuation: carry the optimum forward only if this radius converged
        if r.converged && isfinite(r.sfmin)
            prev_seed = (r.kyhat, r.width)
        end
    end
    return (; radii = rs, sfmin, kyhat, width, evals, converged = conv, binding, total_evals = sum(evals))
end

# Default sensitivity knobs for a TGLF-EP operating point: the per-species inverse
# scale lengths (density a/Ln = RLNS, temperature a/LT = RLTS) and densities (AS),
# plus the local safety factor Q_LOC — the standard AE drive/damping parameters.
# Species are labeled e/i1/i2/.../EP; the EP species (inputsEP.IS_EP) is the
# primary Alfvén-eigenmode drive.
function default_sensitivity_knobs(inputF)
    ns = inputF.NS
    knobs = Tuple{Symbol,Union{Int,Nothing}}[]
    for is in 1:ns
        push!(knobs, (:RLNS, is))
        push!(knobs, (:RLTS, is))
    end
    for is in 1:ns
        push!(knobs, (:AS, is))
    end
    push!(knobs, (:Q_LOC, nothing))
    return knobs
end

# `ep_slot` is the EP species SLOT in InputTJLF, i.e. IS_EP+1 (TJLF_map sets
# `is = inputsEP.IS_EP + 1`), not IS_EP itself. Slot 1 is the electron; the
# remaining ions are labeled i1,i2,… skipping the EP slot.
function _knob_label(fld::Symbol, idx, ep_slot::Int)
    idx === nothing && return string(fld)
    tag = idx == 1 ? "e" : (idx == ep_slot ? "EP" : "i$(idx-1)")
    return "$(fld)[$tag]"
end

"""
    gamma_input_sensitivities(inputsEP, inputsPR; knobs=nothing, use_gpu=false) -> NamedTuple

AD-exact sensitivities of the per-mode growth rate `γ` (and real frequency) to the
local TGLF plasma inputs, at a **fixed** operating point `(IR, KYHAT_IN, WIDTH_IN,
FACTOR_IN)`. This is the genuinely differentiable, grid-independent quantity the
Fortran TGLF-EP cannot provide — unlike the critical factor `sfmin`, which sits at
a discrete keep-flag transition and is not smoothly differentiable.

The growth rate is differentiated through TGLF's eigensolve in a single forward
pass by seeding multi-partial `ForwardDiff.Dual`s on the selected inputs of the
`InputTJLF` produced by `TJLF_map` (so the eigenvalue rule in TJLF's
`TJLFForwardDiffExt` provides exact `∂λ/∂input`). Uses `IFLUX=false` (eigenvalues
only), so it is cheap.

`knobs` is a vector of `(field::Symbol, idx)` pairs; `idx::Int` selects a species
entry of a vector field (`:RLNS,:RLTS,:TAUS,:AS,:VPAR,:VPAR_SHEAR`) and `idx=nothing`
a scalar field (`:Q_LOC,:Q_PRIME_LOC,:RMIN_LOC,...`). Defaults to per-species
a/Ln, a/LT, density, plus Q_LOC.

Returns `(; gamma, freq, dgamma, dfreq, logsens, knobs, labels, base)` where
`dgamma`/`dfreq` are `NMODES × Nknob` Jacobians, `base` the knob base values, and
`logsens[m,j] = base[j]·∂γ_m/∂knob_j` the dimensionless (logarithmic) sensitivity
used to rank drive (>0) vs damping (<0) terms.
"""
function gamma_input_sensitivities(inputsEP::Options{Float64}, inputsPR::profile{Float64};
                                   knobs::Union{Nothing,AbstractVector} = nothing,
                                   use_gpu::Bool = false)
    inputF = TJLF_map(inputsEP, inputsPR)
    inputF isa Integer && error("gamma_input_sensitivities: TJLF_map rejected IR=$(inputsEP.IR)")
    _configure_inputTJLF_for_ky!(inputF, inputsEP)

    ks = knobs === nothing ? default_sensitivity_knobs(inputF) : knobs
    N = length(ks)
    # EP species slot in InputTJLF is IS_EP+1 (see TJLF_map: is = IS_EP + 1); 0 if unset.
    ep_slot = inputsEP.IS_EP === missing ? 0 : Int(inputsEP.IS_EP) + 1

    Tag = typeof(ForwardDiff.Tag(gamma_input_sensitivities, Float64))
    D   = ForwardDiff.Dual{Tag, Float64, N}
    inputD = _to_dual_input(inputF, D)

    base = zeros(Float64, N)
    labels = String[]
    for (i, (fld, idx)) in enumerate(ks)
        v = idx === nothing ? getfield(inputF, fld) : getfield(inputF, fld)[idx]
        base[i] = Float64(v)
        seed = ForwardDiff.Dual{Tag}(base[i], ntuple(j -> j == i ? 1.0 : 0.0, N)...)
        if idx === nothing
            setfield!(inputD, fld, seed)
        else
            getfield(inputD, fld)[idx] = seed
        end
        push!(labels, _knob_label(fld, idx, ep_slot))
    end

    res = _tjlf_run_dual(inputD, D; use_gpu = use_gpu)
    g = res.eigenvalue[:, 1, 1]
    f = res.eigenvalue[:, 1, 2]
    NM = length(g)

    gamma = ForwardDiff.value.(g)
    freq  = ForwardDiff.value.(f)
    dgamma = [ForwardDiff.partials(g[m], i) for m in 1:NM, i in 1:N]
    dfreq  = [ForwardDiff.partials(f[m], i) for m in 1:NM, i in 1:N]
    logsens = [base[i] * dgamma[m, i] for m in 1:NM, i in 1:N]

    return (; gamma, freq, dgamma, dfreq, logsens, knobs = ks, labels, base)
end
