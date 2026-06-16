# Threads-timed validation of the guarded :ad path (critical_factor_ad_guarded), the speed-keeping
# fix for :ad's local-minimum outliers. Compares, per radius:
#   ad      : critical_factor_optimize (single descent) + faithful confirm   — the fast baseline
#   ad-adapt: critical_factor_ad_guarded guard=:adaptive  (detect-and-reseed) + faithful confirm
#   ad-all  : critical_factor_ad_guarded guard=:always     (multistart all)   + faithful confirm
# All timed under :threads (the :ad-class optimum: short serial descent + small seed grid → MPS
# overhead doesn't pay; threads won for :ad and should win here). sfmin is the FAITHFUL keep onset.
#
# Accuracy is judged against fixed references measured by the DIRECT experiment (job 54490874):
#   GRID   = Fortran/grid production sfmin
#   DENSE  = 6x10 faithful continuous truth
#   DIRECT = NLopt GN_DIRECT_L global search (the most accurate; beat every fixed grid at IR=48)
# The key question: does guarded :ad recover DIRECT-class accuracy at IR=48 (the off-node spike,
# where the 4x8 grid sat +91% above dense) while keeping ~:ad wallclock?
#
# Env: USE_GPU (0/1, default 1), NB (default 32), RADII (csv, default 22,38,48,95).

using TJLF
using TJLFEP
using Printf

const USE_GPU = get(ENV, "USE_GPU", "1") == "1"
if USE_GPU
    using CUDA
    @assert CUDA.functional() "USE_GPU=1 but no functional GPU"
end

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const NB     = parse(Int, get(ENV, "NB", "32"))
const TGLFEP = joinpath(CASE, "input_scan20_nb$(NB).TGLFEP")
const RADII  = parse.(Int, split(get(ENV, "RADII", "22,38,48,95"), ',', keepempty=false))

# References from the DIRECT experiment (job 54490874), nb=32. (IR=95 filled after that run lands.)
const GRID   = Dict(22 => 0.17577, 38 => 0.01953, 48 => 0.039062, 95 => 2.636713)
const DENSE  = Dict(22 => 0.16246, 38 => 0.019531, 48 => 0.028553, 95 => 4.2007)
const DIRECT = Dict(22 => 0.16476, 38 => 0.019531, 48 => 0.026728, 95 => 1.3812)

pct(x, ref) = (isfinite(x) && isfinite(ref) && ref != 0) ? @sprintf("%+6.1f%%", 100*(x-ref)/ref) : "   n/a "

# faithful sfmin from a critical_factor_optimize result (AE-onset min, then keep-confirmed)
ad_faithful(r) = (r.faithful !== nothing && r.faithful.binding !== :none &&
                  isfinite(r.faithful.factor_faithful)) ? r.faithful.factor_faithful : r.sfmin

function main()
    opts, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    opts.N_BASIS = NB
    dev = USE_GPU ? CUDA.name(first(CUDA.devices())) : "CPU"
    @printf("DIII-D N_BASIS=%d  %s  guarded :ad validation (inner=:threads)\n", NB, dev)
    flush(stdout)

    let ep = deepcopy(opts); ep.IR = 38
        TJLFEP.critical_factor_optimize(ep, prof; faithful_confirm=true, inner=:threads, use_gpu=USE_GPU)
        TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:adaptive, faithful_confirm=true, inner=:threads, use_gpu=USE_GPU)
        TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:always, faithful_confirm=true, inner=:threads, use_gpu=USE_GPU)
    end

    for ir in RADII
        ep = deepcopy(opts); ep.IR = ir
        g = get(GRID, ir, NaN); d = get(DENSE, ir, NaN); di = get(DIRECT, ir, NaN)

        t_ad = @elapsed r_ad = TJLFEP.critical_factor_optimize(ep, prof; faithful_confirm=true,
                inner=:threads, use_gpu=USE_GPU)
        t_ga = @elapsed r_ga = TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:adaptive,
                faithful_confirm=true, inner=:threads, use_gpu=USE_GPU)
        t_gl = @elapsed r_gl = TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:always,
                faithful_confirm=true, inner=:threads, use_gpu=USE_GPU)

        s_ad = ad_faithful(r_ad)
        @printf("\nIR=%-3d  grid=%-9.5g dense=%-9.5g direct=%-9.5g\n", ir, g, d, di)
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s vsdirect=%s  evals=%-5d %6.1fs  conv=%s\n",
                "ad", s_ad, pct(s_ad, g), pct(s_ad, d), pct(s_ad, di), r_ad.evals, t_ad, r_ad.converged)
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s vsdirect=%s  evals=%-5d %6.1fs  starts=%d flag=%s(edge=%s,amb=%s)\n",
                "ad-adapt", r_ga.sfmin, pct(r_ga.sfmin, g), pct(r_ga.sfmin, d), pct(r_ga.sfmin, di),
                r_ga.evals, t_ga, r_ga.n_starts, r_ga.flagged, r_ga.on_edge, r_ga.ambiguous)
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s vsdirect=%s  evals=%-5d %6.1fs  starts=%d\n",
                "ad-all", r_gl.sfmin, pct(r_gl.sfmin, g), pct(r_gl.sfmin, d), pct(r_gl.sfmin, di),
                r_gl.evals, t_gl, r_gl.n_starts)
        flush(stdout)
    end
    println("\n=== guarded experiment done ===")
end

main()
