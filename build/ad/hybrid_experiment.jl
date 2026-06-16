# Offline accuracy-per-eval experiment for the (ky,w) outer solver of critical_factor_robust.
#
# Question: can a coarse grid (global bracket) + a LOCAL AD-IFT polish (outer=:hybrid) recover
# the off-node accuracy of a full grid-zoom (refine_rounds=2) at the eval cost of refine=0?
#
# For each radius it runs four configurations of TJLFEP.critical_factor_robust and compares
# their sfmin and FAITHFUL eval count (total_evals_full, the dominant IFLUX=true cost):
#   refine0  : refine_rounds=0                      (coarse grid only — cheap, the speed target)
#   refine2  : refine_rounds=2, outer=:grid_zoom    (adaptive grid-zoom — the accuracy target)
#   hybrid   : refine_rounds=2, outer=:hybrid       (coarse bracket + gated AD-IFT polish + confirm)
#   dense    : nkyhat=DENSE_NKY × nefwid=DENSE_NW, refine_rounds=0  (continuous-truth global min)
# Accuracy is reported vs the Fortran/grid sfmin (GRID, the production answer) AND vs the dense
# faithful grid min (an independent continuous reference). The dense run is the most expensive
# and can be disabled with DENSE=0.
#
# Env: USE_GPU (0/1, default 0=CPU offline), NB (default 32), RADII (csv, default spikes+controls),
#      DENSE (0/1, default 1), DENSE_NKY (default 10), DENSE_NW (default 20).

using TJLF
using TJLFEP
using Printf

const USE_GPU = get(ENV, "USE_GPU", "0") == "1"
if USE_GPU
    using CUDA
    @assert CUDA.functional() "USE_GPU=1 but no functional GPU"
end

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const NB     = parse(Int, get(ENV, "NB", "32"))
const TGLFEP = joinpath(CASE, "input_scan20_nb$(NB).TGLFEP")
const DENSE  = get(ENV, "DENSE", "1") == "1"
const DENSE_NKY = parse(Int, get(ENV, "DENSE_NKY", "10"))
const DENSE_NW  = parse(Int, get(ENV, "DENSE_NW", "20"))
const RADII = let s = get(ENV, "RADII", "22,33,38,48,95")
    parse.(Int, split(s, ',', keepempty=false))
end

# Fortran/grid sfmin per radius (production answer) from the scan20 grid run sfmin_scan.txt.
const GRID = Dict(2 => 0.937419, 7 => 0.624945, 17 => 0.234367, 22 => 0.175775,
                  33 => 0.117178, 38 => 0.019530, 43 => 0.019531, 48 => 0.039062,
                  95 => 2.636713)

pct(x, ref) = (isfinite(x) && isfinite(ref) && ref != 0) ? @sprintf("%+6.1f%%", 100*(x-ref)/ref) : "   n/a "

function run_cfg(ep, prof; kw...)
    t = @elapsed r = TJLFEP.critical_factor_robust(ep, prof; nkyhat=4, nefwid=8,
            gamma_thresh=nothing, use_gpu=USE_GPU, kw...)
    (; r, t)
end

function main()
    opts, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    opts.N_BASIS = NB
    dev = USE_GPU ? CUDA.name(first(CUDA.devices())) : "CPU"
    @printf("DIII-D N_BASIS=%d  %s   dense-truth=%s (%dx%d)\n", NB, dev,
            DENSE ? "on" : "off", DENSE_NKY, DENSE_NW)
    flush(stdout)

    # warm up / compile every branch once on a cheap interior radius
    let ep = deepcopy(opts); ep.IR = 38
        TJLFEP.critical_factor_robust(ep, prof; refine_rounds=0, use_gpu=USE_GPU)
        TJLFEP.critical_factor_robust(ep, prof; refine_rounds=1, outer=:grid_zoom, use_gpu=USE_GPU)
        TJLFEP.critical_factor_robust(ep, prof; refine_rounds=1, outer=:hybrid, use_gpu=USE_GPU)
        TJLFEP.critical_factor_confirm(ep, prof; use_gpu=USE_GPU)
        DENSE && TJLFEP.critical_factor_robust(ep, prof; nkyhat=DENSE_NKY, nefwid=DENSE_NW,
                                               refine_rounds=0, use_gpu=USE_GPU)
    end

    for ir in RADII
        ep = deepcopy(opts); ep.IR = ir
        gref = get(GRID, ir, NaN)

        dtruth = NaN; dfull = 0
        if DENSE
            t = @elapsed rd = TJLFEP.critical_factor_robust(ep, prof; nkyhat=DENSE_NKY,
                    nefwid=DENSE_NW, refine_rounds=0, gamma_thresh=nothing, use_gpu=USE_GPU)
            dtruth = rd.sfmin; dfull = rd.total_evals_full
        end

        r0 = run_cfg(ep, prof; refine_rounds=0)
        r2 = run_cfg(ep, prof; refine_rounds=2, outer=:grid_zoom, adaptive=true)
        hy = run_cfg(ep, prof; refine_rounds=2, outer=:hybrid, adaptive=true)
        tc = @elapsed rc = TJLFEP.critical_factor_confirm(ep, prof; nkyhat=4, nefwid=8,
                gamma_thresh=nothing, use_gpu=USE_GPU)

        @printf("\nIR=%-3d  grid=%-10.5g  dense(%dx%d)=%-10.5g [full=%d]\n",
                ir, gref, DENSE_NKY, DENSE_NW, dtruth, dfull)
        for (tag, c) in (("refine0", r0), ("refine2", r2), ("hybrid ", hy))
            r = c.r
            @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  full=%-4d eig=%-5d  %5.1fs  %-14s rf=%d pol=%s st=%s\n",
                    tag, r.sfmin, pct(r.sfmin, gref), pct(r.sfmin, dtruth),
                    r.total_evals_full, r.total_evals_eig, c.t, String(r.binding),
                    r.refine_done, r.polished, String(r.status))
        end
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  full=%-4d eig=%-5d  %5.1fs  %-14s nconf=%d/%d cheap=%.4g st=%s\n",
                "confirm", rc.sfmin, pct(rc.sfmin, gref), pct(rc.sfmin, dtruth),
                rc.total_evals_full, rc.total_evals_eig, tc, String(rc.binding),
                rc.n_confirm, rc.npts, rc.cheap_min, String(rc.status))
        flush(stdout)
    end
    println("\n=== hybrid experiment done ===")
end

main()
