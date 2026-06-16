# Validate the hardened robust_ad reduction (TJLFEP.critical_factor_robust, scan_lo defaulting
# to the grid floor scan_hi/512, ADAPTIVE refinement with budget refine_rounds=2) against the
# Fortran/grid kwscale_scan sfmin at DIII-D N_BASIS=32. Covers the AD-threads spike radii
# (2,7,17,33,95), the strong-drive floor radii (38,43), and controls (22,48). Reports the number
# of zoom rounds the adaptive trigger actually spent (rf), coarse feasibility (feas), status,
# and inner-eval cost — so we can see refinement is spent only on the radii that need it. Run
# JIT on a GPU node.

using CUDA
using TJLF
using TJLFEP
using Printf

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const TGLFEP = joinpath(CASE, "input_scan20_nb32.TGLFEP")

# grid (Fortran-equivalent) sfmin per radius from the scan20 grid run sfmin_scan.txt
const GRID = Dict(2 => 0.937419, 7 => 0.624945, 17 => 0.234367, 22 => 0.175775,
                  33 => 0.117178, 38 => 0.019530, 43 => 0.019531, 48 => 0.039062,
                  95 => 2.636713)
# AD-threads production sfmin (the spikes we want to fix)
const ADTH = Dict(2 => 1.3407, 7 => 0.85103, 17 => 1.0648, 22 => 0.20917,
                  33 => 0.72710, 38 => 0.017844, 43 => 0.020903, 48 => 0.028360,
                  95 => 10.0)

function main()
    @assert CUDA.functional() "run on a GPU node"
    opts, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    opts.N_BASIS = 32
    @printf("DIII-D N_BASIS=32  GPU=%s   (scan_lo = scan_hi/512 grid floor)\n",
            CUDA.name(first(CUDA.devices())))

    let ep = deepcopy(opts); ep.IR = 38   # warm up / compile
        TJLFEP.critical_factor_robust(ep, prof; nkyhat = 4, nefwid = 8, refine_rounds = 2, use_gpu = true)
    end

    @printf("\n%-4s %10s %12s %14s %8s  %-14s %s\n",
            "IR", "grid", "AD_threads", "robust_ad", "vs grid", "binding/status", "[rf feas cost]")
    for ir in (2, 7, 17, 22, 33, 38, 43, 48, 95)
        ep = deepcopy(opts); ep.IR = ir
        t = @elapsed r = TJLFEP.critical_factor_robust(ep, prof;
                nkyhat = 4, nefwid = 8, refine_rounds = 2, adaptive = true, use_gpu = true)
        g = GRID[ir]
        @printf("%-4d %10.5g %12.5g %14.5g %+7.1f%%  %-14s [rf=%d feas=%d/%d full=%d %.0fs status=%s]\n",
                ir, g, ADTH[ir], r.sfmin, 100*(r.sfmin-g)/g, String(r.binding),
                r.refine_done, r.n_feasible_coarse, r.npts_coarse,
                r.total_evals_full, t, String(r.status))
    end
    println("\n=== validate robust_ad (adaptive, budget=2) done ===")
end

main()
