# Diagnose why the AD production scan (critical_factor_optimize, faithful_confirm) overestimates
# sfmin at the strong-drive DIII-D radii (ir=33,38,43) at N_BASIS=32 relative to the grid/Fortran
# kwscale_scan. Two suspects:
#   (1) coarse scan_hi = FACTOR_IN = 10  -> the AE-band geometric factor scan under-resolves the
#       narrow low-factor instability bump (the single-point faithful benchmark used scan_hi=0.3
#       and recovered 0.0171 at ir=38, vs the production scan's 0.0550).
#   (2) (ky,w) basin-miss -> the AE-band seed+descent lands on a shallower local onset than the
#       grid's global (ky,w) argmin.
#
# For each radius we run critical_factor_optimize (faithful_confirm=true, use_gpu) in several
# configs and print the faithful onset + (ky,w) it converges to, against the known grid and
# production-AD values. Run JIT on a GPU node (no sysimage) to pick up current source.

using CUDA
using TJLF
using TJLFEP
using Printf

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const TGLFEP = joinpath(CASE, "input_scan20_nb32.TGLFEP")

# (grid sfmin, production-AD sfmin) per radius, read off the scan20 sfmin_scan.txt files.
const REF = Dict(
    22 => (grid = 0.175775, ad_prod = 0.174972),   # control: AD already matches grid
    33 => (grid = 0.117178, ad_prod = 0.727104),   # outlier +520%
    38 => (grid = 0.019530, ad_prod = 0.055015),   # outlier +182%
    43 => (grid = 0.019531, ad_prod = 0.066813),   # outlier +242%
)

function run_cfg(ep0, prof, ir; scan_hi, nseed_ky=4, nseed_w=4, seed=nothing)
    ep = deepcopy(ep0); ep.IR = ir
    t = @elapsed res = critical_factor_optimize(ep, prof;
            scan_hi = scan_hi, nseed_ky = nseed_ky, nseed_w = nseed_w, seed = seed,
            faithful_confirm = true, use_gpu = true)
    fth = res.faithful
    sf  = (fth !== nothing && fth.binding != :none && isfinite(fth.factor_faithful)) ?
          fth.factor_faithful : res.sfmin
    bind = fth === nothing ? :none : fth.binding
    return (; sfmin = sf, ky = res.kyhat, w = res.width, binding = bind,
            ae = res.sfmin, evals = res.evals, conv = res.converged, wall = t)
end

function main()
    @assert CUDA.functional() "run on a GPU node"
    opts, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    opts.N_BASIS = 32
    @printf("DIII-D N_BASIS=32   GPU=%s\n", CUDA.name(first(CUDA.devices())))

    # warm up / compile
    run_cfg(opts, prof, 38; scan_hi = 0.3)

    for ir in (22, 33, 38, 43)
        r = REF[ir]
        @printf("\n========== IR=%d   grid=%.5g   AD_prod=%.5g (%+.0f%%) ==========\n",
                ir, r.grid, r.ad_prod, 100*(r.ad_prod-r.grid)/r.grid)
        cfgs = [
            ("prod (scan_hi=10)",        (; scan_hi = 10.0)),
            ("scan_hi=0.3",              (; scan_hi = 0.3)),
            ("scan_hi=1.0",             (; scan_hi = 1.0)),
            ("scan_hi=0.3 dense8x8",     (; scan_hi = 0.3, nseed_ky = 8, nseed_w = 8)),
        ]
        for (name, kw) in cfgs
            c = run_cfg(opts, prof, ir; kw...)
            @printf("  %-22s sfmin=%.5g  (ky=%.4f w=%.4f)  bind=%-14s  vs grid %+7.1f%%  [evals=%d %.1fs]\n",
                    name, c.sfmin, c.ky, c.w, String(c.binding),
                    100*(c.sfmin - r.grid)/r.grid, c.evals, c.wall)
        end
    end
    println("\n=== diag done ===")
end

main()
