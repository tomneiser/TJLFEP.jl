# Pure-AD vs confirm-AD width-extended `:ad` comparison.
#
# The production width-extended `:ad` (faithful_confirm=true) faithful-confirms (IFLUX=true) the
# located narrow candidates, so its sfmin is the faithful keep onset and matches :robust_ad bitwise
# at the located mode. The "pure AD" variant (faithful_confirm=false) stays on the cheap AE-band
# onset surface end-to-end (NO IFLUX=true confirm anywhere) — much cheaper, but reports the AE-band
# onset (≤ the faithful keep onset) so it does NOT match :robust_ad bitwise and can dip below it.
#
# Over the 20-radius DIII-D scan we compare, on identical hardware:
#   confirm-ext :ad  (faithful_confirm=true, extend_width=true)   ← production
#   pure-ext    :ad  (faithful_confirm=false, extend_width=true)  ← this experiment
#   robust_ad        (extend_width=true)                          ← faithful reference
# recording sfmin, wallclock, and eigensolve counts for each.
#
#   module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#   USE_GPU=1 NB=8 julia --project=. build/ad/profile_ad_pure_vs_confirm.jl
# env: RADII (default all 20), NB (default 8), MODES=pure,confirm,robust (subset to skip robust)

using TJLF, TJLFEP, Printf

const USE_GPU = get(ENV, "USE_GPU", "0") == "1"
if USE_GPU
    using CUDA
    @assert CUDA.functional() "USE_GPU=1 but no functional GPU"
    CUDA.device!(first(CUDA.devices()))
end

const AX_CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const AX_GACODE = joinpath(AX_CASE, "input.gacode")
const AX_NB     = parse(Int, get(ENV, "NB", "8"))
const AX_MODES  = Set(split(get(ENV, "MODES", "pure,confirm,robust"), ','))

faithful_of(r) = (r.faithful !== nothing && r.faithful.binding != :none) ? r.faithful.factor_faithful : r.sfmin

function run_case(opts, prof, ir)
    ep = deepcopy(opts); ep.IR = ir
    gth = TJLFEP._gamma_thresh_for(ep, prof); shi = Float64(ep.FACTOR_IN); slo = shi / 512.0
    kw = (; gamma_thresh = gth, scan_lo = slo, scan_hi = shi, inner = :threads, team = nothing, use_gpu = USE_GPU)

    @printf("\n===== IR=%d  NB=%d  (FACTOR_IN=%.4g, WIDTH_MIN=%.3g) =====\n",
            ir, ep.N_BASIS, shi, Float64(ep.WIDTH_MIN)); flush(stdout)

    row = Dict{Symbol,Any}(:ir => ir)

    if "confirm" in AX_MODES
        t = @elapsed c = critical_factor_optimize(ep, prof; faithful_confirm = true, extend_width = true, kw...)
        cf = faithful_of(c)
        row[:confirm] = cf; row[:confirm_w] = c.width; row[:confirm_ev] = c.evals; row[:confirm_t] = t
        @printf("  confirm-ext :ad   sfmin=%.5g  at (ky=%.4f, w=%.4f)  evals=%d  n_ext=%d  %.1fs\n",
                cf, c.kyhat, c.width, c.evals, c.n_ext_confirm, t); flush(stdout)
    end
    if "pure" in AX_MODES
        t = @elapsed p = critical_factor_optimize(ep, prof; faithful_confirm = false, extend_width = true, kw...)
        row[:pure] = p.sfmin; row[:pure_w] = p.width; row[:pure_ev] = p.evals; row[:pure_t] = t
        @printf("  pure-ext :ad      sfmin=%.5g  at (ky=%.4f, w=%.4f)  evals=%d  (AE-band, no confirm)  %.1fs\n",
                p.sfmin, p.kyhat, p.width, p.evals, t); flush(stdout)
    end
    if "robust" in AX_MODES
        t = @elapsed rob = critical_factor_robust(ep, prof; extend_width = true, kw...)
        row[:robust] = rob.sfmin; row[:robust_w] = rob.width
        row[:robust_ev] = rob.total_evals_full + rob.total_evals_eig; row[:robust_t] = t
        @printf("  robust_ad (ref)   sfmin=%.5g  at (ky=%.4f, w=%.4f)  evals=%d  %.1fs\n",
                rob.sfmin, rob.kyhat, rob.width, row[:robust_ev], t); flush(stdout)
    end
    return row
end

function main()
    # IR list: default = the 20-radius scan grid for this case.
    radii_env = get(ENV, "RADII", "")
    @printf("pure vs confirm width-extended :ad (%s)  NB=%d  modes=%s  threads=%d\n",
            USE_GPU ? "GPU" : "CPU", AX_NB, join(sort(collect(AX_MODES)), ","), Threads.nthreads()); flush(stdout)
    tglfep = joinpath(AX_CASE, "input_scan20_nb$(AX_NB).TGLFEP")
    opts, prof, _ = preprocess_gacode_inputs(AX_GACODE, tglfep)
    opts.N_BASIS = AX_NB

    # Mirror the production scan grid: IR_EXP[1:SCAN_N] (falling back to ir_exp_from_scan when unset).
    ir_exp = (!isempty(opts.IR_EXP) && !all(iszero, opts.IR_EXP)) ? Int.(opts.IR_EXP) :
             TJLFEP.ir_exp_from_scan(prof.NR, prof.IRS, opts.SCAN_N)
    radii = isempty(radii_env) ? Int.(ir_exp[1:opts.SCAN_N]) : parse.(Int, split(radii_env, ','))
    @printf("radii (%d): %s\n", length(radii), string(radii)); flush(stdout)

    # Warm up compilation on the first radius so per-radius timings exclude JIT.
    run_case(opts, prof, first(radii))

    rows = Dict{Symbol,Any}[]
    for ir in radii
        push!(rows, run_case(opts, prof, ir))
    end

    println("\n===================== SUMMARY (sfmin) =====================")
    @printf("  %-5s %-12s %-12s %-12s %-10s %-10s\n", "IR", "confirm", "pure", "robust", "pure/conf", "pure/rob")
    for r in rows
        cf = get(r, :confirm, NaN); pu = get(r, :pure, NaN); ro = get(r, :robust, NaN)
        @printf("  %-5d %-12.5g %-12.5g %-12.5g %-10.3f %-10.3f\n",
                r[:ir], cf, pu, ro, pu / max(cf, eps()), pu / max(ro, eps()))
    end

    println("\n===================== SUMMARY (cost) =====================")
    @printf("  %-5s %-12s %-12s %-12s %-10s %-10s\n",
            "IR", "conf_t", "pure_t", "rob_t", "conf_ev", "pure_ev")
    tc = tp = tr = 0.0
    for r in rows
        ct = get(r, :confirm_t, 0.0); pt = get(r, :pure_t, 0.0); rt = get(r, :robust_t, 0.0)
        tc += ct; tp += pt; tr += rt
        @printf("  %-5d %-12.2f %-12.2f %-12.2f %-10d %-10d\n",
                r[:ir], ct, pt, rt, get(r, :confirm_ev, 0), get(r, :pure_ev, 0))
    end
    @printf("  %-5s %-12.2f %-12.2f %-12.2f\n", "SUM", tc, tp, tr)
    tp > 0 && tc > 0 && @printf("  pure/confirm wallclock = %.3f   (confirm/pure speedup = %.2fx)\n", tp / tc, tc / tp)
end

main()
