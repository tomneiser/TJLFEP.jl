# (A) Re-run grid-floor-guarded adf1 + lean DIRECT-20 on the 4 radii (scored vs grid/dense/DIRECT-40).
# (B) IR=95 sanity: the DIRECT-40 run had wild 4x dispersion (grid 2.64 / dense 4.20 / refine0 5.49 /
#     DIRECT 1.38). Settle it with a FINE faithful (ky,w) grid = ground truth global-min, locate the
#     DIRECT optimum, and check whether ~1.38 is a robust basin or a knife-edge / box-edge artifact.
#
# Env: USE_GPU(1) NB(32) RADII(22,38,48,95) DIRECT20_EVALS(20) FINE_NKY(12) FINE_NW(20)
#      INNER(mps_team) MPS_TEAM(4)

using TJLF, TJLFEP, NLopt, Printf, Distributed

const USE_GPU  = get(ENV, "USE_GPU", "1") == "1"
const INNER    = Symbol(get(ENV, "INNER", "threads"))
const MPS_TEAM = parse(Int, get(ENV, "MPS_TEAM", "0"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
const USE_MPS  = USE_GPU && INNER === :mps_team && MPS_TEAM > 0

if USE_GPU
    using CUDA
    @assert CUDA.functional() "USE_GPU=1 but no functional GPU"
end

if USE_MPS
    let root = normpath(@__DIR__, "..", ".."),
        team_gpus = String.(split(get(ENV, "TEAM_GPUS", get(ENV, "CUDA_VISIBLE_DEVICES", "0")), ',', keepempty=false)),
        base_env = Dict{String,String}()
        for k in ("JULIA_DEPOT_PATH", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                  "JULIA_CUDA_USE_COMPAT", "JULIA_CUDA_MEMORY_POOL")
            haskey(ENV, k) && (base_env[k] = ENV[k])
        end
        base_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
        for w in 1:MPS_TEAM
            env = copy(base_env)
            env["CUDA_VISIBLE_DEVICES"] = team_gpus[(w - 1) % length(team_gpus) + 1]
            addprocs(1; exeflags=`--project=$(root) -t $(THREADS_PER_WORKER) --startup-file=no`, env=env)
        end
    end
    @everywhere begin
        using CUDA, TJLFEP, TJLF, LinearAlgebra
        BLAS.set_num_threads(1)
        CUDA.functional() && CUDA.device!(first(CUDA.devices()))
    end
end

include(joinpath(@__DIR__, "direct_solver.jl"))

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const NB     = parse(Int, get(ENV, "NB", "32"))
const TGLFEP = joinpath(CASE, "input_scan20_nb$(NB).TGLFEP")
const RADII  = parse.(Int, split(get(ENV, "RADII", "22,38,48,95"), ',', keepempty=false))
const D20_EVALS = parse(Int, get(ENV, "DIRECT20_EVALS", "20"))
const FINE_NKY  = parse(Int, get(ENV, "FINE_NKY", "12"))
const FINE_NW   = parse(Int, get(ENV, "FINE_NW", "20"))

const GRID    = Dict(22 => 0.17577, 38 => 0.01953, 48 => 0.039062, 95 => 2.636713)
const DENSE   = Dict(22 => 0.16246, 38 => 0.019531, 48 => 0.028553, 95 => 4.2007)
const DIRECT40= Dict(22 => 0.16476, 38 => 0.019531, 48 => 0.026728, 95 => 1.3812)

pct(x, ref) = (isfinite(x) && isfinite(ref) && ref != 0) ? @sprintf("%+6.1f%%", 100*(x-ref)/ref) : "   n/a "

# Fine faithful (ky,w) grid → ground-truth global min. Point-parallel over the team; each point's
# faithful eval runs serial to avoid nested team usage.
function fine_faithful_grid(ep0, prof; nky, nw, team, use_gpu)
    gth = TJLFEP._gamma_thresh_for(ep0, prof)
    shi = Float64(ep0.FACTOR_IN); slo = shi/512.0
    wlo, whi = Float64(ep0.WIDTH_MIN), Float64(ep0.WIDTH_MAX)
    kyhats = [(1.0/nky)*i for i in 1:nky]
    widths = nw == 1 ? [0.5*(wlo+whi)] : [wlo + (whi-wlo)/(nw-1)*(i-1) for i in 1:nw]
    pts = [(ky, w) for ky in kyhats for w in widths]
    res = TJLFEP._ad_pmap(idx -> begin
            ky, w = pts[idx]
            r = TJLFEP.marginal_factor_faithful(ep0, prof; kyhat=ky, width=w, gamma_thresh=gth,
                    scan_lo=slo, scan_hi=shi, threaded=false, inner=:serial, team=nothing, use_gpu=use_gpu)
            f = (r.binding !== :none && isfinite(r.factor_faithful)) ? r.factor_faithful : Inf
            (; ky=ky, w=w, f=f, binding=r.binding)
        end, length(pts); inner=(team===nothing ? :threads : :mps_team), team=team)
    j = argmin([Float64(r.f) for r in res])
    nfeas = count(r -> isfinite(r.f), res)
    return (; best=res[j], nfeas=nfeas, npts=length(pts), wlo=wlo, whi=whi)
end

function main()
    opts, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    opts.N_BASIS = NB
    team = USE_MPS ? workers() : nothing
    dev = USE_GPU ? CUDA.name(first(CUDA.devices())) : "CPU"
    @printf("DIII-D N_BASIS=%d  %s  inner=%s team=%s  direct20=%d  fine=%dx%d\n",
            NB, dev, String(INNER), team===nothing ? "-" : string(length(team)), D20_EVALS, FINE_NKY, FINE_NW)
    flush(stdout)

    let ep = deepcopy(opts); ep.IR = 38
        critical_factor_direct(ep, prof; max_evals=12, inner=INNER, team=team, use_gpu=USE_GPU)
        critical_factor_ad_f1seed(ep, prof; inner=INNER, team=team, use_gpu=USE_GPU)
    end

    println("\n========== (A) grid-floor-guarded adf1 vs lean DIRECT-20 ==========")
    for ir in RADII
        ep = deepcopy(opts); ep.IR = ir
        g = get(GRID, ir, NaN); d = get(DENSE, ir, NaN); d40 = get(DIRECT40, ir, NaN)
        t2 = @elapsed r2 = critical_factor_direct(ep, prof; max_evals=D20_EVALS, inner=INNER, team=team, use_gpu=USE_GPU)
        t1 = @elapsed r1 = critical_factor_ad_f1seed(ep, prof; inner=INNER, team=team, use_gpu=USE_GPU)
        @printf("\nIR=%-3d  grid=%-9.5g dense=%-9.5g direct40=%-9.5g\n", ir, g, d, d40)
        @printf("  %-8s sfmin=%-10.5g vsgrid=%s vsdense=%s vsD40=%s  full=%-4d eig=%-5d %6.1fs nconf=%d/%d st=%s\n",
                "direct20", r2.sfmin, pct(r2.sfmin,g), pct(r2.sfmin,d), pct(r2.sfmin,d40),
                r2.total_evals_full, r2.total_evals_eig, t2, r2.n_confirm, r2.n_samples, String(r2.status))
        @printf("  %-8s sfmin=%-10.5g vsgrid=%s vsdense=%s vsD40=%s  full=%-4d eig=%-5d %6.1fs ndesc=%d nconf=%d st=%s\n",
                "adf1*", r1.sfmin, pct(r1.sfmin,g), pct(r1.sfmin,d), pct(r1.sfmin,d40),
                r1.total_evals_full, r1.total_evals_eig, t1, r1.n_descend, r1.n_confirm, String(r1.status))
        flush(stdout)
    end

    println("\n========== (B) IR=95 sanity: is DIRECT-40's 1.38 a real basin? ==========")
    ep = deepcopy(opts); ep.IR = 95
    # DIRECT-40 optimum location (where it found 1.38)
    rd = critical_factor_direct(ep, prof; max_evals=40, inner=INNER, team=team, use_gpu=USE_GPU)
    wlo, whi = Float64(ep.WIDTH_MIN), Float64(ep.WIDTH_MAX)
    on_w_edge = (rd.width <= wlo + 1e-6*(whi-wlo)) || (rd.width >= whi - 1e-6*(whi-wlo))
    on_ky_edge = (rd.kyhat <= 1.1e-6) || (rd.kyhat >= 0.999)
    @printf("DIRECT-40 opt: sfmin=%.5g at (ky=%.5g, w=%.5g)  binding=%s  ky_edge=%s w_edge=%s [w in (%.4g,%.4g)]\n",
            rd.sfmin, rd.kyhat, rd.width, String(rd.binding), on_ky_edge, on_w_edge, wlo, whi)
    flush(stdout)
    # Fine faithful grid = ground truth
    tg = @elapsed fg = fine_faithful_grid(ep, prof; nky=FINE_NKY, nw=FINE_NW, team=team, use_gpu=USE_GPU)
    @printf("FINE faithful grid %dx%d (%d feasible/%d pts, %.1fs): min sfmin=%.5g at (ky=%.5g, w=%.5g) binding=%s\n",
            FINE_NKY, FINE_NW, fg.nfeas, fg.npts, tg, fg.best.f, fg.best.ky, fg.best.w, String(fg.best.binding))
    @printf("  refs: grid=%.5g dense=%.5g direct40=%.5g  ->  fine-grid truth vs direct40: %s\n",
            GRID[95], DENSE[95], DIRECT40[95], pct(rd.sfmin, fg.best.f))
    println("\n=== rerun+sanity experiment done ===")
end

main()
