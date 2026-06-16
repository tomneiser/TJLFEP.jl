# Offline experiment: DIRECT global search for the (ky,w) critical-factor optimum.
#
# Motivation. The fixed 4x8 (ky,w) grid (critical_factor_confirm / robust refine=0) is EXACT on
# its own nodes but can miss an off-node basin: at DIII-D IR=48 every 4x8 method (incl. adaptive
# grid-zoom) sat ~+90% above the 6x10 dense truth. Fixed-grid refinement can't fix a basin the
# coarse sample never bracketed. DIRECT (DIviding RECTangles, Jones 1993) is a deterministic
# global box optimizer that adaptively refines wherever the surface looks promising — no fixed
# nodes — so it can localize an off-node minimum within a fixed eval budget.
#
# Design (mirrors the cheap-search + few-confirm structure of critical_factor_confirm):
#   1. Run NLopt GN_DIRECT_L on the CHEAP AE-band onset f1(ky,w) (IFLUX=false hull scan; AE-stable
#      -> ceiling penalty). DIRECT places ~max_evals samples, dense in the low-onset basin.
#   2. Faithful-confirm the lowest-onset SAMPLES with the expensive keep filters (IFLUX=true),
#      in increasing cheap order, early-stopping when cheap >= best faithful (faithful >= cheap
#      node-wise). Exact over the sampled set; DIRECT supplies the off-node coverage.
#
# Reported per radius vs the Fortran/grid sfmin AND vs the dense faithful min:
#   refine0  : critical_factor_robust refine=0       (fixed 4x8 grid — the speed/accuracy baseline)
#   confirm  : critical_factor_confirm               (fixed 4x8 grid, cheap-search + few-confirm)
#   direct   : critical_factor_direct (this file)     (DIRECT global search + few-confirm)
#   dense    : robust nkyhat=DENSE_NKY x nefwid=DENSE_NW refine=0  (continuous-truth global min)
#
# Env: USE_GPU (0/1, default 0=CPU offline), NB (default 32), RADII (csv),
#      DENSE (0/1, default 1), DENSE_NKY (default 6), DENSE_NW (default 10),
#      DIRECT_EVALS (default 40, DIRECT cheap-eval budget), DIRECT_NEIG (default 24, hull pts/eval),
#      INNER (threads|mps_team), MPS_TEAM (team size for mps_team).

using TJLF
using TJLFEP
using NLopt
using Printf
using Distributed   # top-level so @everywhere is in scope when the USE_MPS block is macroexpanded

const USE_GPU  = get(ENV, "USE_GPU", "0") == "1"
const INNER    = Symbol(get(ENV, "INNER", "threads"))
const MPS_TEAM = parse(Int, get(ENV, "MPS_TEAM", "0"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
const USE_MPS  = USE_GPU && INNER === :mps_team && MPS_TEAM > 0

if USE_GPU
    using CUDA
    @assert CUDA.functional() "USE_GPU=1 but no functional GPU"
end

# MPS team: spawn MPS_TEAM local worker processes that share this task's GPU via Hyper-Q so their
# Xgeev eigensolves overlap (the within-GPU latency lever). Mirrors run_gacode_scan20_mps_task.jl;
# requires the MPS control daemon already up on the node (launch via common/mps-scan-wrapper.sh).
# The (ky,w)/factor eigensolves inside _ae_unstable_window/marginal_factor_faithful fan out across
# team=workers() via _ad_pmap; DIRECT's outer loop still drives the master serially (NLopt).
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
        using CUDA
        using TJLFEP
        using TJLF
        using LinearAlgebra
        BLAS.set_num_threads(1)
        CUDA.functional() && CUDA.device!(first(CUDA.devices()))
    end
end

const CASE   = joinpath(@__DIR__, "..", "..", "examples", "DIIID_202017C42_500ms_v3.1")
const GACODE = joinpath(CASE, "input.gacode")
const NB     = parse(Int, get(ENV, "NB", "32"))
const TGLFEP = joinpath(CASE, "input_scan20_nb$(NB).TGLFEP")
const DENSE  = get(ENV, "DENSE", "1") == "1"
const DENSE_NKY = parse(Int, get(ENV, "DENSE_NKY", "6"))
const DENSE_NW  = parse(Int, get(ENV, "DENSE_NW", "10"))
const DIRECT_EVALS = parse(Int, get(ENV, "DIRECT_EVALS", "40"))
const DIRECT_NEIG  = parse(Int, get(ENV, "DIRECT_NEIG", "24"))
const RADII = let s = get(ENV, "RADII", "22,38,48,95")
    parse.(Int, split(s, ',', keepempty=false))
end

# Fortran/grid sfmin per radius (production answer) from the scan20 grid run sfmin_scan.txt.
const GRID = Dict(2 => 0.937419, 7 => 0.624945, 17 => 0.234367, 22 => 0.175775,
                  33 => 0.117178, 38 => 0.019530, 43 => 0.019531, 48 => 0.039062,
                  95 => 2.636713)

pct(x, ref) = (isfinite(x) && isfinite(ref) && ref != 0) ? @sprintf("%+6.1f%%", 100*(x-ref)/ref) : "   n/a "

# ── DIRECT-based critical factor (prototype; reuses TJLFEP internals) ─────────────────────────
# Globally minimize the cheap AE-band onset f1(ky,w) over the (ky,w) box with NLopt GN_DIRECT_L,
# recording every sample, then faithful-confirm the lowest-onset samples with early-stop.
function critical_factor_direct(ep0, prof; gamma_thresh=nothing,
                                scan_lo=nothing, scan_hi=nothing,
                                n_eig::Int=DIRECT_NEIG, max_evals::Int=DIRECT_EVALS,
                                inner::Symbol=:threads, team=nothing,
                                use_gpu::Bool=false, verbose::Bool=false)
    gth = gamma_thresh === nothing ? TJLFEP._gamma_thresh_for(ep0, prof) : gamma_thresh
    shi = scan_hi === nothing ? Float64(ep0.FACTOR_IN) : scan_hi
    slo = scan_lo === nothing ? shi / 512.0 : scan_lo
    kylo, kyhi = 1.0e-6, 1.0
    wlo, whi = Float64(ep0.WIDTH_MIN), Float64(ep0.WIDTH_MAX)
    PENALTY = shi   # AE-stable (no interior onset) ranks as "onset at the ceiling" — keeps scale tight

    samples = NamedTuple[]
    eig_evals = Ref(0)

    cheap_onset = function (ky, w)
        ep = deepcopy(ep0); ep.KYHAT_IN = ky; ep.WIDTH_IN = w
        win = TJLFEP._ae_unstable_window(ep, prof, gth; scan_lo=slo, scan_hi=shi,
                  n_eig=n_eig, threaded=true, use_gpu=use_gpu, inner=inner, team=team)
        eig_evals[] += win.evals
        f = win.unstable ? win.f1 : PENALTY
        push!(samples, (; ky=ky, w=w, f=f, unstable=win.unstable, pinned=win.pinned_lo))
        return f
    end

    opt = NLopt.Opt(:GN_DIRECT_L, 2)
    NLopt.lower_bounds!(opt, [kylo, wlo])
    NLopt.upper_bounds!(opt, [kyhi, whi])
    NLopt.maxeval!(opt, max_evals)
    NLopt.min_objective!(opt, (x, grad) -> cheap_onset(x[1], x[2]))
    x0 = [0.5*(kylo+kyhi), 0.5*(wlo+whi)]
    (_minf, _minx, ret) = NLopt.optimize(opt, x0)

    # confirm: lowest cheap onset first, early-stop when cheap >= best faithful (faithful >= cheap)
    order = sortperm([Float64(s.f) for s in samples])
    best_f = Inf; best_ky=NaN; best_w=NaN; best_bind=:none
    total_full = 0; n_confirm = 0; results = Any[]
    for i in order
        s = samples[i]
        s.unstable || break          # remaining samples are AE-stable penalties — nothing to confirm
        s.f >= best_f && break        # bound: no remaining sample can yield a smaller faithful onset
        r = TJLFEP.marginal_factor_faithful(ep0, prof; kyhat=s.ky, width=s.w,
                gamma_thresh=gth, scan_lo=slo, scan_hi=shi, threaded=true,
                inner=inner, team=team, use_gpu=use_gpu)
        n_confirm += 1
        total_full += r.evals_full; eig_evals[] += r.evals_eig
        push!(results, (; s.ky, s.w, cheap=s.f, faithful=r.factor_faithful, binding=r.binding))
        if r.binding != :none && isfinite(r.factor_faithful) && r.factor_faithful < best_f
            best_f = r.factor_faithful; best_ky=s.ky; best_w=s.w; best_bind=r.binding
        end
        verbose && @info "direct-confirm" ky=s.ky w=s.w cheap=s.f faithful=r.factor_faithful binding=r.binding best=best_f
    end

    status = (best_bind===:none || !isfinite(best_f)) ? :no_onset :
             (best_f >= 0.999*shi ? :cap : :ok)
    return (; sfmin=best_f, kyhat=best_ky, width=best_w, binding=best_bind, status=status,
            n_samples=length(samples), n_confirm=n_confirm, nlopt_ret=ret,
            total_evals_full=total_full, total_evals_eig=eig_evals[], results=results)
end

function main()
    opts, prof, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
    opts.N_BASIS = NB
    team = USE_MPS ? workers() : nothing
    dev = USE_GPU ? CUDA.name(first(CUDA.devices())) : "CPU"
    @printf("DIII-D N_BASIS=%d  %s   dense=%s(%dx%d)  DIRECT evals=%d neig=%d  inner=%s team=%s\n",
            NB, dev, DENSE ? "on" : "off", DENSE_NKY, DENSE_NW, DIRECT_EVALS, DIRECT_NEIG,
            String(INNER), team === nothing ? "-" : string(length(team)))
    flush(stdout)

    # warm up / compile every branch once on a cheap interior radius
    let ep = deepcopy(opts); ep.IR = 38
        TJLFEP.critical_factor_robust(ep, prof; refine_rounds=0, inner=INNER, team=team, use_gpu=USE_GPU)
        TJLFEP.critical_factor_confirm(ep, prof; inner=INNER, team=team, use_gpu=USE_GPU)
        critical_factor_direct(ep, prof; max_evals=12, inner=INNER, team=team, use_gpu=USE_GPU)
        TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:adaptive, faithful_confirm=true, inner=INNER, team=team, use_gpu=USE_GPU)
        TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:always, faithful_confirm=true, inner=INNER, team=team, use_gpu=USE_GPU)
        DENSE && TJLFEP.critical_factor_robust(ep, prof; nkyhat=DENSE_NKY, nefwid=DENSE_NW,
                                               refine_rounds=0, inner=INNER, team=team, use_gpu=USE_GPU)
    end

    for ir in RADII
        ep = deepcopy(opts); ep.IR = ir
        gref = get(GRID, ir, NaN)

        dtruth = NaN; dfull = 0
        if DENSE
            rd = TJLFEP.critical_factor_robust(ep, prof; nkyhat=DENSE_NKY, nefwid=DENSE_NW,
                    refine_rounds=0, gamma_thresh=nothing, inner=INNER, team=team, use_gpu=USE_GPU)
            dtruth = rd.sfmin; dfull = rd.total_evals_full
        end

        t0 = @elapsed r0 = TJLFEP.critical_factor_robust(ep, prof; refine_rounds=0,
                gamma_thresh=nothing, inner=INNER, team=team, use_gpu=USE_GPU)
        tc = @elapsed rc = TJLFEP.critical_factor_confirm(ep, prof; nkyhat=4, nefwid=8,
                gamma_thresh=nothing, inner=INNER, team=team, use_gpu=USE_GPU)
        td = @elapsed rd2 = critical_factor_direct(ep, prof; inner=INNER, team=team, use_gpu=USE_GPU)
        # guarded :ad (faithful-confirmed): detect-and-reseed vs always-multistart, the speed-keeping
        # fix for :ad's local-min outliers. faithful_confirm lifts the AE-onset min to the keep onset.
        # Timed under :threads (NOT the experiment's INNER): guarded :ad is :ad-class — a short serial
        # descent with only a small parallel seed grid, so threads beats MPS (same as the :ad path).
        tga = @elapsed rga = TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:adaptive,
                faithful_confirm=true, inner=:threads, team=nothing, use_gpu=USE_GPU)
        tgl = @elapsed rgl = TJLFEP.critical_factor_ad_guarded(ep, prof; guard=:always,
                faithful_confirm=true, inner=:threads, team=nothing, use_gpu=USE_GPU)

        @printf("\nIR=%-3d  grid=%-10.5g  dense(%dx%d)=%-10.5g [full=%d]\n",
                ir, gref, DENSE_NKY, DENSE_NW, dtruth, dfull)
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  full=%-4d eig=%-5d  %6.1fs  %-14s st=%s\n",
                "refine0", r0.sfmin, pct(r0.sfmin, gref), pct(r0.sfmin, dtruth),
                r0.total_evals_full, r0.total_evals_eig, t0, String(r0.binding), String(r0.status))
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  full=%-4d eig=%-5d  %6.1fs  %-14s nconf=%d/%d st=%s\n",
                "confirm", rc.sfmin, pct(rc.sfmin, gref), pct(rc.sfmin, dtruth),
                rc.total_evals_full, rc.total_evals_eig, tc, String(rc.binding),
                rc.n_confirm, rc.npts, String(rc.status))
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  full=%-4d eig=%-5d  %6.1fs  %-14s nconf=%d/%d ret=%s st=%s\n",
                "direct", rd2.sfmin, pct(rd2.sfmin, gref), pct(rd2.sfmin, dtruth),
                rd2.total_evals_full, rd2.total_evals_eig, td, String(rd2.binding),
                rd2.n_confirm, rd2.n_samples, String(rd2.nlopt_ret), String(rd2.status))
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  evals=%-5d            %6.1fs  %-14s starts=%d flag=%s(edge=%s,amb=%s) st=%s\n",
                "ad-adapt", rga.sfmin, pct(rga.sfmin, gref), pct(rga.sfmin, dtruth),
                rga.evals, tga, String(rga.binding), rga.n_starts, rga.flagged, rga.on_edge, rga.ambiguous, String(rga.status))
        @printf("  %-8s sfmin=%-10.5g  vsgrid=%s vsdense=%s  evals=%-5d            %6.1fs  %-14s starts=%d st=%s\n",
                "ad-all", rgl.sfmin, pct(rgl.sfmin, gref), pct(rgl.sfmin, dtruth),
                rgl.evals, tgl, String(rgl.binding), rgl.n_starts, String(rgl.status))
        flush(stdout)
    end
    println("\n=== direct experiment done ===")
end

main()
