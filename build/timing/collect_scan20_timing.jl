# Harvest SCAN_N=20 compute times (seconds) into timing_runs/scan20_timing.csv

const BUILD = normpath(@__DIR__, "..")
const OUT_CSV = joinpath(BUILD, "timing_runs", "scan20_timing.csv")
const BASIS = [6, 8, 16, 32]

const NB6_FALLBACK = (
    fortran = 62.506,
    julia_cpu = 154.719,
    julia_gpu = 199.813,
)

# Completed comparison/debug jobs without TIMING_RESULT in the log (sacct srun step, seconds).
const KNOWN_SCAN20 = Dict(
    16 => (fortran = 347.0,  julia_cpu = nothing,  julia_gpu = 294.0),   # 53260814.0 (5:47), 53260865 @time, 53260819.0 (4:54)
    32 => (fortran = 1546.0, julia_cpu = 13200.76, julia_gpu = 581.0),   # 53260741.0 (25:46), 53260866 @time, 53260659.0 (9:41)
)

# Report TOTAL WALLCLOCK of the run (phase=total_job): end-to-end wall including process
# spawn, sysimage load, compute, and merge. The Fortran binary compile and the Julia sysimage
# build are the equivalent one-time costs and are EXCLUDED from both (neither is in total_job).
# Falls back to scan/compute for older logs that lack a total_job line.
# Returns (fort, cpu, gpu, cpu_ad, gpu_ad) seconds. The `solver` token (solver=grid|ad)
# distinguishes the autodiff series; lines without it are treated as solver=grid (back-compat).
function parse_timing_log(path::String)
    isfile(path) || return nothing, nothing, nothing, nothing, nothing
    vals = Dict{String,Float64}()   # "backend|device|solver|phase" => seconds
    for line in readlines(path)
        occursin("TIMING_RESULT", line) || continue
        s = replace(line, " " => "")
        m = match(r"seconds=([0-9.]+)", s)
        m === nothing && continue
        t = parse(Float64, m.captures[1])
        be  = occursin("backend=fortran", s) ? "fortran" :
              occursin("backend=julia", s)   ? "julia"   : ""
        dev = occursin("device=gpu", s) ? "gpu" :
              occursin("device=cpu", s) ? "cpu" : ""
        sol = occursin("solver=ad", s) ? "ad" : "grid"
        ph  = occursin("phase=total_job", s) ? "total"   :
              occursin("phase=scan", s)      ? "scan"    :
              occursin("phase=compute", s)   ? "compute" : ""
        (isempty(be) || isempty(ph)) && continue
        vals["$be|$dev|$sol|$ph"] = t
    end
    pick(ks...) = (for k in ks; haskey(vals, k) && return vals[k]; end; nothing)
    fort   = pick("fortran|cpu|grid|total", "fortran|cpu|grid|scan")
    cpu    = pick("julia|cpu|grid|total", "julia|cpu|grid|compute", "julia|cpu|grid|scan")
    gpu    = pick("julia|gpu|grid|total", "julia|gpu|grid|scan")
    cpu_ad = pick("julia|cpu|ad|total", "julia|cpu|ad|compute", "julia|cpu|ad|scan")
    gpu_ad = pick("julia|gpu|ad|total", "julia|gpu|ad|scan")
    return fort, cpu, gpu, cpu_ad, gpu_ad
end

function parse_ok_in(path::String)
    isfile(path) || return nothing
    for line in reverse(readlines(path))
        m = match(r"OK in ([0-9.]+) s", line)
        m !== nothing && return parse(Float64, m.captures[1])
    end
    return nothing
end

"""@time runTHD pmap wall from debug_compare_*_distributed.jl logs."""
function parse_julia_pmap_compute(path::String)
    isfile(path) || return nothing
    for line in reverse(readlines(path))
        m = match(r"^([0-9.]+) seconds \(", line)
        m !== nothing && return parse(Float64, m.captures[1])
    end
    return nothing
end

"""Max per-task 'OK scan_index=… in X s' from gacode GPU comparison logs."""
function parse_gpu_max_task(path::String)
    isfile(path) || return nothing
    times = Float64[]
    for line in readlines(path)
        m = match(r"OK scan_index=\d+ ir=\d+ sfmin=[^\n]+ in ([0-9.]+) s", line)
        m === nothing && continue
        push!(times, parse(Float64, m.captures[1]))
    end
    return isempty(times) ? nothing : maximum(times)
end

function newest(pattern::String)
    rx = Regex("^" * replace(pattern, "*" => ".*") * "\$")
    files = String[]
    for f in readdir(BUILD)
        occursin(rx, f) || continue
        push!(files, joinpath(BUILD, f))
    end
    isempty(files) && return nothing
    return first(sort(files, by=f -> mtime(f), rev=true))
end

function collect_nb(nb::Int)
    fort = cpu = gpu = cpu_ad = gpu_ad = gpu_ad_mps = nothing

    # Timing harness logs (preferred -- these are the fresh per-nbasis runs). Take these
    # FIRST so a fresh nb6 run wins over the legacy no-nbasis logs / NB6_FALLBACK below.
    # Grid globs use _[0-9]* so they do NOT match the autodiff _ad_ logs below.
    f_time = newest("time_scan20_nb$(nb)_fortran_[0-9]*.out")
    j_time = newest("time_scan20_nb$(nb)_julia_cpu_[0-9]*.out")
    g_time = newest("time_scan20_nb$(nb)_julia_gpu_[0-9]*.out")
    if fort === nothing && f_time !== nothing
        ft, _, _ = parse_timing_log(f_time)
        fort = ft
    end
    if cpu === nothing && j_time !== nothing
        _, ct, _ = parse_timing_log(j_time)
        cpu = ct
    end
    if gpu === nothing && g_time !== nothing
        _, _, gt = parse_timing_log(g_time)
        gpu = gt
    end

    # Autodiff (solver=:ad) harness logs. The MPS-AD log (_gpu_ad_mps_) carries the SAME
    # device=gpu solver=ad TIMING_RESULT tokens as the baseline AD-GPU log, so they are
    # disambiguated only by filename: the baseline globs use _[0-9]* to exclude _mps_.
    ja_time  = newest("time_scan20_nb$(nb)_julia_cpu_ad_[0-9]*.out")
    ga_time  = newest("time_scan20_nb$(nb)_julia_gpu_ad_[0-9]*.out")
    gam_time = newest("time_scan20_nb$(nb)_julia_gpu_ad_mps_*.out")
    if ja_time !== nothing
        cpu_ad = parse_timing_log(ja_time)[4]
    end
    if ga_time !== nothing
        gpu_ad = parse_timing_log(ga_time)[5]
    end
    if gam_time !== nothing
        gpu_ad_mps = parse_timing_log(gam_time)[5]
    end

    f_dbg = newest("debug_nb$(nb)_fortran20_10n_*.out")
    j_dbg = newest("debug_nb$(nb)_julia20_10n_*.out")
    g_dbg = newest("gacode_nb$(nb)_scan20_gpu5_*.out")

    if fort === nothing && f_dbg !== nothing
        ft, _, _ = parse_timing_log(f_dbg)
        fort = ft
    end
    if cpu === nothing && j_dbg !== nothing
        _, ct, _ = parse_timing_log(j_dbg)
        pmap_t = parse_julia_pmap_compute(j_dbg)
        ok = parse_ok_in(j_dbg)
        if ct !== nothing
            cpu = ct
        elseif pmap_t !== nothing
            cpu = pmap_t
        elseif ok !== nothing
            cpu = ok
        end
    end
    if gpu === nothing && g_dbg !== nothing
        _, _, gt = parse_timing_log(g_dbg)
        gpu = gt
    end

    # KNOWN_SCAN20 are OLD pre-optimization/pre-sysimage numbers: use them only as a
    # FALLBACK when no fresh log was found, never to override a freshly measured run.
    if haskey(KNOWN_SCAN20, nb)
        k = KNOWN_SCAN20[nb]
        if k.fortran !== nothing
            fort = fort === nothing ? k.fortran : fort
        end
        if k.julia_cpu !== nothing
            cpu = cpu === nothing ? k.julia_cpu : cpu
        end
        if k.julia_gpu !== nothing
            gpu = gpu === nothing ? k.julia_gpu : gpu
        end
    end
    if gpu === nothing && g_dbg !== nothing
        gpu = parse_gpu_max_task(g_dbg)
    end

    # nb6 legacy fallbacks: older runs used logs without the "nb6" tag. Only consulted if
    # no fresh nb6-tagged log was found above.
    if nb == 6
        if fort === nothing
            lf = newest("time_scan20_fortran_*.out")
            lf !== nothing && (fort = parse_timing_log(lf)[1])
        end
        if cpu === nothing
            lc = newest("time_scan20_julia_cpu_*.out")
            lc !== nothing && (cpu = parse_timing_log(lc)[2])
        end
        if gpu === nothing
            lg = newest("time_scan20_julia_gpu_*.out")
            lg !== nothing && (gpu = parse_timing_log(lg)[3])
        end
        fort = fort === nothing ? NB6_FALLBACK.fortran : fort
        cpu = cpu === nothing ? NB6_FALLBACK.julia_cpu : cpu
        gpu = gpu === nothing ? NB6_FALLBACK.julia_gpu : gpu
    end

    return fort, cpu, gpu, cpu_ad, gpu_ad, gpu_ad_mps
end

function collect_scan20_timing!()
    rows = ["n_basis,fortran_s,julia_cpu_s,julia_gpu_s,julia_cpu_ad_s,julia_gpu_ad_s,julia_gpu_ad_mps_s,notes"]
    for nb in BASIS
        fort, cpu, gpu, cpu_ad, gpu_ad, gpu_ad_mps = collect_nb(nb)
        notes = String[]
        fort === nothing && push!(notes, "fortran_missing")
        cpu === nothing && push!(notes, "julia_cpu_missing")
        gpu === nothing && push!(notes, "julia_gpu_missing")
        cpu_ad === nothing && push!(notes, "julia_cpu_ad_missing")
        gpu_ad === nothing && push!(notes, "julia_gpu_ad_missing")
        # gpu_ad_mps is Phase B (pending the GPU/MPS AD kernel); absence is expected, not flagged.
        note = isempty(notes) ? "ok" : join(notes, ";")
        fs = fort === nothing ? "" : string(fort)
        cs = cpu === nothing ? "" : string(cpu)
        gs = gpu === nothing ? "" : string(gpu)
        cas = cpu_ad === nothing ? "" : string(cpu_ad)
        gas = gpu_ad === nothing ? "" : string(gpu_ad)
        gams = gpu_ad_mps === nothing ? "" : string(gpu_ad_mps)
        push!(rows, "$nb,$fs,$cs,$gs,$cas,$gas,$gams,$note")
    end
    mkpath(dirname(OUT_CSV))
    write(OUT_CSV, join(rows, "\n") * "\n")
    return OUT_CSV, rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(normpath(@__DIR__, "..", ".."))
    path, rows = collect_scan20_timing!()
    println("Wrote ", path)
    for line in rows[2:end]
        println("  ", line)
    end
end
