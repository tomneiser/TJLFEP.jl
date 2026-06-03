# One radius of SCAN_N=20, run with an MPS *team*: the radius's inner kw-scan (256
# combos x 4 k-rounds) is distributed across MPS_TEAM local worker processes that all
# share this task's single GPU. Each worker has its own CUDA context (MPS), so their
# Xgeev eigensolves overlap on the device via Hyper-Q — the within-GPU latency lever
# that intra-process concurrency could not unlock (it corrupted Xgeev).
#
# Layout: 1 SLURM task = 1 radius = 1 GPU (pinned via CUDA_VISIBLE_DEVICES set by the
# batch script) + MPS_TEAM coordinator-local workers on that GPU. The MPS control
# daemon must already be running on the node (see mps-wrapper.sh / the batch script).
#
# Env:
#   MPS_TEAM     worker processes per GPU (default 4)
#   SCAN_INDEX   1-based radius to run (default: SLURM array task id + 1)
#   USE_GPU      must be "1"
#   TGLFEP_FILE / GACODE_FILE / CASE_DIR / OUT_DIR  as in run_gacode_scan20_array_task.jl

ENV["TJLFEP_FILE_ONLY"] = "1"

using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

@assert get(ENV, "USE_GPU", "") == "1" "run_gacode_scan20_mps_task.jl is GPU-only; set USE_GPU=1"

using CUDA
let rv = CUDA.runtime_version()
    rv >= v"12.6" || error("GPU run needs CUDA >= 12.6 for cusolverDnXgeev, but CUDA runtime is $rv. " *
                           "Load cudatoolkit/12.9.")
end

using Distributed

const ROOT = normpath(@__DIR__, "..")
const NTEAM = parse(Int, get(ENV, "MPS_TEAM", "4"))
const THREADS_PER_WORKER = parse(Int, get(ENV, "JULIA_WORKER_THREADS", "2"))
# INNER selects the within-radius parallelism: :mps_team (default; addproc'd MPS clients) or
# :threads (single-process threaded baseline on this task's GPU, for speedup/timing comparison).
const INNER = Symbol(get(ENV, "INNER", "mps_team"))
# Optional GPU pin list for the team workers, e.g. "0,1" to spread a radius across 2 GPUs.
# Defaults to this task's CUDA_VISIBLE_DEVICES (one GPU). Workers are round-robined over it.
const TEAM_GPUS = let s = get(ENV, "TEAM_GPUS", get(ENV, "CUDA_VISIBLE_DEVICES", "0"))
    String.(split(s, ',', keepempty=false))
end
# Optional GPU-worker sysimage (TJLFEP_gpu_sysimage.so): bakes the TJLF/TJLFEP/CUDA GPU
# eigensolve path so workers skip the ~110 s/team JIT on a cold spawn. Empty -> JIT as before.
const GPU_SYSIMAGE = get(ENV, "TJLFEP_GPU_SYSIMAGE", "")
const _SYSIMG_FLAGS = (!isempty(GPU_SYSIMAGE) && isfile(GPU_SYSIMAGE)) ? `--sysimage=$(GPU_SYSIMAGE)` : ``

if INNER === :mps_team
    # Spawn the team. Workers inherit the MPS pipe dir and are pinned (round-robin) across
    # TEAM_GPUS, so each worker connects as an MPS client on its assigned physical GPU.
    base_env = Dict{String,String}()
    for k in ("JULIA_DEPOT_PATH", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
              "TJLFEP_FILE_ONLY",
              # JULIA_CUDA_USE_COMPAT=false is REQUIRED: CUDA.jl's forward-compat driver shim
              # hangs in cuInit when initializing as an MPS client on Perlmutter. Without it,
              # every team worker's `using CUDA` blocks indefinitely.
              "JULIA_CUDA_USE_COMPAT", "JULIA_CUDA_MEMORY_POOL")
        haskey(ENV, k) && (base_env[k] = ENV[k])
    end
    base_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

    for w in 1:NTEAM
        env = copy(base_env)
        env["CUDA_VISIBLE_DEVICES"] = TEAM_GPUS[(w - 1) % length(TEAM_GPUS) + 1]
        addprocs(1; exeflags=`--project=$(ROOT) -t $(THREADS_PER_WORKER) --startup-file=no $(_SYSIMG_FLAGS)`,
                 env=env)
    end

    @everywhere begin
        ENV["TJLFEP_FILE_ONLY"] = "1"
        using CUDA
        using TJLFEP
        using TJLF
        using LinearAlgebra
        BLAS.set_num_threads(1)
        # touch the GPU so each worker's CUDA context (MPS client) is created up front
        CUDA.functional() && CUDA.device!(first(CUDA.devices()))
    end
end

using TJLFEP
using TJLF
using LinearAlgebra
BLAS.set_num_threads(1)

const CASE   = get(ENV, "CASE_DIR", joinpath(ROOT, "src", "DIIIDfiles", "202017C42_500ms_v3.1"))
const GACODE = get(ENV, "GACODE_FILE", joinpath(CASE, "input.gacode"))
const TGLFEP = get(ENV, "TGLFEP_FILE", joinpath(ROOT, "build", "debug_nb32", "input_scan20.TGLFEP"))
const OUT_DIR = get(ENV, "OUT_DIR", joinpath(@__DIR__, "gacode_scan20_mps_$(get(ENV, "SLURM_JOB_ID", "local"))_tasks"))

@assert isfile(GACODE) "missing $GACODE"
@assert isfile(TGLFEP) "missing $TGLFEP"

opts, _, _ = preprocess_gacode_inputs(GACODE, TGLFEP)
scan_n = opts.SCAN_N

scan_index = haskey(ENV, "SCAN_INDEX") ? parse(Int, ENV["SCAN_INDEX"]) : slurm_array_task_id() + 1
@assert 1 <= scan_index <= scan_n "scan_index=$scan_index invalid for SCAN_N=$scan_n"

printout = get(ENV, "TJLFEP_PRINTOUT", "0") == "1"

const TEAM = INNER === :mps_team ? workers() : nothing
println("=== gacode scan task (inner=$INNER) ===")
println("worker sysimage=", isempty(_SYSIMG_FLAGS.exec) ? "<none, JIT>" : GPU_SYSIMAGE)
println("scan_index=$scan_index / $scan_n  team=$(INNER === :mps_team ? nworkers() : 0) workers  " *
        "threads/worker=$THREADS_PER_WORKER  team_gpus=$(join(TEAM_GPUS, ','))")
println("OUT_DIR=$OUT_DIR")
println("CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "<unset>")) MPS=$(get(ENV, "CUDA_MPS_PIPE_DIRECTORY", "<none>"))")
println("GPU=", CUDA.functional() ? CUDA.name(first(CUDA.devices())) : "n/a")
flush(stdout)

t0 = time()
result = run_gacode_scan_task(
    GACODE, TGLFEP, scan_index;
    out_dir=OUT_DIR,
    use_gpu=true,
    printout=printout,
    inner=INNER,
    team=TEAM,
)
println("OK scan_index=$(result.scan_index) ir=$(result.ir) sfmin=$(result.sfmin) in $(round(time() - t0; digits=1)) s")
flush(stdout)

# Per-worker probe dump: each MPS team worker reports its own combo count and the time it
# spent in TJLFEP_ky vs TJLF.run (eigensolve), so we can see whether the workers actually
# overlapped on the GPU (high combo count per worker at low wall) or time-sliced/serialized.
if INNER === :mps_team && get(ENV, "TJLFEP_PROBE", "0") == "1"
    for w in workers()
        n, tky, trun = remotecall_fetch(w) do
            (TJLFEP._PROBE_N[], TJLFEP._PROBE_KY[], TJLFEP._PROBE_RUN[])
        end
        println("  [worker $w] combos=$n  sum(TJLFEP_ky)=$(round(tky; digits=1))s  sum(TJLF.run)=$(round(trun; digits=1))s")
    end
    flush(stdout)
end
