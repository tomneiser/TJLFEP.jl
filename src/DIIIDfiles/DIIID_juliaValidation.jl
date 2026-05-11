using Distributed

SCAN_N = 20
IS_EP = 1 # fast D is EP driver
rho = [0.01, 0.06, 0.11, 0.16, 0.21, 0.27, 0.32, 0.37, 0.42, 0.47,
        0.53, 0.58, 0.63, 0.68, 0.73, 0.79, 0.84, 0.89, 0.94, 1.0]
# rho = [0.01, 0.06, 0.11, 0.16, 0.21, 0.27, 0.32, 0.37]
# rho = [0.01]
N_BASIS = 8
tot = 400
nthreads = max(1, div(tot-2, SCAN_N + 1))

sysimage_path = expanduser("~/.julia/dev/EP_TJLF/build/noTJLF_TJLFEP_sysimage.so")
project_path = expanduser("../../")
addprocs(SCAN_N, exeflags=`--project=$project_path --sysimage=$sysimage_path --threads=$nthreads`)
@everywhere println("worker $(myid()) on $(gethostname())")

@everywhere using Pkg
@everywhere Pkg.activate("../../")
@everywhere @time using CUDA
@everywhere using TJLFEP
@everywhere using TJLFEP: TJLF
@everywhere using LinearAlgebra
@everywhere import FUSE
@everywhere import IMAS
@everywhere import GACODE
@everywhere BLAS.set_num_threads(1)
@everywhere TJLF.pick_device(:auto)

# Verify per-worker GPU state before any compute
@everywhere begin
    wid = myid()
    ext_loaded = !isnothing(Base.get_extension(TJLFEP, :TJLFEPCUDAExt))
    cuda_ok    = TJLF._cuda_functional()
    solve_set  = !isnothing(TJLF._CUDA_SOLVE[])
    dev        = TJLF.pick_device(:auto)
    println("worker $wid: ext_loaded=$ext_loaded  cuda_functional=$cuda_ok  _CUDA_SOLVE_set=$solve_set  device=$dev")
end

begin
    dir = pwd()

    use_gpu   = (TJLF.pick_device(:auto) === :gpu)
    println("Using device: ", use_gpu ? "GPU" : "CPU")

    if use_gpu
        CUDA.zeros(1) # warm up GPU and CUDA.jl
        Threads.@threads for i in 1:Threads.nthreads()
            CUDA.zeros(1)
        end

        if isnothing(TJLF._CUDA_SOLVE[])
            println("WARNING: _CUDA_SOLVE[] is nothing on main process — GPU dispatch disabled")
        end
    end

    # check CUDA is in use
    if !isnothing(Base.get_extension(TJLFEP, :TJLFEPCUDAExt))
        println("extension loaded: ", !isnothing(Base.get_extension(TJLFEP, :TJLFEPCUDAExt)))
        println("cuda functional: ", TJLF._cuda_functional())
        println("cuda solve loaded: ", !isnothing(TJLF._CUDA_SOLVE[])) 
    end

    inputFile = joinpath(dir, "input.gacode_202017C42_500ms")
    println("rho = ", rho)

    OptionsDict = Dict{String, Any}("nn" => 5, "nr" => 101, "jtscale_max" => 1, "nmodes" => 4,
    "PROCESS_IN" => 5, "THRESHOLD_FLAG" => 0, "N_BASIS" => N_BASIS, "SCAN_METHOD" => 2, "REJECT_I_PINCH_FLAG" => 0, "REJECT_E_PINCH_FLAG" => 0, "REJECT_TH_PINCH_FLAG" => 0, "REJECT_EP_PINCH_FLAG" => 0,
    "REJECT_TEARING_FLAG" => 1, "ROTATIONAL_SUPPRESSION_FLAG" => 0, "PPRIME_METHOD" => 3,"QL_RATIO_THRESH" => 10.0, "THETA_SQ_THRESH" => 100.0, "Q_SCALE" => 1.0,
    "WRITE_WAVEFUNCTION" => 1, "KY_MODEL" => 2, "SCAN_N" => SCAN_N, "IRS" => 2, "FACTOR_IN_PROFILE" => false, "FACTOR_IN" => 10.0,
    "WIDTH_IN_FLAG" => false, "WIDTH_MIN" => 1.0, "WIDTH_MAX" => 2.0, "INPUT_PROFILE_METHOD" => 2, "N_ION" => 2, "IS_EP" => IS_EP, "REAL_FREQ" => 1)

    @time inputGACODE = GACODE.load(inputFile)
    @time dd = IMAS.dd(inputGACODE)

    job_id = get(ENV, "SLURM_JOB_ID", "local")
    outdir = joinpath(@__DIR__, "$(use_gpu ? "GPU" : "CPU")_n$(N_BASIS)_$(SCAN_N)_$(job_id)")
    mkpath(outdir)

    if use_gpu
        lines = readlines(`nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`)
        for l in lines
            idx, mem, util, temp = strip.(split(l, ","))
            println("GPU $idx AFTER: mem=$(mem) MiB  util=$(util)%  temp=$(temp)°C")
        end
    end

    t1 = time()
    cd(outdir) do
        @time runTHD(dd, rho, OptionsDict; printout = true, saveFiles = false, dir = joinpath(@__DIR__, "fileInput"), use_gpu = use_gpu)
    end
    t2 = time()

    if use_gpu
        lines = readlines(`nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`)
        for l in lines
            idx, mem, util, temp = strip.(split(l, ","))
            println("GPU $idx AFTER: mem=$(mem) MiB  util=$(util)%  temp=$(temp)°C")
        end
    end

    @time make_crit_grad_plots(""; dir=outdir)
end
println("example done in $(t2 - t1) s")
