using Pkg
Pkg.activate("../../")
# include("TJLFEP.jl")
# Pkg.instantiate()
# using .TJLFEP
@time using Revise
@time using TJLFEP
@time using TJLF
@time using LinearAlgebra
@time import FUSE
@time import IMAS
@time import GACODE
Pkg.status()
BLAS.set_num_threads(1)
begin
    dir = pwd()

    inputFile = joinpath(dir, "input.gacode_202017C42_500ms")
    SCAN_N = 20
    IS_EP = 1 # fast D (merged with thermal D as single IMAS ion) is EP driver
    # rho = [0.01, 0.21, 0.41, 0.61, 0.81, 1.01]
    rho = [0.01, 0.06, 0.11, 0.16, 0.21, 0.27, 0.32, 0.37, 0.42, 0.47,
            0.53, 0.58, 0.63, 0.68, 0.73, 0.79, 0.84, 0.89, 0.94, 1.0]
    # rho = [0.01, 0.06, 0.11, 0.16, 0.21, 0.27, 0.32, 0.37, 0.42, 0.47,
            # 0.53, 0.58, 0.63, 0.68, 0.73, 0.79, 0.84, 0.89, 0.94]
    println("rho = ", rho)

    println("pre dict initialize")
    OptionsDict = Dict{String, Any}("nn" => 5, "nr" => 101, "jtscale_max" => 1, "nmodes" => 4,
    "PROCESS_IN" => 5, "THRESHOLD_FLAG" => 0, "N_BASIS" => 8,
    "SCAN_METHOD" => 2, "REJECT_I_PINCH_FLAG" => 0, "REJECT_E_PINCH_FLAG" => 0, "REJECT_TH_PINCH_FLAG" => 0, "REJECT_EP_PINCH_FLAG" => 0,
    "REJECT_TEARING_FLAG" => 1, "ROTATIONAL_SUPPRESSION_FLAG" => 0, "PPRIME_METHOD" => 3,"QL_RATIO_THRESH" => 10.0, "THETA_SQ_THRESH" => 100.0, "Q_SCALE" => 1.0,
    "WRITE_WAVEFUNCTION" => 1, "KY_MODEL" => 2, "SCAN_N" => SCAN_N, "IRS" => 2, "FACTOR_IN_PROFILE" => false, "FACTOR_IN" => 10.0,
    "WIDTH_IN_FLAG" => false, "WIDTH_MIN" => 1.0, "WIDTH_MAX" => 2.0, "INPUT_PROFILE_METHOD" => 2, "N_ION" => 2, "IS_EP" => IS_EP, "REAL_FREQ" => 1)

    println("pre dd")
    inputGACODE = GACODE.load(inputFile)
    dd = IMAS.dd(inputGACODE)
    println("dd done")
    
    println("runTHD")
    outdir = joinpath(@__DIR__, "output")
    mkpath(outdir)
    cd(outdir) do
        runTHD(dd, rho, OptionsDict; printout = true, saveFiles = true, dir = joinpath(@__DIR__, "fileInput"))
    end
    make_crit_grad_plots(""; dir=outdir)
    println("runTHD done")
end
println("example done")
