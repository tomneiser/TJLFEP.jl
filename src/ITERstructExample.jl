using Pkg
Pkg.activate("..")
# include("TJLFEP.jl")
# Pkg.instantiate()
# using .TJLFEP
@time using Revise
@time using TJLFEP
@time using TJLF
@time using LinearAlgebra
@time import FUSE
@time import IMAS
Pkg.status()
BLAS.set_num_threads(1)
begin

    SCAN_N = 6
    rho = [0.01, 0.2, 0.41, 0.61, 0.81, 1.0]
    println("rho = ", rho)

    println("pre dict initialize")
    OptionsDict = Dict{String, Any}("nn" => 5, "nr" => 201, "jtscale_max" => 1, "nmodes" => 4,
    "PROCESS_IN" => 5, "THRESHOLD_FLAG" => 0, "N_BASIS" => 2,
    "SCAN_METHOD" => 1, "REJECT_I_PINCH_FLAG" => 0, "REJECT_E_PINCH_FLAG" => 0, "REJECT_TH_PINCH_FLAG" => 1, "REJECT_EP_PINCH_FLAG" => 0,
    "REJECT_TEARING_FLAG" => 1, "ROTATIONAL_SUPPRESSION_FLAG" => 1, "QL_RATIO_THRESH" => 0.001, "THETA_SQ_THRESH" => 100.0, "Q_SCALE" => 1.0,
    "WRITE_WAVEFUNCTION" => 1, "KY_MODEL" => 2, "SCAN_N" => SCAN_N, "IRS" => 2, "FACTOR_IN_PROFILE" => false, "FACTOR_IN" => 1.0,
    "WIDTH_IN_FLAG" => false, "WIDTH_MIN" => 1.0, "WIDTH_MAX" => 2.0, "INPUT_PROFILE_METHOD" => 2, "N_ION" => 3, "IS_EP" => 3, "REAL_FREQ" => 1)
    
    println("pre case params")
    ini, act = FUSE.case_parameters(:ITER; init_from=:ods);
    println("pre dd")
    dd = IMAS.dd()
    println("dd done")
    @time FUSE.init(dd, ini, act);
    println("checkin")
    FUSE.@checkin :hw_init dd ini act;
    
    println("runTHD")
    runTHD(dd, rho, OptionsDict; printout = true)
    println("runTHD done")
end
println("example done")
