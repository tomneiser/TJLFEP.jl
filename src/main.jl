using Revise
using Pkg
using Plots
Pkg.activate("..")
include("TJLFEP.jl")
# include("../../TJLF/src/TJLF.jl")
using .TJLFEP
using .TJLFEP: convert_input
using .TJLFEP: revert_input
# using .TJLF
using TJLF
using Base.Threads
using LinearAlgebra
using Dates
Pkg.status()
BLAS.set_num_threads(1)
begin
    homedirectory = pwd()

    
    tglfepfilepath = homedirectory*"/../tests/isEP3v6/input.TGLFEP"
    mtglffilepath = homedirectory*"/../tests/isEP3v6/input.MTGLF"
    exprofilepath = homedirectory*"/../tests/isEP3v6/input.EXPRO"  

    outdir = joinpath(@__DIR__, "mainOut")
    mkpath(outdir)
    cd(outdir) do
        runTHD(tglfepfilepath, mtglffilepath, exprofilepath, printout = true)
    end
    make_crit_grad_plots("main"; dir=outdir)
end
println("runTHD finished")