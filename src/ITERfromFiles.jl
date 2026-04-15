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

    
    tglfepfilepath = homedirectory*"/ITERfiles/input.TGLFEP"
    mtglffilepath = homedirectory*"/ITERfiles/input.MTGLF"
    exprofilepath = homedirectory*"/ITERfiles/input.EXPRO"  

    outdir = joinpath(homedirectory, "fileOutputs")
    mkpath(outdir)
    cd(outdir) do
        runTHD(tglfepfilepath, mtglffilepath, exprofilepath, printout = true)
    end
    make_crit_grad_plots("FILE"; dir=outdir)
end
println("runTHD finished")
