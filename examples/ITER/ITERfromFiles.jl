using Revise
using Pkg
using Plots
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using TJLFEP
using TJLFEP: convert_input
using TJLFEP: revert_input
using TJLF
using Base.Threads
using LinearAlgebra
using Dates
Pkg.status()
BLAS.set_num_threads(1)
begin
    tglfepfilepath = joinpath(@__DIR__, "input.TGLFEP")
    mtglffilepath = joinpath(@__DIR__, "input.MTGLF")
    exprofilepath = joinpath(@__DIR__, "input.EXPRO")

    outdir = joinpath(@__DIR__, "fileOutputs")
    mkpath(outdir)

    use_gpu   = (TJLF.pick_device(:auto) === :gpu)
    println("Using device: ", use_gpu ? "GPU" : "CPU")

    cd(outdir) do
        runTHD(tglfepfilepath, mtglffilepath, exprofilepath; printout = true, use_gpu = use_gpu)
    end
    make_crit_grad_plots("FILE"; dir=outdir)
end
println("runTHD finished")
