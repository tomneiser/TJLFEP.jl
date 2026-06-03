#!/usr/bin/env julia
# Bar plot: SCAN_N=20 wall time vs N_BASIS for Fortran / Julia CPU / Julia GPU.
using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "collect_scan20_timing.jl"))
using Plots
using Printf

const OUTDIR = joinpath(@__DIR__, "timing_runs")

function read_timing_csv(path::String)
    lines = filter(!isempty, strip.(readlines(path)))
    basis = Int[]
    fort, cpu, gpu = Float64[], Float64[], Float64[]
    for line in lines[2:end]
        p = split(line, ",")
        length(p) < 4 && continue
        push!(basis, parse(Int, p[1]))
        push!(fort, isempty(strip(p[2])) ? NaN : parse(Float64, p[2]))
        push!(cpu, isempty(strip(p[3])) ? NaN : parse(Float64, p[3]))
        push!(gpu, isempty(strip(p[4])) ? NaN : parse(Float64, p[4]))
    end
    return basis, fort, cpu, gpu
end

csv_path, _ = collect_scan20_timing!()
basis, fort, cpu, gpu = read_timing_csv(csv_path)
fort ./= 60
cpu ./= 60
gpu ./= 60

# The Julia CPU sysimage runs may still be pending; the collector then backfills STALE
# pre-optimization numbers. Set SKIP_JULIA_CPU=1 to omit the Julia CPU series entirely
# (Fortran + Julia GPU only) until the fresh CPU runs complete.
const SHOW_CPU = get(ENV, "SKIP_JULIA_CPU", "0") != "1"

default(legendfontsize=10, guidefontsize=11, tickfontsize=9, dpi=200,
        fontfamily="Computer Modern")
mkpath(OUTDIR)

ymax = maximum(filter(!isnan, SHOW_CPU ? vcat(fort, cpu, gpu) : vcat(fort, gpu)); init=0.0)
ymax = ymax > 0 ? ymax * 1.12 : 30.0

bw = 0.22
x = Float64.(basis)
p = plot(
    xlim=(minimum(x) - 0.6, maximum(x) + 0.6),
    ylim=(0, ymax),
    xlabel="N_BASIS (basis functions)",
    ylabel="Total wallclock (min)",
    title="SCAN_N=20 radial scan — DIII-D 202017C42_500ms_v3.1",
    legend=:topright,
    xticks=(x, string.(basis)),
)
if SHOW_CPU
    bar!(p, x .- bw, fort; bar_width=bw, label="Fortran (10 CPU nodes)", color=:steelblue)
    bar!(p, x, cpu; bar_width=bw, label="Julia (10 CPU nodes)", color=:darkorange)
    bar!(p, x .+ bw, gpu; bar_width=bw, label="Julia (5 GPU nodes)", color=:seagreen)
else
    # Recenter the two remaining series so there's no empty gap where CPU would be.
    bar!(p, x .- bw/2, fort; bar_width=bw, label="Fortran (10 CPU nodes)", color=:steelblue)
    bar!(p, x .+ bw/2, gpu; bar_width=bw, label="Julia (5 GPU nodes)", color=:seagreen)
end

out_png = joinpath(OUTDIR, "scan20_timing_by_nbasis.png")
savefig(p, out_png)
println("Wrote ", out_png)

p2 = plot(
    basis, fort; label="Fortran (10 CPU nodes)", marker=:circle, linewidth=2, markersize=6,
    xlabel="N_BASIS", ylabel="Total wallclock (min)", title="SCAN_N=20 timing vs N_BASIS",
    xticks=basis, legend=:topleft, ylim=(0, ymax),
)
SHOW_CPU && plot!(p2, basis, cpu; label="Julia (10 CPU nodes)", marker=:square, linewidth=2, markersize=6)
plot!(p2, basis, gpu; label="Julia (5 GPU nodes)", marker=:diamond, linewidth=2, markersize=6)
out2 = joinpath(OUTDIR, "scan20_timing_lines.png")
savefig(p2, out2)
println("Wrote ", out2)

println("\nData ($csv_path), times in minutes", SHOW_CPU ? "" : " (Julia CPU omitted: SKIP_JULIA_CPU=1)", ":")
for i in eachindex(basis)
    @printf("  N_BASIS=%d: Fortran=%.2f  JuliaCPU=%s  JuliaGPU=%s\n", basis[i], fort[i],
        (!SHOW_CPU || isnan(cpu[i])) ? "—" : @sprintf("%.2f", cpu[i]),
        isnan(gpu[i]) ? "—" : @sprintf("%.2f", gpu[i]))
end
