#!/usr/bin/env julia
# Bar plot: SCAN_N=20 wall time vs N_BASIS for Fortran / Julia CPU / Julia GPU.
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "collect_scan20_timing.jl"))
using Plots
using Printf

const OUTDIR = joinpath(normpath(@__DIR__, ".."), "timing_runs")

function read_timing_csv(path::String)
    lines = filter(!isempty, strip.(readlines(path)))
    basis = Int[]
    fort, cpu, gpu = Float64[], Float64[], Float64[]
    cpu_ad, gpu_ad, gpu_ad_mps, gpu_truth = Float64[], Float64[], Float64[], Float64[]
    num(p, i) = (length(p) >= i && !isempty(strip(p[i]))) ? parse(Float64, p[i]) : NaN
    for line in lines[2:end]
        p = split(line, ",")
        length(p) < 4 && continue
        push!(basis, parse(Int, p[1]))
        push!(fort, num(p, 2))
        push!(cpu, num(p, 3))
        push!(gpu, num(p, 4))
        # Schema versions: 5 cols (no AD; notes@5), 7 cols (AD@5,6; notes@7),
        # 8 cols (+gpu_ad_mps@7; notes@8), 9 cols (+gpu_truth@8; notes@9). Guard by field
        # count so notes is never parsed as a number.
        push!(cpu_ad,     length(p) >= 7 ? num(p, 5) : NaN)
        push!(gpu_ad,     length(p) >= 7 ? num(p, 6) : NaN)
        push!(gpu_ad_mps, length(p) >= 8 ? num(p, 7) : NaN)
        push!(gpu_truth,  length(p) >= 9 ? num(p, 8) : NaN)
    end
    return basis, fort, cpu, gpu, cpu_ad, gpu_ad, gpu_ad_mps, gpu_truth
end

csv_path, _ = collect_scan20_timing!()
basis, fort, cpu, gpu, cpu_ad, gpu_ad, gpu_ad_mps, gpu_truth = read_timing_csv(csv_path)
fort ./= 60
cpu ./= 60
gpu ./= 60
cpu_ad ./= 60
gpu_ad ./= 60
gpu_ad_mps ./= 60
gpu_truth ./= 60

# The Julia CPU sysimage runs may still be pending; the collector then backfills STALE
# pre-optimization numbers. Set SKIP_JULIA_CPU=1 to omit the Julia CPU series entirely
# (Fortran + Julia GPU only) until the fresh CPU runs complete.
const SHOW_CPU = get(ENV, "SKIP_JULIA_CPU", "0") != "1"

default(legendfontsize=10, guidefontsize=11, tickfontsize=9, dpi=200,
        fontfamily="Computer Modern")
mkpath(OUTDIR)

# AD series (line plot only). Simplified to show ONLY the fastest AD route (GPU in-process
# threads); AD-CPU and AD-GPU+MPS are measured but hidden to keep the plot readable.
const SHOW_AD_CPU = false
const SHOW_AD_GPU = any(!isnan, gpu_ad)
const SHOW_AD_GPU_MPS = false
# Physical-truth (solver=:truth) GPU+MPS series: the production min(robust_ad, truth) profile.
const SHOW_TRUTH = any(!isnan, gpu_truth)

ymax_series = SHOW_CPU ? vcat(fort, cpu, gpu) : vcat(fort, gpu)
SHOW_AD_CPU && (ymax_series = vcat(ymax_series, cpu_ad))
SHOW_AD_GPU && (ymax_series = vcat(ymax_series, gpu_ad))
SHOW_AD_GPU_MPS && (ymax_series = vcat(ymax_series, gpu_ad_mps))
SHOW_TRUTH && (ymax_series = vcat(ymax_series, gpu_truth))
ymax = maximum(filter(!isnan, ymax_series); init=0.0)
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
SHOW_CPU && plot!(p2, basis, cpu; label="Julia grid (10 CPU nodes)", marker=:square, linewidth=2, markersize=6)
plot!(p2, basis, gpu; label="Julia grid MPS (5 GPU nodes)", marker=:diamond, linewidth=2, markersize=6)
SHOW_AD_CPU && plot!(p2, basis, cpu_ad; label="Julia AD (10 CPU nodes)", marker=:utriangle, linewidth=2, markersize=6, linestyle=:dash)
SHOW_AD_GPU && plot!(p2, basis, gpu_ad; label="Julia AD threads (5 GPU nodes)", marker=:star5, linewidth=2, markersize=7, linestyle=:dash)
SHOW_AD_GPU_MPS && plot!(p2, basis, gpu_ad_mps; label="Julia AD GPU+MPS (5 nodes)", marker=:hexagon, linewidth=2, markersize=6, linestyle=:dot)
SHOW_TRUTH && plot!(p2, basis, gpu_truth; label="Julia truth MPS (5 GPU nodes)", marker=:pentagon, linewidth=2, markersize=7, linestyle=:dashdot, color=:purple)
out2 = joinpath(OUTDIR, "scan20_timing_lines.png")
savefig(p2, out2)
println("Wrote ", out2)

println("\nData ($csv_path), times in minutes", SHOW_CPU ? "" : " (Julia CPU omitted: SKIP_JULIA_CPU=1)", ":")
for i in eachindex(basis)
    @printf("  N_BASIS=%d: Fortran=%.2f  JuliaCPU=%s  JuliaGPU=%s  AD-CPU=%s  AD-GPU=%s  AD-GPU+MPS=%s  Truth=%s\n", basis[i], fort[i],
        (!SHOW_CPU || isnan(cpu[i])) ? "—" : @sprintf("%.2f", cpu[i]),
        isnan(gpu[i]) ? "—" : @sprintf("%.2f", gpu[i]),
        isnan(cpu_ad[i]) ? "—" : @sprintf("%.2f", cpu_ad[i]),
        isnan(gpu_ad[i]) ? "—" : @sprintf("%.2f", gpu_ad[i]),
        isnan(gpu_ad_mps[i]) ? "—" : @sprintf("%.2f", gpu_ad_mps[i]),
        isnan(gpu_truth[i]) ? "—" : @sprintf("%.2f", gpu_truth[i]))
end
