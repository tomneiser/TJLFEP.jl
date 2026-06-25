#!/usr/bin/env julia
# Node-hours vs N_BASIS for the SCAN_N=20 DIII-D scan, adding the fast width-aware :ad :wide mode to
# the established 1-node-backfill series. All series here use the SAME node-hours-minimal layout
# (1 node, 4 GPU-workers draining a 20-radius claim queue, MPS team reused across radii) so the
# comparison is apples-to-apples for NN-database generation cost:
#   * Fortran (reference baseline; its own node count from the CSV)
#   * Julia grid MPS              (julia_gpu_nh)
#   * Julia robust_ad MPS         (julia_gpu_robust_ad_nh) -- production truth-tier width solver
#   * Julia :ad :locate (extended)(julia_gpu_ad_nh)        -- multistart locate-grid width extension
#   * Julia :ad :wide  (kdesc=2)  -- parsed from the WIDE backfill logs written by this work
# :wide trades a small accuracy gap (see ad/ad_wide_accuracy_nb32.png) for materially lower node-hours
# than :locate / robust_ad, with the gap widening at higher N_BASIS (confirms get more expensive).
#   julia --project=. build/timing/plot_scan20_timing_wide.jl
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))
using Plots, Printf

const BUILD = normpath(@__DIR__, "..")
const CSV   = joinpath(BUILD, "timing_runs", "scan20_timing.csv")
const OUT   = joinpath(BUILD, "timing_runs", "scan20_timing_wide_lines.png")

# --- established series from the node-hours CSV (col 1 = n_basis; node-hours block at cols 10..17) ---
num(p, i) = (length(p) >= i && !isempty(strip(p[i]))) ? parse(Float64, p[i]) : NaN
basis = Int[]; fort = Float64[]; grid = Float64[]; locate = Float64[]; robust = Float64[]; truth = Float64[]
for ln in readlines(CSV)[2:end]
    isempty(strip(ln)) && continue
    p = split(ln, ",")
    length(p) < 17 && continue
    push!(basis, parse(Int, p[1]))
    push!(fort,   num(p, 10))   # fortran_nh
    push!(grid,   num(p, 12))   # julia_gpu_nh (grid MPS)
    push!(locate, num(p, 14))   # julia_gpu_ad_nh (:ad :locate, 1-node backfill)
    push!(truth,  num(p, 16))   # julia_gpu_truth_nh
    push!(robust, num(p, 17))   # julia_gpu_robust_ad_nh
end

# --- :wide series: parse node-hours from the 1-node-backfill WIDE logs (scan-phase seconds, nodes=1) ---
function wide_nh(nb::Int)
    pat = Regex("time_scan20_nb$(nb)_julia_gpu_ad_WIDE_[0-9]+\\.out")
    files = sort(filter(f -> occursin(pat, f), readdir(BUILD)))
    isempty(files) && return NaN
    for f in reverse(files)                       # newest job id wins
        for ln in eachline(joinpath(BUILD, f))
            occursin("phase=scan", ln) || continue
            m = match(r"seconds=\s*([0-9.]+)", ln)
            n = match(r"nodes=([0-9]+)", ln)
            m === nothing && continue
            nodes = n === nothing ? 1.0 : parse(Float64, n.captures[1])
            return parse(Float64, m.captures[1]) * nodes / 3600
        end
    end
    return NaN
end
wide = [wide_nh(nb) for nb in basis]

default(legendfontsize=9, guidefontsize=11, tickfontsize=9, dpi=200, fontfamily="Computer Modern")
ys = filter(!isnan, vcat(fort, grid, locate, robust, truth, wide))
ymax = (isempty(ys) ? 1.0 : maximum(ys)) * 1.12
p = plot(xlabel="N_BASIS", ylabel="Node-hours (nodes × wallclock)",
         title="SCAN_N=20 node-hours vs N_BASIS (1-node backfill) — DIII-D",
         xticks=basis, legend=:topleft, ylim=(0, ymax), size=(900, 560))
plot!(p, basis, fort;   label="Fortran",                marker=:circle,   linewidth=2, markersize=6, color=:steelblue)
plot!(p, basis, grid;   label="Julia grid MPS (GPU)",   marker=:diamond,  linewidth=2, markersize=6, color=:seagreen)
plot!(p, basis, robust; label="Julia robust_ad (GPU)",  marker=:star6,    linewidth=2, markersize=7, color=:firebrick, linestyle=:dash)
plot!(p, basis, locate; label="Julia :ad :locate (default)",marker=:utriangle,linewidth=2, markersize=6, color=:purple,    linestyle=:dash)
plot!(p, basis, wide;   label="Julia :ad :wide (2x faster than :locate)", marker=:star5, linewidth=2.5, markersize=8, color=:darkorange)
any(!isnan, truth) && plot!(p, basis, truth; label="Julia truth MPS (GPU)", marker=:pentagon, linewidth=2, markersize=6, color=:gray55, linestyle=:dashdot)
savefig(p, OUT)
println("Wrote ", OUT)

println("\n N_BASIS  Fortran   grid     robust   :locate   :wide   (locate/wide)")
for i in eachindex(basis)
    c(x) = isnan(x) ? "  —   " : @sprintf("%6.3f", x)
    rat = (!isnan(locate[i]) && !isnan(wide[i]) && wide[i] > 0) ? @sprintf("%5.2fx", locate[i]/wide[i]) : "  —  "
    @printf("  %4d   %s  %s  %s  %s  %s   %s\n", basis[i], c(fort[i]), c(grid[i]), c(robust[i]), c(locate[i]), c(wide[i]), rat)
end
