#!/usr/bin/env julia
# Accuracy comparison at N_BASIS=32 over the 20-radius DIII-D 202017C42_500ms_v3.1 scan:
# sfmin (critical IR factor) vs radial grid index for
#   * grid       — the Fortran-equivalent w>=1 grid solver (cannot resolve narrow-width modes)
#   * robust_ad  — the production truth-tier width-extended solver (reference)
#   * :ad :locate — the width-extended AD solver (multistart locate-grid; bitwise tracks robust_ad)
#   * :ad :wide   — the fast single-pass width-aware AD mode (extend_mode=:wide, wide_kdesc=2)
# Shows that :wide recovers nearly all of robust_ad's narrow-width accuracy that the w>=1 grid misses,
# at a fraction of the cost (see node-hours-vs-nbasis plot), with one residual under-prediction at
# IR=80 and a ~1.8x over-prediction at IR=90.
#   julia --project=. build/ad/plot_ad_wide_nb32.jl
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))
using Plots, Printf

const BUILD = normpath(@__DIR__, "..")
read_sfmin(p) = begin
    ir = Int[]; f = Float64[]
    for ln in eachline(p)
        t = split(strip(ln))
        length(t) >= 3 || continue
        push!(ir, parse(Int, t[2])); push!(f, parse(Float64, t[3]))
    end
    ir, f
end

series = [
    ("grid (Fortran-equiv, w≥1)", joinpath(BUILD, "gacode_nb32_scan20_jgpu_53859063_tasks", "sfmin_scan.txt"),
        :gray55, :dash, :circle),
    ("robust_ad (reference)", joinpath(BUILD, "gacode_nb32_scan20_1node_robust_ad_54638544_tasks", "sfmin_scan.txt"),
        :black, :solid, :diamond),
    (":ad :locate (default)", joinpath(BUILD, "gacode_nb32_scan20_jgpu_ad_54949227_tasks", "sfmin_scan.txt"),
        :seagreen, :solid, :utriangle),
    (":ad :wide (2x faster than :locate)", joinpath(BUILD, "gacode_nb32_scan20_1node_ad_54979250_tasks", "sfmin_scan.txt"),
        :firebrick, :solid, :star5),
    (":ad :only (bare w≥1, fast-turnaround)", joinpath(BUILD, "ad", "ad_threads_sfmin_nb32_ad_only.txt"),
        :darkorange, :dot, :rect),
]

default(legendfontsize=9, guidefontsize=11, tickfontsize=9, dpi=200, fontfamily="Computer Modern")
p = plot(xlabel="radial grid index IR", ylabel="sfmin (critical factor)",
         title="SCAN_N=20 accuracy @ N_BASIS=32 — DIII-D 202017C42_500ms_v3.1",
         yscale=:log10, legend=:topleft, size=(900, 560))
for (lab, path, col, ls, mk) in series
    isfile(path) || (@warn "missing $path"; continue)
    ir, f = read_sfmin(path)
    plot!(p, ir, f; label=lab, color=col, linestyle=ls, marker=mk, markersize=5, linewidth=2)
end
out = joinpath(@__DIR__, "ad_wide_accuracy_nb32.png")
savefig(p, out)
println("Wrote ", out)

# Quick numeric table: ratio of each AD mode to robust_ad.
irg, _ = read_sfmin(series[2][2]); _, rob = read_sfmin(series[2][2])
_, loc = read_sfmin(series[3][2]); _, wid = read_sfmin(series[4][2])
println("\n IR   robust    locate   (loc/rob)    wide    (wide/rob)")
for i in eachindex(irg)
    @printf("%4d  %8.4f  %8.4f  %7.2fx  %8.4f  %7.2fx\n",
        irg[i], rob[i], loc[i], loc[i]/rob[i], wid[i], wid[i]/rob[i])
end
