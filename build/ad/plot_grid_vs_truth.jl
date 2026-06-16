# Plot Fortran-faithful grid vs robust_ad vs physical-truth critical scale factor sfmin(radius)
# for DIII-D N_BASIS=32, SCAN_N=20.
#   - grid      : Fortran-equivalent kwscale_scan on the canonical w∈[1,2] box (53859063)
#   - robust_ad : critical_factor_robust, refined (ky,w) grid-zoom of the FAITHFUL onset in w≥1
#                 (refine=1) -- the production FLOOR
#   - truth     : critical_factor_truth, extended-width (ky,w) locate + separable nbasis convergence
#   - prod      : min(robust_ad, truth) == what the production :truth / :triggered policy reports
# robust_ad refines the same w≥1 box as grid (so robust_ad ≤ grid everywhere) and wins the deep core
# where truth's narrow-width sequence is still climbing; truth wins outward, capturing narrow-width
# EP-driven AEs the w≥1 box excludes (sfmin up to ~12× lower at near-marginal radii). The min() of
# the two dominates both grid and robust_ad at every radius and is the trustworthy production value.
#   module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#   julia --project=. build/ad/plot_grid_vs_truth.jl
using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))
using Plots
using Printf

const BUILD      = normpath(@__DIR__, "..")
const GRID_TXT   = get(ENV, "GRID_TXT",
    joinpath(BUILD, "gacode_nb32_scan20_jgpu_53859063_tasks", "sfmin_scan.txt"))
const ROBUST_TXT = get(ENV, "ROBUST_TXT",
    joinpath(@__DIR__, "ad_threads_sfmin_nb32_robust_ad_r1.txt"))
const TRUTH_TXT  = get(ENV, "TRUTH_TXT",
    joinpath(BUILD, "gacode_nb32_scan20_truth_mps_team_54580579_tasks", "sfmin_scan.txt"))
const OUT_PNG    = get(ENV, "OUT_PNG",
    joinpath(normpath(BUILD, ".."), "docs", "plots", "sfmin_grid_vs_truth_nb32.png"))

function read_sfmin(path)
    ir = Int[]; sf = Float64[]
    for line in eachline(path)
        p = split(strip(line))
        length(p) >= 3 || continue
        push!(ir, parse(Int, p[2])); push!(sf, parse(Float64, p[3]))
    end
    return ir, sf
end

function main()
    gir, gsf = read_sfmin(GRID_TXT)
    rir, rsf = read_sfmin(ROBUST_TXT)
    tir, tsf = read_sfmin(TRUTH_TXT)

    rmap = Dict(zip(rir, rsf))
    tmap = Dict(zip(tir, tsf))
    irs  = sort(collect(intersect(keys(rmap), keys(tmap))))
    prod = [min(rmap[ir], tmap[ir]) for ir in irs]

    default(legendfontsize=10, guidefontsize=11, tickfontsize=9, dpi=200,
            fontfamily="Computer Modern")

    plt = plot(gir, gsf; marker=:circle, lw=2, ms=5, ls=:dot,
               label="grid (Fortran-faithful, w≥1)",
               color=:gray55, yscale=:log10,
               xlabel="radial index IR", ylabel="critical scale factor  sfmin",
               title="DIII-D  N_BASIS=32  SCAN_N=20 :  grid / robust_ad / truth",
               legend=:topleft, size=(940, 540))
    plot!(plt, rir, rsf; marker=:circle, lw=2, ms=6,
          label="robust_ad (refined w≥1, floor)", color=:dodgerblue)
    plot!(plt, tir, tsf; marker=:diamond, lw=2, ms=6,
          label="truth (extended w, nbasis-converged)", color=:darkorange)
    plot!(plt, irs, prod; marker=:utriangle, lw=2.5, ms=6, ls=:dash,
          label="production  min(robust_ad, truth)  [:truth / :triggered]", color=:seagreen)

    # where does the production min come from, and how much below robust_ad/grid is it?
    rfloor = count(ir -> rmap[ir] <= tmap[ir], irs)
    ratios = [rmap[ir] / tmap[ir] for ir in irs]      # robust_ad / truth
    order  = sortperm(ratios; rev=true)
    @printf("robust_ad floors at %d/%d radii; truth wins %d.\n", rfloor, length(irs), length(irs)-rfloor)
    @printf("robust_ad/truth ratio: median=%.2fx  max=%.2fx (IR=%d)\n",
            sort(ratios)[cld(length(ratios),2)], maximum(ratios), irs[argmax(ratios)])
    @printf("top-5 truth reductions vs robust_ad: %s\n",
            join([@sprintf("IR%d=%.1fx", irs[i], ratios[i]) for i in order[1:min(5,end)]], "  "))

    mkpath(dirname(OUT_PNG))
    savefig(plt, OUT_PNG)
    println("wrote ", OUT_PNG)
end

main()
