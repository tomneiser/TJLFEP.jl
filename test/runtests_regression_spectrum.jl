# process_in=3 (spectrum mode) regression: assert the Julia γ(ky)/ω(ky) eigenvalue
# spectra for the DIII-D 202017C42 case (single radius ir=50, fixed width=1.5, ky_model=0,
# nbasis=32) match the Fortran TGLF-EP reference produced by TGLFEP_driver.
#
# The reference out.eigenvalue_m{1,2,4} live under test/fixtures/spectrum/ and were
# generated with the same input.gacode + input_spectrum.TGLFEP (archived alongside). See
# the header of test/fixtures/spectrum/README for the exact command.
#
# Fidelity note: process_in=3 does NOT read SCAN_METHOD / PPRIME_METHOD; the Fortran leaves
# those module variables at their zero defaults and Julia mirrors that (SCAN_METHOD->0 = no
# factor-driven EP scaling, PPRIME_METHOD->2). The comparison is therefore apples-to-apples
# on the same input file.

using Test
using TJLFEP

const _SPEC_DIR  = @__DIR__
const _ROOT      = normpath(_SPEC_DIR, "..")
const _CASE_DIR  = joinpath(_ROOT, "examples", "DIIID_202017C42_500ms_v3.1")
const _GACODE    = joinpath(_CASE_DIR, "input.gacode")
const _TGLFEP    = joinpath(_CASE_DIR, "input_spectrum.TGLFEP")
const _FIX_DIR   = joinpath(_SPEC_DIR, "fixtures", "spectrum")

# Parse an out.eigenvalue_m<mode> file: skip the 2 header lines, then read
# `ky  (gamma_n freq_n)*nmodes` per row. Returns (ky::Vector, gamma::Matrix nky×nmodes,
# freq::Matrix nky×nmodes).
function _read_golden_eigenvalue(path::AbstractString)
    lines = readlines(path)
    data = Vector{Vector{Float64}}()
    for line in lines
        toks = split(strip(line))
        length(toks) < 3 && continue
        nums = tryparse.(Float64, toks)
        any(isnothing, nums) && continue          # header / comment lines
        push!(data, Float64[x for x in nums])
    end
    isempty(data) && error("no numeric rows in $path")
    ncol = length(data[1])
    nmodes = (ncol - 1) ÷ 2
    nky = length(data)
    ky = Vector{Float64}(undef, nky)
    gamma = Matrix{Float64}(undef, nky, nmodes)
    freq = Matrix{Float64}(undef, nky, nmodes)
    for (i, row) in enumerate(data)
        ky[i] = row[1]
        for n in 1:nmodes
            gamma[i, n] = row[2n]
            freq[i, n] = row[2n + 1]
        end
    end
    return ky, gamma, freq
end

@testset "process_in=3 spectrum vs Fortran" begin
    @test isfile(_GACODE)
    @test isfile(_TGLFEP)

    golden_files = Dict(m => joinpath(_FIX_DIR, "out.eigenvalue_m$(m)_r040") for m in (1, 2, 4))
    if !all(isfile, values(golden_files))
        @warn "golden eigenvalue fixtures missing; skipping Fortran comparison" _FIX_DIR
        @test_skip false
        return
    end

    out_dir = mktempdir()
    r = run_gacode_scan_task(_GACODE, _TGLFEP, 1; out_dir=out_dir, use_gpu=false, printout=false)
    @test r.spectra !== nothing

    for mode in (1, 2, 4)
        kg, gg, fg = _read_golden_eigenvalue(golden_files[mode])
        s = r.spectra[mode]
        @test length(s.ky) == length(kg)
        @test isapprox(collect(s.ky), kg; rtol=1e-4, atol=1e-6)

        nmodes = min(size(gg, 2), size(s.gamma, 2))
        gj = s.gamma[:, 1:nmodes]
        fj = s.freq[:, 1:nmodes]
        gg_ = gg[:, 1:nmodes]
        fg_ = fg[:, 1:nmodes]

        max_rel_gamma = maximum(abs.(gj .- gg_) ./ (abs.(gg_) .+ 1e-4))
        @info "spectrum mode regression" mode max_rel_gamma

        # The Julia port reproduces the Fortran TGLF transport-model spectrum to ~6-7
        # significant figures (see fixtures/spectrum/README.md). 1% leaves margin for
        # BLAS/eigensolver rounding while still catching any real divergence.
        @test isapprox(gj, gg_; rtol=1e-2, atol=1e-3)
        # Frequency only meaningful where the mode is appreciably unstable; compare there.
        mask = gg_ .> 0.02
        if any(mask)
            @test isapprox(fj[mask], fg_[mask]; rtol=1e-2, atol=5e-3)
        end
    end
end
