# Coverage for tjlfep_ad_extensions.jl: pure helpers plus a single-radius end-to-end
# :ad solve. The solve itself is fast (~seconds); the cost is one-time AD/ForwardDiff
# compilation, amortized by the CI depot cache. Uses the cheapest AD path
# (AD_EXTEND_MODE=:only, bare w>=1 box) to keep the wallclock modest.

using Test
using TJLFEP

@testset "AD extensions" begin

    @testset "_nbasis_extrapolate (pure)" begin
        # No finite samples -> NaN limit, not converged.
        r0 = TJLFEP._nbasis_extrapolate(Int[], Float64[])
        @test isnan(r0.limit) && r0.converged == false

        # Fewer than 3 samples -> return the finest measured value.
        r1 = TJLFEP._nbasis_extrapolate([6, 8], [1.0, 1.1])
        @test r1.limit == 1.1
        @test r1.nb_finest == 8
        @test r1.converged == false

        # Flat tail -> converged at the finest value.
        r2 = TJLFEP._nbasis_extrapolate([6, 8, 16], [1.0, 1.0, 1.0])
        @test r2.converged == true
        @test r2.limit == 1.0

        # Geometric shrink (ratio 0.5) -> Richardson limit, clamped to the band.
        r3 = TJLFEP._nbasis_extrapolate([6, 8, 16], [1.0, 1.5, 1.75])
        @test r3.converged == true
        @test r3.ratio ≈ 0.5
        @test r3.limit ≈ 2.0
    end

    @testset "_clamp_to / _knob_label (pure)" begin
        @test TJLFEP._clamp_to(5.0, 0.0, 3.0) == 3.0
        @test TJLFEP._clamp_to(-1.0, 0.0, 3.0) == 0.0
        @test TJLFEP._clamp_to(1.5, 0.0, 3.0) == 1.5

        @test TJLFEP._knob_label(:RLNS, nothing, 3) == "RLNS"
        @test TJLFEP._knob_label(:RLNS, 1, 3) == "RLNS[e]"
        @test TJLFEP._knob_label(:RLNS, 3, 3) == "RLNS[EP]"
        @test TJLFEP._knob_label(:RLNS, 2, 3) == "RLNS[i1]"
    end

    @testset "single-radius :ad solve (end-to-end)" begin
        root = normpath(@__DIR__, "..")
        cd = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1")
        gac = joinpath(cd, "input.gacode")
        tgl = joinpath(cd, "input_scan20_nb6.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        saved = get(ENV, "AD_EXTEND_MODE", nothing)
        ENV["AD_EXTEND_MODE"] = "only"      # cheapest AD path (bare w>=1 box)
        try
            out = mktempdir()
            r = run_gacode_scan_task(gac, tgl, 2; out_dir=out, use_gpu=false,
                                     printout=false, solver=:ad)
            @test r.ir == 7
            @test isfinite(r.sfmin) && r.sfmin > 0
            @test isfinite(r.width) && r.width > 0
            rm(out; recursive=true, force=true)
        finally
            saved === nothing ? delete!(ENV, "AD_EXTEND_MODE") : (ENV["AD_EXTEND_MODE"] = saved)
        end
    end

    # Remaining production solver tiers on CPU at the cheapest setting (nb=2,
    # single radius). The one-time ForwardDiff compile is amortized by the :only
    # solve above, so each tier here adds only a few seconds. This covers the
    # width-extension branches of _mainsub_ad (:wide, :locate) and the otherwise
    # 0%-covered _mainsub_robust_ad path.
    @testset "AD tiers: :ad :wide / :ad :locate / :robust_ad (nb=2)" begin
        root = normpath(@__DIR__, "..")
        gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
        tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
        @test isfile(gac) && isfile(tgl)

        saved = get(ENV, "AD_EXTEND_MODE", nothing)
        try
            for mode in ("wide", "locate")
                ENV["AD_EXTEND_MODE"] = mode
                out = mktempdir()
                r = run_gacode_scan_task(gac, tgl, 1; out_dir=out, use_gpu=false,
                                         printout=false, solver=:ad)
                @test r.ir == 2
                @test isfinite(r.sfmin) && r.sfmin > 0
                @test isfinite(r.width) && r.width > 0
                rm(out; recursive=true, force=true)
            end
        finally
            saved === nothing ? delete!(ENV, "AD_EXTEND_MODE") : (ENV["AD_EXTEND_MODE"] = saved)
        end

        # Production adaptive-refinement solver (its own mainsub branch).
        out = mktempdir()
        r = run_gacode_scan_task(gac, tgl, 1; out_dir=out, use_gpu=false,
                                 printout=false, solver=:robust_ad, refine_rounds=1)
        @test r.ir == 2
        @test isfinite(r.sfmin) && r.sfmin > 0
        @test isfinite(r.width) && r.width > 0
        rm(out; recursive=true, force=true)
    end
end
