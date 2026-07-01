# Small radial-grid scan + finalize, the standalone-file mirror of FUSE's
# `ActorTJLFEP` smoke test (test/runtests_actors.jl). FUSE runs a SCAN_N=2 /
# N_BASIS=2 ITER scan through the IMAS `dd` path; here we run the equivalent
# through the file/gacode path so the multi-radius orchestration, the
# separatrix edge point, and the α post-processing get exercised in TJLFEP's
# own CI first (bugs caught here rather than downstream in FUSE).
#
# With SCAN_N=2 / IRS=2, ir_exp_from_scan puts the last scan point at ir=NR
# (rho~1, the separatrix), where the TGLF Hermite matrix can go singular — the
# solve must degrade gracefully (finite SFmin) instead of erroring.

using Test
using TJLFEP

@testset "small radial-grid scan + finalize (mirrors FUSE ActorTJLFEP)" begin
    root = normpath(@__DIR__, "..")
    gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
    tgl  = joinpath(@__DIR__, "fixtures", "scan2", "input_scan2_nb2.TGLFEP")
    @test isfile(gac) && isfile(tgl)

    opts, prof, _ = preprocess_gacode_inputs(gac, tgl)
    @test opts.SCAN_N == 2
    @test opts.PROCESS_IN == 5
    @test opts.N_BASIS == 2
    # Last scan point is the separatrix ir=NR (rho~1) — the graceful-degradation case.
    @test opts.IR_EXP[end] == prof.NR

    out = mktempdir()

    # Per-radius array tasks (as a Slurm --array would run them).
    for si in 1:opts.SCAN_N
        r = run_gacode_scan_task(gac, tgl, si; out_dir=out, use_gpu=false,
                                 printout=false, solver=:grid)
        @test r.ir == opts.IR_EXP[si]
        @test isfinite(r.sfmin) && r.sfmin > 0          # separatrix must not error
        @test isfinite(r.width) && r.width > 0
        @test isfile(joinpath(out, "task_$(si).jls"))
    end

    # Merge tasks + build the α critical-gradient profiles.
    width, kymark, SFmin, dpdr, dndr =
        finalize_gacode_scan(gac, tgl, out; printout=true)

    @test length(SFmin) == opts.SCAN_N
    @test all(isfinite, SFmin) && all(SFmin .> 0)
    @test length(width) == opts.SCAN_N

    # α critical-gradient profiles span the full radial grid and are finite.
    @test length(dndr) == prof.NR
    @test length(dpdr) == prof.NR
    @test all(isfinite, dndr) && all(isfinite, dpdr)

    # finalize writes the α inputs and the merged SFmin table.
    @test isfile(joinpath(out, "alpha_dndr_crit.input"))
    @test isfile(joinpath(out, "alpha_dpdr_crit.input"))
    @test isfile(joinpath(out, "sfmin_scan.txt"))

    rm(out; recursive=true, force=true)
end
