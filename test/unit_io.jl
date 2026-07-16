# Unit tests for the file readers and writers in tjlfep_read_inputs.jl, run on
# the tracked example fixtures (examples/ITER, examples/DIIID_...). These cover
# the input-parsing / preprocessing / save code paths without a TGLF solve.

using Test
using TJLFEP

const _IO_ROOT   = normpath(@__DIR__, "..")
const _ITER_DIR  = joinpath(_IO_ROOT, "examples", "ITER")
const _DIIID_DIR = joinpath(_IO_ROOT, "examples", "DIIID_202017C42_500ms_v3.1")

@testset "file IO on fixtures" begin

    @testset "readMTGLF (ITER)" begin
        f = joinpath(_ITER_DIR, "input.MTGLF")
        @test isfile(f)
        prof, irexp = readMTGLF(f)
        @test prof.NR == 201
        @test prof.NS == 4
        @test size(prof.AS) == (201, 4)
        @test size(prof.ZS) == (201, 4)
        @test prof.SIGN_BT == -1.0
        @test all(!isnan, prof.Q)
        # ZS is broadcast per-species across all radii.
        @test all(prof.ZS[ir, :] == prof.ZS[1, :] for ir in 1:201)
    end

    @testset "readEXPRO (ITER)" begin
        f = joinpath(_ITER_DIR, "input.EXPRO")
        @test isfile(f)
        ni, Ti, dlnnidr, dlntidr, cs, rmin_ex, gammaE, gammap, omegaGAM =
            readEXPRO(f, 2)
        for v in (ni, Ti, dlnnidr, dlntidr, cs, rmin_ex)
            @test length(v) == 201
            @test all(!isnan, v)
        end
        # EP dln n/dr is floored at 1.0 (Fortran TGLFEP_read_EXPRO).
        @test all(dlnnidr .>= 1.0)
        # Out-of-range is_EP fails loudly (was a silent `return 1` sentinel that
        # surfaced as a BoundsError at the caller's tuple destructuring).
        @test_throws ErrorException readEXPRO(f, 9)
    end

    @testset "readTGLFEP (DIII-D scan)" begin
        f = joinpath(_DIIID_DIR, "input_scan20_nb6.TGLFEP")
        @test isfile(f)
        ir_exp = ir_exp_from_scan(101, 2, 20)
        o = readTGLFEP(f, ir_exp)
        @test o.PROCESS_IN == 5
        @test o.N_BASIS == 6
        @test o.SCAN_N == 20
        @test o.IRS == 2
        @test o.KY_MODEL == 2
        @test o.NTOROIDAL == 3          # ky_model != 0 -> 3
        @test o.IR_EXP == ir_exp
        @test length(o.FACTOR) == 20
        @test all(==(o.FACTOR_IN), o.FACTOR)   # FACTOR_IN_PROFILE=false -> constant
    end

    @testset "read_input_profile (DIII-D dump.profile)" begin
        f = joinpath(_DIIID_DIR, "dump.profile")
        @test isfile(f)
        prof = read_input_profile(f)
        @test prof.NR == 101
        @test prof.NS !== missing && prof.NS >= 2
        @test size(prof.AS) == (101, prof.NS)
        @test all(!isnan, prof.Q)
    end

    @testset "profile_from_gacode (DIII-D input.gacode)" begin
        f = joinpath(_DIIID_DIR, "input.gacode")
        @test isfile(f)
        prof = profile_from_gacode(f; is_ep=2, tglfep_nion=2)
        @test prof.NR == 101
        @test prof.NS == 3              # tglfep_nion + 1
        @test prof.GEOMETRY_FLAG == 1
        @test all(prof.AS[:, 1] .== 1.0)   # electrons normalized
        @test all(!isnan, prof.Q)
        @test all(!isnan, prof.BETAE)
        @test all(prof.BETAE .> 0)
        # EP a/Ln floored (>= 0 after * a_m; the raw dln/dr floored at 1.0).
        @test all(isfinite, prof.RLNS[:, 3])
    end

    @testset "expro_vectors_from_gacode + preprocess_gacode_inputs" begin
        gac = joinpath(_DIIID_DIR, "input.gacode")
        tgl = joinpath(_DIIID_DIR, "input_scan20_nb6.TGLFEP")
        opts, prof, expro = preprocess_gacode_inputs(gac, tgl)
        @test opts.SCAN_N == 20
        @test length(opts.IR_EXP) == 20
        @test prof.NR == 101
        @test length(expro.ni) == 101
        @test length(expro.cs) == 101
        @test all(expro.dlnnidr .>= 1.0)
        @test prof.gammaE === expro.gammaE
    end

    @testset "setup_gacode_file_inputs round-trips through readers" begin
        gac = joinpath(_DIIID_DIR, "input.gacode")
        tgl = joinpath(_DIIID_DIR, "input_scan20_nb6.TGLFEP")
        out = mktempdir()
        prof, ir_exp = setup_gacode_file_inputs(gac, out; tglfep_file=tgl)
        @test isfile(joinpath(out, "input.MTGLF"))
        @test isfile(joinpath(out, "input.EXPRO"))
        @test isfile(joinpath(out, "input.TGLFEP"))
        @test length(ir_exp) == 20

        # The written MTGLF re-reads to the same radial Q profile.
        prof2, _ = readMTGLF(joinpath(out, "input.MTGLF"))
        @test prof2.NR == prof.NR
        @test prof2.NS == prof.NS
        @test prof2.Q ≈ prof.Q rtol=1e-10
        rm(out; recursive=true, force=true)
    end

    @testset "setup_fortran_file_inputs" begin
        out = mktempdir()
        prof, ir_exp = setup_fortran_file_inputs(_DIIID_DIR, out;
            tglfep_file=joinpath(_DIIID_DIR, "input_scan20_nb6.TGLFEP"))
        @test isfile(joinpath(out, "input.MTGLF"))
        @test isfile(joinpath(out, "input.EXPRO"))
        @test prof.NR == 101
        @test length(ir_exp) == 20
        rm(out; recursive=true, force=true)
    end

    @testset "save_TGLFEP / save_MTGLF / save_EXPRO round-trip" begin
        gac = joinpath(_DIIID_DIR, "input.gacode")
        tgl = joinpath(_DIIID_DIR, "input_scan20_nb6.TGLFEP")
        opts, prof, _ = preprocess_gacode_inputs(gac, tgl)

        dir = mktempdir()
        # save_all writes all three files.
        save_all(opts, prof, expro_dict_from_profile(prof), dir)
        @test isfile(joinpath(dir, "input.TGLFEP"))
        @test isfile(joinpath(dir, "input.MTGLF"))
        @test isfile(joinpath(dir, "input.EXPRO"))

        # TGLFEP scalars survive the round-trip.
        o2 = readTGLFEP(joinpath(dir, "input.TGLFEP"), opts.IR_EXP)
        @test o2.PROCESS_IN == opts.PROCESS_IN
        @test o2.N_BASIS == opts.N_BASIS
        @test o2.SCAN_N == opts.SCAN_N

        # EXPRO electron density (species 1) is all ones and round-trips exactly.
        # readEXPRO always returns length-201 arrays; compare the populated NR block.
        ni1, = readEXPRO(joinpath(dir, "input.EXPRO"), 1)
        @test ni1[1:prof.NR] ≈ prof.AS[:, 1] rtol=1e-12
        rm(dir; recursive=true, force=true)
    end
end
