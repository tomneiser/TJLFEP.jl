# Extra solve-free coverage: the input generator (TJLFEP_generate_input) and the
# marginal-QL / ALPHA-mode helpers in tjlfep_ql_extract.jl.

using Test
using TJLFEP

@testset "extra helpers" begin

    @testset "TJLFEP_generate_input" begin
        fix = joinpath(@__DIR__, "fixtures", "generate", "input.TJLFEP.generate")
        @test isfile(fix)
        tmp = mktempdir()
        cp(fix, joinpath(tmp, "input.TJLFEP.generate"))
        # Reads input.TJLFEP.generate from CWD and builds the random profile arrays.
        cd(tmp) do
            @test (TJLFEP_generate_input(); true)
        end
        rm(tmp; recursive=true, force=true)
    end

    @testset "_island_shift / _nearest_scan_index (pure)" begin
        @test TJLFEP._island_shift(1) == 0
        @test TJLFEP._island_shift(2) == 20
        @test TJLFEP._island_shift(3) == 5
        @test TJLFEP._island_shift(6) == 0        # beyond table -> 0

        rho_scan = [0.4, 0.6]
        @test TJLFEP._nearest_scan_index(0.41, rho_scan) == 1
        @test TJLFEP._nearest_scan_index(0.55, rho_scan) == 2
        @test TJLFEP._nearest_scan_index(0.6, rho_scan) == 2
    end

    @testset "build_alpha_ql_modes" begin
        mq = MarginalQLData{Float64}(
            fill(0.5, 5), fill(2.0, 5), fill(0.1, 5),
            0.3, 1.2, 1.5, true, Float64[],
        )
        marginals = [mq, nothing]           # second scan point unstable-less
        rho_scan = [0.4, 0.6]
        rho_grid = collect(0.0:0.25:1.0)    # 5 radial points
        dndr_crit = fill(1.0, 5)

        modes = build_alpha_ql_modes(marginals, rho_scan, rho_grid, dndr_crit; km_max=5)
        @test length(modes) == 5
        @test length(modes[1].gamma_star) == length(rho_grid)
        @test length(modes[1].diff_star) == length(rho_grid)
        @test modes[1].crit_index_shift == 0        # km==1 -> no island shift
        @test modes[2].crit_index_shift == 20       # _island_shift(2)
        @test all(isfinite, modes[3].gamma_star)
        # Points nearest the `nothing` scan index fall back to the (0.1, 1.0) defaults.
        @test all(modes[1].diff_star .>= 0)
    end

    @testset "extract_marginal_ql (fallback diffusivity path)" begin
        ep = Options{Float64}(1, false, 5, 1, 1, 1)
        ep.IR = 1
        ep.IS_EP = 1
        ep.NMODES = 1
        ep.F_REAL = [100.0]
        ep.FACTOR_IN = 1.5
        ep.KYMARK = 0.3
        ep.WIDTH_IN = 1.2
        ep.WIDTH_MIN = 1.0

        pr = profile{Float64}(1, 3)
        pr.RMIN = [0.5]
        pr.RLNS = fill(2.0, 1, 3)
        pr.AS = fill(0.5, 1, 3)
        pr.RHO_STAR = [0.01]
        pr.RMAJ = [3.0]
        pr.B_UNIT = [2.5]

        nfactor = 2
        growth = zeros(1, 1, nfactor, 1); growth[1, 1, 1, 1] = 0.4
        dep = zeros(1, 1, nfactor, 1); dep[1, 1, 1, 1] = 0.2; dep[1, 1, 2, 1] = 0.5
        factor = [1.0, 2.0]
        imark = fill(1, 1, 1)

        # Unstable branch (imark_min <= nfactor).
        mq = extract_marginal_ql(ep, pr, growth, dep, factor;
            imark=imark, ikyhat_mark=1, iefwid_mark=1, imark_min=1,
            nkyhat=1, nefwid=1, nfactor=nfactor, use_gpu=false, use_flux_scan=false)
        @test mq isa MarginalQLData
        @test length(mq.gamma_star) == 1
        @test mq.unstable == true
        @test mq.kymark == 0.3
        @test all(isfinite, mq.diff_star)
        @test mq.gamma_star[1] ≈ 100.0 * 0.4    # F_REAL * growthrate

        # Stable branch (imark_min > nfactor) -> width defaults to WIDTH_MIN, kymark 0.
        mq2 = extract_marginal_ql(ep, pr, growth, dep, factor;
            imark=imark, ikyhat_mark=1, iefwid_mark=1, imark_min=nfactor + 1,
            nkyhat=1, nefwid=1, nfactor=nfactor, use_gpu=false, use_flux_scan=false)
        @test mq2.unstable == false
        @test mq2.kymark == 0.0
        @test mq2.width == 1.0
    end

    @testset "mainsub PROCESS_IN no-op branches + solver guard" begin
        ep = Options{Float64}(1, false, 5, 1, 1, 1)
        pr = profile{Float64}(1, 1)
        # PROCESS_IN 1/2/4/6 are not-implemented stubs that just return "No".
        for pin in (1, 2, 4, 6)
            ep.PROCESS_IN = pin
            @test TJLFEP.mainsub(ep, pr, false) == "No"
        end
        # Unknown solver is rejected before any dispatch.
        ep.PROCESS_IN = 5
        @test_throws ErrorException TJLFEP.mainsub(ep, pr, false; solver=:bogus)
    end

    @testset "slurm_array_task_id (env parsing)" begin
        saved = Dict(k => get(ENV, k, nothing)
                     for k in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASKID", "SLURM_PROCID"))
        try
            for k in keys(saved); delete!(ENV, k); end
            @test TJLFEP.slurm_array_task_id() == 0          # nothing set -> 0

            ENV["SLURM_PROCID"] = "7"
            @test TJLFEP.slurm_array_task_id() == 7          # falls back to PROCID

            ENV["SLURM_ARRAY_TASK_ID"] = "3"
            @test TJLFEP.slurm_array_task_id() == 3          # array id takes precedence
        finally
            for (k, v) in saved
                v === nothing ? delete!(ENV, k) : (ENV[k] = v)
            end
        end
    end

    @testset "_resolve_runTHD_parallel / _unpack_mainsub!" begin
        @test TJLFEP._resolve_runTHD_parallel(:threads) == :threads
        @test TJLFEP._resolve_runTHD_parallel(:distributed) == :distributed
        # :auto resolves to :threads on a single-worker process.
        @test TJLFEP._resolve_runTHD_parallel(:auto) == :threads

        ret = ((:g, :ep, :mt, :mql), (:sf, :wf))
        growth, ep, mt, mql = TJLFEP._unpack_mainsub!(ret)
        @test (growth, ep, mt, mql) == (:g, :ep, :mt, :mql)
    end

    @testset "_write_radius_buffers" begin
        tmp = mktempdir()
        # nothing / empty -> no files written.
        TJLFEP._write_radius_buffers(nothing, "_r001"; out_dir=tmp)
        TJLFEP._write_radius_buffers((nothing, nothing), "_r001"; out_dir=tmp)
        @test isempty(readdir(tmp))

        # scalefactor buffer + one wavefunction file.
        sf = ["line1", "line2"]
        wf = [("out.wavefunction_r001", ["a", "b"]), ("empty", String[])]
        TJLFEP._write_radius_buffers((sf, wf), "_r001"; out_dir=tmp)
        @test isfile(joinpath(tmp, "out.scalefactor_r001"))
        @test isfile(joinpath(tmp, "out.wavefunction_r001"))
        @test !isfile(joinpath(tmp, "empty"))           # empty buffer is skipped
        @test readlines(joinpath(tmp, "out.scalefactor_r001")) == sf
        rm(tmp; recursive=true, force=true)
    end
end
