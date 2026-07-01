# Unit tests for the pure numeric / parsing helpers in tjlfep_read_inputs.jl,
# tjlfep_ql_extract.jl, run_tjlfep_file.jl and tjlfep_generate_input.jl.
#
# These are cheap, dependency-free checks (no TGLF solve) that exercise the
# file-parsing and small-math code paths the regression tests never reach.

using Test
using TJLFEP

@testset "pure helpers" begin

    @testset "ir_exp_from_scan" begin
        # scan_n == 1 returns the single start index.
        @test ir_exp_from_scan(201, 2, 1) == [2]
        # Evenly spaced from irs to nr, first == irs, last == nr.
        v = ir_exp_from_scan(201, 2, 20)
        @test length(v) == 20
        @test v[1] == 2
        @test v[end] == 201
        @test issorted(v)
        # Exact spacing formula for a small case.
        @test ir_exp_from_scan(11, 1, 3) == [1, 6, 11]
    end

    @testset "expro_bound_deriv (exact for quadratics)" begin
        r = collect(0.0:0.25:2.0)
        # f = 3r^2 + 2r + 1  ->  f' = 6r + 2  (3-pt Lagrange derivative is exact).
        f = @. 3r^2 + 2r + 1
        df = expro_bound_deriv(f, r)
        @test df ≈ (6 .* r .+ 2) rtol=1e-10
        # Linear function -> constant slope everywhere (incl. endpoints).
        g = @. 4r - 7
        @test expro_bound_deriv(g, r) ≈ fill(4.0, length(r)) rtol=1e-10
        # Length mismatch throws.
        @test_throws ErrorException expro_bound_deriv([1.0, 2.0], [1.0])
    end

    @testset "expro_log_gradients" begin
        r = collect(0.0:0.1:2.0)
        # n = exp(-2r), T = exp(-3r)  ->  -dln n/dr = 2, -dln T/dr = 3.
        ni = @. exp(-2r)
        ti = @. exp(-3r)
        dlnn, dlnt = expro_log_gradients(ni, ti, r)
        @test dlnn ≈ fill(2.0, length(r)) rtol=1e-8
        @test dlnt ≈ fill(3.0, length(r)) rtol=1e-8
        @test_throws ErrorException expro_log_gradients([1.0, 2.0], [1.0, 2.0], [1.0])
    end

    @testset "expro_species_for_gacode_is_ep" begin
        @test expro_species_for_gacode_is_ep(1) == 2
        @test expro_species_for_gacode_is_ep(3) == 4
    end

    @testset "gacode header parsers" begin
        lines = [
            "# nexp", "3",
            "# nion", "2",
            "# masse", " 5.44e-4",
            "# mass", " 2.0 3.0 12.0",
            "# z", " 1.0 1.0 6.0",
        ]
        @test TJLFEP._read_gacode_header_int(lines, "nexp") == 3
        @test TJLFEP._read_gacode_header_int(lines, "nion") == 2
        @test TJLFEP._read_gacode_header_float(lines, "masse") ≈ 5.44e-4
        @test TJLFEP._read_gacode_header_vector(lines, "mass", 2) == [2.0, 3.0]
        @test TJLFEP._read_gacode_header_vector(lines, "z", 3) == [1.0, 1.0, 6.0]
        @test_throws ErrorException TJLFEP._read_gacode_header_int(lines, "nope")
    end

    @testset "gacode block field readers" begin
        path, io = mktemp()
        write(io, """
        # rmin | m
           1  0.10
           2  0.20
           3  0.30
        # ni | 10^19/m^3
           1  8.0 2.0
           2  7.0 1.5
           3  6.0 1.0
        # te | keV
           1  5.0
           2  4.0
           3  3.0
        """)
        close(io)
        @test read_gacode_scalar_field(path, "rmin", 3) == [0.10, 0.20, 0.30]
        @test read_gacode_scalar_field(path, "te", 3) == [5.0, 4.0, 3.0]
        # ion_index selects the column (1-based ion index).
        @test read_gacode_ion_field(path, "ni", 1, 3) == [8.0, 7.0, 6.0]
        @test read_gacode_ion_field(path, "ni", 2, 3) == [2.0, 1.5, 1.0]
        # Missing block / incomplete data error out.
        @test_throws ErrorException read_gacode_scalar_field(path, "bogus", 3)
        @test_throws ErrorException read_gacode_scalar_field(path, "rmin", 4)
        rm(path; force=true)
    end

    @testset "slurm_array_task_id (ENV precedence)" begin
        saved = Dict(k => get(ENV, k, nothing)
                     for k in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASKID", "SLURM_PROCID"))
        try
            for k in keys(saved); delete!(ENV, k); end
            @test slurm_array_task_id() == 0            # nothing set -> 0
            ENV["SLURM_PROCID"] = "7"
            @test slurm_array_task_id() == 7            # falls back to PROCID
            ENV["SLURM_ARRAY_TASK_ID"] = "3"
            @test slurm_array_task_id() == 3            # array id takes precedence
        finally
            for (k, v) in saved
                v === nothing ? delete!(ENV, k) : (ENV[k] = v)
            end
        end
    end

    @testset "readline_values" begin
        # Scalar line -> both returns equal.
        @test TJLFEP.readline_values(IOBuffer("1.5\n"), 0) == (1.5, 1.5)
        # Tuple "(lo,hi)" line -> the pair.
        @test TJLFEP.readline_values(IOBuffer("(1.0,2.0)\n"), 0) == (1.0, 2.0)
    end

    @testset "QL scan helpers" begin
        # _default_flux_scan_factors: [0.5, 1.0, 1.5]*f0 clamped to [0.01, fmax].
        @test TJLFEP._default_flux_scan_factors(0.2, 10.0) ≈ [0.1, 0.2, 0.3]
        # f0 floored at 0.05 when fmark is tiny.
        @test TJLFEP._default_flux_scan_factors(0.0, 10.0) ≈ [0.025, 0.05, 0.075]
        # Upper clamp at fmax.
        f = TJLFEP._default_flux_scan_factors(10.0, 2.0)
        @test all(f .<= 2.0)

        # _slope_linear: exact least-squares slope of a line.
        x = collect(1.0:5.0)
        @test TJLFEP._slope_linear(2 .* x .+ 3, x) ≈ 2.0
        @test TJLFEP._slope_linear([1.0], [1.0]) == 0.0        # < 2 points
        @test TJLFEP._slope_linear([1.0, 1.0], [2.0, 2.0]) == 0.0  # zero variance in x
    end
end
