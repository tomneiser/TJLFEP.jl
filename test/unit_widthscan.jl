# Ballooning-width scan (TJLFEP_ky_widthscan), the WIDTH_IN_FLAG=false branch of
# the PROCESS_IN=3 spectrum path. The spectrum regression pins a fixed width
# (WIDTH_IN_FLAG=true), so the width scan itself is otherwise never exercised in
# CI. We call TJLFEP_ky_widthscan directly on a narrow width grid rather than the
# full PROCESS_IN=3 driver: the 3-mode TM spectrum that follows the scan is
# already covered by the spectrum regression and is far too slow to duplicate,
# whereas the scan + find_max + out.ky_widthscan buffer here run in a few seconds.

using Test
using TJLFEP

@testset "TJLFEP_ky_widthscan (PROCESS_IN=3 auto-width branch)" begin
    root = normpath(@__DIR__, "..")
    gac  = joinpath(root, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
    tgl  = joinpath(@__DIR__, "fixtures", "widthscan", "input_widthscan_nb2.TGLFEP")
    @test isfile(gac) && isfile(tgl)

    opts, prof, expro = preprocess_gacode_inputs(gac, tgl)
    @test opts.PROCESS_IN == 3
    @test opts.WIDTH_IN_FLAG == false
    @test opts.N_BASIS == 2
    @test opts.WIDTH_MIN ≈ 1.50
    @test opts.WIDTH_MAX ≈ 1.60

    TJLFEP._apply_runthd_expro_setup!(opts, prof, expro)

    ep = deepcopy(opts)
    mt = deepcopy(prof)
    ep.IR = ep.IR_EXP[1]
    ep.SUFFIX = "_r" * lpad(string(ep.IR), 3, '0')
    ep.FACTOR_IN = ep.FACTOR[1]
    ep.MODE_IN = 2   # _mainsub_spectrum forces EP-only drive before the scan

    width_in, gmark, fmark, ky_in, (fname, buffer) =
        TJLFEP.TJLFEP_ky_widthscan(ep, mt; use_gpu=false, inner=:threads)

    # Chosen width stays inside the scanned grid; ky is constant across the scan.
    @test opts.WIDTH_MIN <= width_in <= opts.WIDTH_MAX
    @test isfinite(ky_in) && ky_in > 0
    @test isfinite(gmark) && gmark >= 0
    @test isfinite(fmark)

    # out.ky_widthscan_m<mode><suffix> buffer: header lines + one row per width.
    @test fname == "out.ky_widthscan_m2" * ep.SUFFIX
    @test length(buffer) >= 3           # >=2 header lines + at least one width row
    @test occursin("widthscan at ky", buffer[1])
    @test occursin("width,(gamma", buffer[2])
end
