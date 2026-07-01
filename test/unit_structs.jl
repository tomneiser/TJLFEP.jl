# Unit tests for the Options/profile structs and the struct-consuming helpers
# (populate_tjlfep_profile!, expro_dict_from_profile, tjlfep_complete_output,
# diff_star_from_D_W) — all exercised on tiny in-memory inputs (no TGLF solve).

using Test
using TJLFEP

@testset "structs and struct helpers" begin

    @testset "Options constructor" begin
        o = Options{Float64}(20, false, 5, 201, 1, 4)
        @test o.SCAN_N == 20
        @test o.NMODES == 4
        @test o.NN == 5
        @test length(o.FACTOR_OUT) == 5
        @test length(o.IR_EXP) == 20
        @test all(==(0), o.IR_EXP)
        @test length(o.LKEEP) == 4
        @test o.WIDTH_IN_FLAG == false
        @test o.WIDTH_IN == 0.0          # widthin=false -> 0.0 sentinel
        @test o.REAL_FREQ == 0

        # widthin=true leaves WIDTH_IN as NaN (unset).
        ow = Options{Float64}(5, true, 5, 201, 1, 4)
        @test ow.WIDTH_IN_FLAG == true
        @test isnan(ow.WIDTH_IN)
    end

    @testset "profile constructor" begin
        p = profile{Float64}(201, 4)
        @test size(p.AS) == (201, 4)
        @test size(p.TAUS) == (201, 4)
        @test size(p.RLNS) == (201, 4)
        @test length(p.MASS) == 4
        @test length(p.RMIN) == 201
        @test length(p.Q) == 201
        @test p.GEOMETRY_FLAG == 1
        @test all(isnan, p.RMIN)                # radial vectors seeded NaN
        @test all(x -> x == 1.0e-7, p.gammaE)   # gammaE seeded to 1e-7
    end

    @testset "populate_tjlfep_profile!" begin
        nr, ns = 3, 2
        p = profile{Float64}(nr, ns)
        extraEP = Dict{String,Any}(
            "RMIN" => [0.1, 0.2, 0.3],
            "omegaGAM" => fill(0.5, nr), "OMEGA_TAE" => fill(0.6, nr),
            "RHO_STAR" => fill(1e-3, nr), "gammaE" => zeros(nr), "gammap" => zeros(nr),
            "SIGN_BT" => -1.0, "SIGN_IT" => 1.0,
            "RMAJ" => fill(1.7, nr), "SHIFT" => zeros(nr), "Q" => fill(2.0, nr),
            "SHEAR" => fill(1.0, nr), "Q_PRIME" => fill(3.0, nr), "P_PRIME" => fill(-0.1, nr),
            "KAPPA" => fill(1.5, nr), "S_KAPPA" => zeros(nr), "DELTA" => fill(0.2, nr),
            "S_DELTA" => zeros(nr), "ZETA" => zeros(nr), "S_ZETA" => zeros(nr),
            "BETAE" => fill(0.01, nr), "ZEFF" => fill(1.8, nr), "B_UNIT" => fill(2.5, nr),
            "ZS" => fill(1.0, nr, ns), "MASS" => [1.0, 2.0], "N_ION" => 1,
            "DENS_1" => [1.0, 1.0, 1.0], "TEMP_1" => [2.0, 2.0, 2.0],
            "DLNNDR_1" => [1.0, 1.0, 1.0], "DLNTDR_1" => [1.0, 1.0, 1.0],
            "DENS_2" => [0.5, 0.5, 0.5], "TEMP_2" => [1.0, 1.0, 1.0],
            "DLNNDR_2" => [2.0, 2.0, 2.0], "DLNTDR_2" => [3.0, 3.0, 3.0],
        )
        populate_tjlfep_profile!(p, extraEP, nr, ns)

        @test p.NS == ns && p.NR == nr
        @test p.IRS == 2
        @test p.SIGN_BT == -1.0 && p.SIGN_IT == 1.0
        # Electrons normalized to 1.
        @test all(==(1.0), p.AS[:, 1]) && all(==(1.0), p.TAUS[:, 1])
        # Species 2 normalized to electrons (ni/ne, Ti/Te).
        @test all(==(0.5), p.AS[:, 2]) && all(==(0.5), p.TAUS[:, 2])
        # a/Ln = dln n/dr * a_m with a_m = RMIN[end] = 0.3.
        @test p.RLNS[1, 2] ≈ 2.0 * 0.3
        @test p.RLTS[1, 2] ≈ 3.0 * 0.3
    end

    @testset "expro_dict_from_profile" begin
        nr, ns = 3, 2
        p = profile{Float64}(nr, ns)
        p.NR = nr; p.NS = ns
        p.RMIN = [0.1, 0.2, 0.3]
        p.AS[:, 1] .= 1.0; p.AS[:, 2] .= 0.5
        p.TAUS[:, 1] .= 1.0; p.TAUS[:, 2] .= 0.4
        p.RLNS[:, 1] .= 0.6; p.RLNS[:, 2] .= 0.9
        p.RLTS[:, 1] .= 0.3; p.RLTS[:, 2] .= 0.6
        p.gammaE = zeros(nr); p.gammap = zeros(nr); p.omegaGAM = fill(0.5, nr)

        d = expro_dict_from_profile(p)
        @test d["NR"] == nr && d["NS"] == ns
        @test d["DENS_2"] == p.AS[:, 2]
        @test d["TEMP_2"] == p.TAUS[:, 2]
        @test d["DLNNDR_2"] ≈ p.RLNS[:, 2] ./ 0.3    # divided by a_m
        @test d["DLNTDR_1"] ≈ p.RLTS[:, 1] ./ 0.3
        @test d["CS"] == fill(1.0e7, nr)
    end

    @testset "tjlfep_complete_output (interpolation)" begin
        o = Options{Float64}(3, false, 5, 10, 1, 4)
        o.IRS = 2
        o.INPUT_PROFILE_METHOD = 2
        o.IR_EXP = [2, 5, 8]
        p = profile{Float64}(10, 2)
        p.NR = 10
        p.RMIN = collect(1.0:10.0)

        pin = [1.0, 2.0, 3.0]
        _, pout, ir_min, ir_max, l_accept = tjlfep_complete_output(pin, o, p)
        @test length(pout) == 10
        @test ir_min == 2 && ir_max == 8
        @test all(l_accept)
        # Scanned radii keep their values; below/above the range are held flat.
        @test pout[2] ≈ 1.0
        @test pout[5] ≈ 2.0
        @test pout[8] ≈ 3.0
        @test pout[1] ≈ 1.0
        @test pout[10] ≈ 3.0
        # Linear interpolation between scanned radii (uniform RMIN grid).
        @test pout[3] ≈ 1.0 + 1 / 3
        @test pout[4] ≈ 1.0 + 2 / 3

        # A rejected (negative/NaN) scan point flips l_accept.
        _, _, _, _, l2 = tjlfep_complete_output([1.0, -1.0, 3.0], o, p)
        @test l2 == [true, false, true]
    end

    @testset "diff_star_from_D_W" begin
        ep = Options{Float64}(1, false, 5, 1, 1, 1)
        ep.IR = 1
        ep.IS_EP = 1          # EP species slot = IS_EP + 1 = 2
        ep.NMODES = 1
        ep.F_REAL = [100.0]

        pr = profile{Float64}(1, 3)
        pr.RMIN = [0.5]
        pr.RLNS = fill(2.0, 1, 3)
        pr.AS = fill(0.5, 1, 3)
        pr.RHO_STAR = [0.01]
        pr.RMAJ = [3.0]

        ep_flux = reshape([1.0, 2.0, 3.0], 3, 1)   # linear in factor
        factors = [0.5, 1.0, 1.5]
        diff = diff_star_from_D_W(ep_flux, factors, ep, pr)
        @test length(diff) == 1
        @test isfinite(diff[1])
        @test diff[1] >= 0.0
        # Flat flux -> zero slope -> zero diffusivity.
        diff0 = diff_star_from_D_W(fill(1.0, 3, 1), factors, ep, pr)
        @test diff0[1] == 0.0
    end
end
