# FUSE-free input helpers extracted from the former actor_context.jl.
# Keeping these here (rather than in the FUSE-coupled actor) lets the live
# `runTHD(dd, ...)` IMAS path build a `profile` without TJLFEP depending on FUSE.

"""
    populate_tjlfep_profile!(prof::profile, extraEP::Dict, nr::Int, ns::Int)

Populates TJLFEP `profile` struct from the `extraEP` dictionary containing full
radial data. Sets the multi-radial EP-specific fields plus geometry/physics
fields; species `AS`/`TAUS` are normalized to electrons (ni/ne and Ti/Te).
"""
function populate_tjlfep_profile!(prof::profile, extraEP::Dict, nr::Int, ns::Int)

    # Copy full radial arrays from extraEP to profile struct
    prof.NS = ns
    prof.NR = nr
    prof.RMIN = extraEP["RMIN"]
    prof.omegaGAM = extraEP["omegaGAM"]
    prof.OMEGA_TAE = extraEP["OMEGA_TAE"]
    prof.RHO_STAR = extraEP["RHO_STAR"]
    prof.gammaE = extraEP["gammaE"]
    prof.gammap = extraEP["gammap"]
    prof.IRS = 2  # Standard starting radius index

    # Geometry and physics fields
    prof.SIGN_BT = extraEP["SIGN_BT"]
    prof.SIGN_IT = extraEP["SIGN_IT"]
    prof.GEOMETRY_FLAG = 1
    prof.ROTATION_FLAG = 0  # VPAR set to 0.0 in TJLF_map when ROTATION_FLAG == 0
    prof.RMAJ    = extraEP["RMAJ"]
    prof.SHIFT   = extraEP["SHIFT"]
    prof.Q       = extraEP["Q"]
    prof.SHEAR   = extraEP["SHEAR"]
    prof.Q_PRIME = extraEP["Q_PRIME"]
    prof.P_PRIME = extraEP["P_PRIME"]
    prof.KAPPA   = extraEP["KAPPA"]
    prof.S_KAPPA = extraEP["S_KAPPA"]
    prof.DELTA   = extraEP["DELTA"]
    prof.S_DELTA = extraEP["S_DELTA"]
    prof.ZETA    = extraEP["ZETA"]
    prof.S_ZETA  = extraEP["S_ZETA"]
    prof.BETAE   = extraEP["BETAE"]
    prof.ZEFF    = extraEP["ZEFF"]
    prof.B_UNIT  = extraEP["B_UNIT"]

    # Extract charge numbers from extraEP, which contains all 4 species (e, D, T, EP)
    # extraEP["ZS"] should have length >= 4
    prof.ZS = extraEP["ZS"]
    prof.MASS = extraEP["MASS"]
    prof.N_ION = extraEP["N_ION"]

    # Species data - populate matrices with full radial profiles
    # Profile struct uses [nr, ns] indexing
    # NOTE: AS and TAUS need to be NORMALIZED (ni/ne and Ti/Te)
    # The raw values are in extraEP, so we need to normalize them
    ne_full = extraEP["DENS_1"]  # Electron density
    Te_full = extraEP["TEMP_1"]  # Electron temperature
    # Minor radius in metres: extraEP["RMIN"] is stored in m (rmin_cm / 100 in context.jl).
    # dlnnidr from context.jl is d(ln n)/dr with r in m, so multiply by a [m] to get the
    # dimensionless TGLF input a/Ln.  Matches Fortran: rlns = a_meters * dlnnidr.
    a_m = extraEP["RMIN"][end]

    for s in 1:ns
        dens_key = "DENS_$s"
        temp_key = "TEMP_$s"
        dlnndr_key = "DLNNDR_$s"
        dlntdr_key = "DLNTDR_$s"

        # All species data (including EP) come from extraEP
        if haskey(extraEP, dens_key)
            for ir in 1:nr
                if s == 1
                    # Electrons: AS=1, TAUS=1 by definition
                    prof.AS[ir, s] = 1.0
                    prof.TAUS[ir, s] = 1.0
                else
                    # Ions and EP: normalize to electrons
                    prof.AS[ir, s] = extraEP[dens_key][ir] / ne_full[ir]
                    prof.TAUS[ir, s] = extraEP[temp_key][ir] / Te_full[ir]
                end
                prof.RLNS[ir, s] = extraEP[dlnndr_key][ir] * a_m  # [1/m] × [m] → a/Ln (dimensionless)
                prof.RLTS[ir, s] = extraEP[dlntdr_key][ir] * a_m  # [1/m] × [m] → a/LT (dimensionless)
            end
        end
    end

    return prof
end
