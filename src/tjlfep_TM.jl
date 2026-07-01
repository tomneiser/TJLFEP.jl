# TGLFEP_TM port (process_in=3 spectrum mode).
#
# Computes the gyro-bohm-normalized growth-rate / frequency spectra gamma(ky), freq(ky)
# by running the full TJLF transport model over a fixed ky grid (nky=30, ky=0.15,
# kygrid_model=0, nbasis=32), exactly as the Fortran TGLFEP_TM / write_eigenvalue_spectrum
# do. `mode_in` selects the drive (1: background plasma + EPs, 2: EP-only, 4: ITG/TEM with
# TAE/EPM filtered) via TJLF_map's mode_in override.
#
# Returns (ky, gamma, freq, (filename, buffer)) where gamma/freq are nky x nmodes matrices
# (row i = ky_spectrum[i]) and the buffer reproduces out.eigenvalue_m<mode><suffix>.

const _TM_NKY    = 30
const _TM_KY     = 0.15
const _TM_NBASIS = 32

"""
    TJLFEP_TM(inputsEP::Options, inputsPR::profile; mode_in, use_gpu=false)

TGLFEP spectrum diagnostic (`PROCESS_IN=3`) port. Computes the gyro-Bohm
normalized growth-rate/frequency spectra `γ(ky)`, `ω(ky)` by running the full
TJLF transport model over a fixed `ky` grid (`nky=30`, `ky=0.15`,
`KYGRID_MODEL=0`, `NBASIS=32`), matching the Fortran `TGLFEP_TM` /
`write_eigenvalue_spectrum`.

`mode_in` selects the drive (via [`TJLF_map`](@ref)'s override): `1` = background
plasma + EPs, `2` = EP-only, `4` = ITG/TEM with TAE/EPM filtered. Pass
`use_gpu=true` to run the eigensolves on CUDA.

Returns `(ky, gamma, freq, (filename, buffer))`, where `gamma`/`freq` are
`nky × nmodes` matrices (row `i` = `ky[i]`) and `buffer` reproduces
`out.eigenvalue_m<mode><suffix>`.
"""
function TJLFEP_TM(inputsEP::Options{T}, inputsPR::profile{T}; mode_in::Int,
                   use_gpu::Bool = false) where {T<:Real}
    ky_model = coalesce(inputsEP.KY_MODEL, 0)
    inputTJLF = TJLF_map(inputsEP, inputsPR; mode_in_override=mode_in,
                         ky_model_override=ky_model, nky=_TM_NKY)

    # Full spectral transport model over the fixed Fortran TGLFEP_TM ky grid.
    inputTJLF.USE_TRANSPORT_MODEL = true
    inputTJLF.KYGRID_MODEL = 0
    inputTJLF.KY  = T(_TM_KY)
    inputTJLF.NKY = _TM_NKY

    inputTJLF.NMODES = inputsEP.NMODES

    # Fortran TGLFEP_TM hard-codes nbasis=32 (independent of N_BASIS).
    inputTJLF.NBASIS_MIN = _TM_NBASIS
    inputTJLF.NBASIS_MAX = _TM_NBASIS
    inputTJLF.NXGRID = 32

    inputTJLF.WIDTH = T(inputsEP.WIDTH_IN)
    inputTJLF.FIND_WIDTH = false

    # Same corrections TJLFEP_ky applies (see tjlfep_ky.jl).
    inputTJLF.USE_AVE_ION_GRID = false
    inputTJLF.WIDTH_SPECTRUM .= inputTJLF.WIDTH
    inputTJLF.FIND_EIGEN = true
    inputTJLF.RLNP_CUTOFF = 18.0
    inputTJLF.BETA_LOC = 0.0
    inputTJLF.DAMP_PSI = 0.0
    inputTJLF.DAMP_SIG = 0.0
    inputTJLF.WDIA_TRAPPED = 0.0

    if inputTJLF.SAT_RULE == 2 || inputTJLF.SAT_RULE == 3
        inputTJLF.UNITS = "CGYRO"
        inputTJLF.XNU_MODEL = 3
        inputTJLF.WDIA_TRAPPED = 1.0
    end

    inputTJLF.KX0_LOC = 0.0

    result = TJLF.run(inputTJLF; use_gpu=use_gpu)

    nky = inputTJLF.NKY
    nmodes = inputTJLF.NMODES
    ky = collect(T, inputTJLF.KY_SPECTRUM[1:nky])
    gamma = Matrix{T}(undef, nky, nmodes)
    freq  = Matrix{T}(undef, nky, nmodes)
    for i in 1:nky
        for n in 1:nmodes
            gamma[i, n] = result.eigenvalue[n, i, 1]
            freq[i, n]  = result.eigenvalue[n, i, 2]
        end
    end

    suffix = coalesce(inputsEP.SUFFIX, "")
    filename = "out.eigenvalue_m$(mode_in)" * suffix
    buffer = String[]
    push!(buffer, "gyro-bohm normalized eigenvalue spectra for mode_flag $(mode_in) factor $(inputsEP.FACTOR_IN) width $(inputsEP.WIDTH_IN)")
    push!(buffer, "ky,(gamma(n),freq(n),n=1,nmodes_in)")
    for i in 1:nky
        parts = [@sprintf("%8.4f", ky[i])]
        for n in 1:nmodes
            push!(parts, @sprintf("%12.7f", gamma[i, n]), @sprintf("%12.7f", freq[i, n]))
        end
        push!(buffer, join(parts))
    end

    return ky, gamma, freq, (filename, buffer)
end
