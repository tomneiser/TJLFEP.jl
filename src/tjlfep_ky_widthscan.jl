# TGLFEP_ky_widthscan port (used by process_in=3 when WIDTH_IN_FLAG=false).
#
# Scans the single-ky growth rate / frequency over a width grid
# (width_min : delta_width : width_max, delta_width=0.01) using TJLFEP_ky at the input
# ky_model and mode_in (the Fortran sets mode_in=2, EP-only, before the scan), then picks
# the width that maximizes the kept-mode growth rate (find_max). Returns that width plus
# the ky used (ky_in, constant across the scan -> kymark) and the out.ky_widthscan_m<mode>
# buffer.
#
# NOTE: the Fortran also writes out.wscan_ql_weight_m<mode> (the full per-width QL weight
# tensor). TJLFEP_ky does not surface that tensor, so it is not reproduced here; the width
# determination and gamma/freq-vs-width spectrum (the primary products) are.

const _WIDTHSCAN_DELTA = 0.01

# One width point of the scan. Pure w.r.t. shared state (deepcopies inputsEP), so it is
# safe under the threaded / MPS-team dispatch.
function _widthscan_combo(i::Int, widths::AbstractVector{T},
                          inputsEP::Options{T}, inputsPR::profile{T};
                          use_gpu::Bool = false) where {T<:Real}
    local_inputsEP = deepcopy(inputsEP)
    local_inputsEP.WIDTH_IN = widths[i]

    gamma_out, freq_out, inputTJLF, _, _, _, _ =
        TJLFEP_ky(local_inputsEP, inputsPR, "", 0; eigen_cache=nothing, use_gpu=use_gpu,
                  mode_in_override=local_inputsEP.MODE_IN,
                  ky_model_override=coalesce(local_inputsEP.KY_MODEL, 0))

    NM = local_inputsEP.NMODES
    return (
        i = i,
        gamma_out = gamma_out[1:NM],
        freq_out  = freq_out[1:NM],
        LKEEP     = local_inputsEP.LKEEP[1:NM],
        ky        = inputTJLF.KY,
    )
end

function TJLFEP_ky_widthscan(inputsEP::Options{T}, inputsPR::profile{T};
                             use_gpu::Bool = false, inner::Symbol = :threads,
                             team::Union{Nothing,AbstractVector{<:Integer}} = nothing) where {T<:Real}
    width_min = T(inputsEP.WIDTH_MIN)
    width_max = T(inputsEP.WIDTH_MAX)
    delta_width = T(_WIDTHSCAN_DELTA)
    nwidth = round(Int, (width_max - width_min) / delta_width)
    nwidth = max(nwidth, 1)

    widths = T[width_min + delta_width * (i - 1) for i in 1:nwidth]
    NM = inputsEP.NMODES

    growthrate = zeros(T, nwidth, NM)
    frequency  = zeros(T, nwidth, NM)
    lkeep_i    = fill(false, nwidth, NM)
    ky_in = T(NaN)

    local results::Vector{Any}
    if inner === :mps_team && team !== nothing && !isempty(team)
        results = _inner_team_map(team, nwidth) do i
            _widthscan_combo(i, widths, inputsEP, inputsPR; use_gpu=use_gpu)
        end
    else
        results = Vector{Any}(undef, nwidth)
        TJLF.with_blas_threads(1) do
            Threads.@threads for i in 1:nwidth
                results[i] = _widthscan_combo(i, widths, inputsEP, inputsPR; use_gpu=use_gpu)
            end
        end
    end

    for r in results
        i = r.i
        ky_in = r.ky
        for n in 1:NM
            growthrate[i, n] = r.gamma_out[n]
            frequency[i, n]  = r.freq_out[n]
            lkeep_i[i, n]    = r.LKEEP[n]
        end
    end

    # find_max: width at the largest kept-mode growth rate (Fortran TGLFEP_ky_widthscan).
    width_in = width_min
    gmark = zero(T)
    fmark = zero(T)
    for i in 1:nwidth
        for n in 1:NM
            if lkeep_i[i, n] && growthrate[i, n] > gmark
                gmark = growthrate[i, n]
                fmark = frequency[i, n]
                width_in = widths[i]
            end
        end
    end

    # out.ky_widthscan_m<mode> buffer (F_REAL-scaled gamma/freq, as in the Fortran).
    mode_in = coalesce(inputsEP.MODE_IN, 2)
    suffix = coalesce(inputsEP.SUFFIX, "")
    f_real = (inputsEP.F_REAL !== missing && inputsEP.IR !== missing) ? inputsEP.F_REAL[inputsEP.IR] : one(T)
    filename = "out.ky_widthscan_m$(mode_in)" * suffix
    buffer = String[]
    push!(buffer, "widthscan at ky = $(ky_in) mode_flag $(mode_in) factor $(inputsEP.FACTOR_IN)")
    push!(buffer, "width,(gamma(n),freq(n),n=1,nmodes)")
    if inputsEP.REAL_FREQ != 0
        push!(buffer, "Frequency in real units, plasma frame [kHz]")
    end
    for i in 1:nwidth
        parts = [@sprintf("%5.2f", widths[i])]
        for n in 1:NM
            push!(parts, @sprintf("%12.7f", f_real * growthrate[i, n]), @sprintf("%12.7f", f_real * frequency[i, n]))
        end
        push!(buffer, join(parts))
    end

    return width_in, gmark, fmark, ky_in, (filename, buffer)
end
