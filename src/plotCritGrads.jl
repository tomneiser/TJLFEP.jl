"""
    read_crit_grad(filepath; code="auto") -> (header, values)

Read a TGLF-EP critical-gradient profile file (`alpha_dndr_crit.input` /
`alpha_dpdr_crit.input`).

Two on-disk layouts are supported and auto-detected (`code="auto"`, the
default):

  * Fortran `TGLFEP`/`Alpha` format — one value per line (`F12.4`) after the
    header (this is what TJLFEP now writes; see [`write_crit_grad`](@ref)).
  * Legacy Julia array-literal format — line 2 is `[v1, v2, ...]`.

Pass `code="julia"` or `code="fortran"` to force a specific layout.
"""
function read_crit_grad(filepath::String; code::String="auto")
    lines = readlines(filepath)
    header = lines[1]
    body = length(lines) >= 2 ? strip(lines[2]) : ""
    use_julia = code == "julia" || (code == "auto" && startswith(body, "["))
    if use_julia
        # Line 2 is a Julia array literal: [v1, v2, ...]
        arr_str = strip(strip(lines[2]), ['[', ']'])
        vals = [parse(Float64, strip(x)) for x in split(arr_str, ",") if !isempty(strip(x))]
    else
        # Fortran format: one value per line starting at line 2
        vals = Float64[]
        for ln in @view lines[2:end]
            s = strip(ln)
            isempty(s) && continue
            x = tryparse(Float64, s)
            x === nothing || push!(vals, x)
        end
    end
    return header, vals
end

"""
    write_crit_grad(io::IO, header, values)
    write_crit_grad(path::AbstractString, header, values)

Write a TGLF-EP critical-gradient profile (`alpha_dndr_crit.input` /
`alpha_dpdr_crit.input`) in the Fortran `TGLFEP_driver` layout so the Fortran
`Alpha` solver can read it: a one-line `header`, then one value per line
formatted as `F12.4`.

This matches `TGLFEP_driver.f90` (`write(22,'(F12.4)') <profile>`) and is what
`Alpha_read_input.f90` expects (it reads the first 5 header chars — `Densi` /
`Press` — followed by `read(6,'(F12.4,A6)')` per value). The previous TJLFEP
behaviour (`println(io, values)`, a single-line Julia array literal) is not
readable by the Fortran `Alpha`; use this helper instead.
"""
function write_crit_grad(io::IO, header::AbstractString, values::AbstractVector)
    println(io, header)
    for v in values
        @printf(io, "%12.4f\n", float(v))
    end
    return nothing
end

function write_crit_grad(path::AbstractString, header::AbstractString, values::AbstractVector)
    open(path, "w") do io
        write_crit_grad(io, header, values)
    end
    return nothing
end

"""
    make_crit_grad_plots(SorF="neither"; dir="", scale="identity", code="julia")

Plot the critical EP pressure- and density-gradient profiles
(`alpha_dpdr_crit`, `alpha_dndr_crit`) written by a TJLFEP scan, saving
`*_dpdr_crit.png` / `*_dndr_crit.png`.

`SorF` selects the input/output directory and filename prefix (`"STRUCT"` →
`structOutputs/`, `"FILE"` → `fileOutputs/`, anything else → `dir`/unprefixed).
`scale` sets the y-axis scale and `code` selects the input file format
(`"julia"` array literal vs. Fortran one-value-per-line).
"""
function make_crit_grad_plots(SorF::String="neither"; dir::String="", scale::String="identity", code::String="julia")
    if SorF == "STRUCT"
        dir  = "structOutputs"
        press_name = "STRUCT_alpha_dpdr_crit.png"
        dens_name = "STRUCT_alpha_dndr_crit.png"
    elseif SorF == "FILE"
        dir  = "fileOutputs"
        press_name = "FILE_alpha_dpdr_crit.png"
        dens_name = "FILE_alpha_dndr_crit.png"
    else
        press_name = "alpha_dpdr_crit.png"
        dens_name = "alpha_dndr_crit.png"
    end

    dpdr_header, dpdr_crit = read_crit_grad(joinpath(dir, "alpha_dpdr_crit.input"); code=code)
    dndr_header, dndr_crit = read_crit_grad(joinpath(dir, "alpha_dndr_crit.input"); code=code)

    nr = length(dpdr_crit)
    rho = range(0.0, 1.0, length=nr)

    p1 = plot(rho, dpdr_crit,
        xlabel = "ρ (normalized radius)",
        ylabel = "dp/dr_crit (10 kPa/m)",
        title  = dpdr_header,
        legend = false,
        yscale = Symbol(scale),
        lw     = 2)

    p2 = plot(rho, dndr_crit,
        xlabel = "ρ (normalized radius)",
        ylabel = "dn/dr_crit (10¹⁹ m⁻⁴)",
        title  = dndr_header,
        legend = false,
        yscale = Symbol(scale),
        lw     = 2)

    dpdr_norm = dpdr_crit ./ maximum(abs.(dpdr_crit))
    dndr_norm = dndr_crit ./ maximum(abs.(dndr_crit))
    p3 = plot(rho, dpdr_norm,
        xlabel = "ρ (normalized radius)",
        ylabel = "Normalized Critical Gradient",
        title  = "Normalized Critical Gradients",
        label  = "dp/dr_crit",
        yscale = Symbol(scale),
        lw     = 2)
    plot!(p3, rho, dndr_norm,
        label  = "dn/dr_crit",
        yscale = Symbol(scale),
        lw     = 2)

    savefig(p1, joinpath(dir, press_name))
    savefig(p2, joinpath(dir, dens_name))
    savefig(p3, joinpath(dir, "normalized_crit_grads.png"))

end
