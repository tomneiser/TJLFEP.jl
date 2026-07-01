function read_crit_grad(filepath::String; code::String="julia")
    lines = readlines(filepath)
    header = lines[1]
    if code == "julia"
        # Line 2 is a Julia array literal: [v1, v2, ...]
        arr_str = strip(strip(lines[2]), ['[', ']'])
        vals = parse.(Float64, split(arr_str, ","))
    else
        # Fortran format: one value per line starting at line 2
        vals = parse.(Float64, strip.(lines[2:end]))
    end
    return header, vals
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
