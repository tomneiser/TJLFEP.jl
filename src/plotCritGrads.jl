function read_crit_grad(filepath::String)
    lines = readlines(filepath)
    header = lines[1]
    # Line 2 is a Julia array literal: [v1, v2, ...]
    arr_str = strip(lines[2])
    arr_str = strip(arr_str, ['[', ']'])
    vals = parse.(Float64, split(arr_str, ","))
    return header, vals
end

function make_crit_grad_plots(SorF::String; dir::String="")
    if !(SorF in ["STRUCT", "FILE"])
        # error("Invalid argument for SorF: must be 'STRUCT' or 'FILE'")
        println("Invalid argument for SorF: must be 'STRUCT' or 'FILE'")
    end
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

    dpdr_header, dpdr_crit = read_crit_grad(joinpath(dir, "alpha_dpdr_crit.input"))
    dndr_header, dndr_crit = read_crit_grad(joinpath(dir, "alpha_dndr_crit.input"))

    nr = length(dpdr_crit)
    rho = range(0.0, 1.0, length=nr)

    p1 = plot(rho, dpdr_crit,
        xlabel = "ρ (normalized radius)",
        ylabel = "dp/dr_crit (10 kPa/m)",
        title  = dpdr_header,
        legend = false,
        yscale = :log10,
        lw     = 2)

    p2 = plot(rho, dndr_crit,
        xlabel = "ρ (normalized radius)",
        ylabel = "dn/dr_crit (10¹⁹ m⁻⁴)",
        title  = dndr_header,
        legend = false,
        yscale = :log10,
        lw     = 2)

    savefig(p1, joinpath(dir, press_name))
    savefig(p2, joinpath(dir, dens_name))

end
