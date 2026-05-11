using Distributed

"""
runTHD(tglfepfilepath::String, mtglffilepath::String, exprofilepath::String)

inputs: tglfepfilepath, mtglffilepath, exprofilepath

inputs are used in the threads version of the TJLFEP code for a single run
"""
function runTHD(tglfepfilepath::String, mtglffilepath::String, exprofilepath::String; printout::Bool=false, use_gpu::Bool=false)

    # Auto-detect device via TJLF.pick_device(:auto); shadows the use_gpu parameter.
    # Thread safety: Threads.@threads runs each iteration in a separate Julia task.
    # CUDA.jl v5 assigns per-task streams, so concurrent GPU calls are stream-isolated.
    # use_gpu = TJLF.pick_device(:auto) === :gpu
    # processor = use_gpu ? "GPU" : "CPU"
    # println("TJLFEP runTHD: using $processor")

    # Default values for EXPRO:
    ni = TJLFEP.exproConst.ni
    Ti = TJLFEP.exproConst.Ti
    dlnnidr = TJLFEP.exproConst.dlnnidr
    dlntidr = TJLFEP.exproConst.dlntidr
    cs = TJLFEP.exproConst.cs
    rmin_ex = TJLFEP.exproConst.rmin_ex
    omegaGAM = TJLFEP.exproConst.omegaGAM
    gammaE = TJLFEP.exproConst.gammaE
    gammap = TJLFEP.exproConst.gammap
    # These should be set from the working directory, but these test cases are good for now:

    homedir = pwd()

    iEPexist::Bool = false
    iMPexist::Bool = false
    iEXPexist::Bool = false



    iEPexist = isfile(tglfepfilepath)
    iMPexist = isfile(mtglffilepath)
    iEXPexist = isfile(exprofilepath)


    @assert iEPexist != false "Requested TGLFEP input file path does not exist"
    @assert iMPexist != false "Requested MTGLF input file path does not exist"
    @assert iEXPexist != false "Requested EXPRO input file path does not exist"

    inputEPfile = tglfepfilepath
    inputMPfile = mtglffilepath
    inputEXPfile = exprofilepath

    #inputEPfile = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.TGLFEP"
    #inputMPfile = "/Users/benagnew/TJLF.jl/outputs/tglfep_tests/input.MTGLF"

    # Set up profile struct:
    prof = TJLFEP.readMTGLF(inputMPfile)
    profile = prof[1]
    ir_exp = prof[2]

    # Set up TGLFEP struct:
    Options = TJLFEP.readTGLFEP(inputEPfile, ir_exp)

    # Set up EXPRO constants:
    # IS_EP in input.TGLFEP matches the EXPRO species index for the EP (Fortran convention).
    ni, Ti, dlnnidr, dlntidr, cs, rmin_ex, gammaE, gammap, omegaGAM = TJLFEP.readEXPRO(inputEXPfile, Options.IS_EP)

    profile.gammaE = gammaE
    profile.gammap = gammap
    profile.omegaGAM = omegaGAM

    println("options ir_exp is ", Options.IR_EXP)

    # If IR_EXP was not saved in the MTGLF file (e.g. legacy test cases), fall back to linear spacing
    if isempty(Options.IR_EXP)
        Options.IR_EXP = fill(0, Options.SCAN_N)
        for i = 1:Options.SCAN_N
            if (Options.SCAN_N != 1)
                jr_exp = profile.IRS + floor((i-1)*(profile.NR-profile.IRS)/(Options.SCAN_N-1))
            else
                jr_exp = profile.IRS
            end
            Options.IR_EXP[i] = jr_exp
        end
        println("IR_EXP not found in file, using linear spacing: ", Options.IR_EXP)
    end

    dpdr_EP = fill(NaN, profile.NR)
    if (Options.INPUT_PROFILE_METHOD == 2)
        for i in eachindex(dpdr_EP)
            dpdr_EP[i] = ni[i]*Ti[i]*(dlnnidr[i]+dlntidr[i])# This has some small changes from old main
        end
        #println(Options.FACTOR)
        dpdr_EP_abs = abs.(dpdr_EP)
        dpdr_EP_max = maximum(dpdr_EP_abs)
        dpdr_EP_max_loc = argmax(dpdr_EP_abs)
        n_at_max = ni[dpdr_EP_max_loc]
        if (Options.PROCESS_IN != 5)
            for ir = 1:Options.SCAN_N
                # Options.FACTOR = Options.FACTOR*dpdr_EP_max/dpdr_EP_abs[Options.IR_EXP[ir]] 
                # matches fortran
                Options.FACTOR[ir] = Options.FACTOR[ir]*dpdr_EP_max/dpdr_EP_abs[Options.IR_EXP[ir]] 
            end
        end
        Options.FACTOR_MAX_PROFILE .= Options.FACTOR
    end

    Options.F_REAL .= 1.0
    if (Options.REAL_FREQ == 1) 
        Options.F_REAL .= (cs[:]/(rmin_ex[profile.NR]))/(2*pi*1.0e3)
    end

    # deepcopy is required so as to avoid overwriting of data:
    n_ir = Options.SCAN_N
    Ts = fill(Options, n_ir)
    Ts[1] = deepcopy(Options)
    for i in 2:n_ir
        Ts[i] = deepcopy(Ts[i-1])
    end
    arrTGLFEP = Ts
    arrMTGLF = Vector{typeof(profile)}(undef, n_ir)
    arrMTGLF[1] = deepcopy(profile)
    for i in 2:n_ir
        arrMTGLF[i] = deepcopy(arrMTGLF[i-1])
    end
    arrgrowth = fill(fill(NaN,(5, 10, 10, Options.NMODES)), n_ir)

    stdout_lock = ReentrantLock()
    Threads.@threads for i in 1:n_ir
        #try
            arrTGLFEP[i].IR = arrTGLFEP[i].IR_EXP[i]
            ir = arrTGLFEP[i].IR
            str_r = lpad(string(ir), 3, '0')
            arrTGLFEP[i].SUFFIX = "_r"*str_r

            arrTGLFEP[i].FACTOR_IN = arrTGLFEP[i].FACTOR[i]
            input1 = arrTGLFEP[i]
            input2 = arrMTGLF[i]

            lock(stdout_lock) do
                println("=============================================================")
                println("pre mainsub")
                println("i is ", i, " ir is ", ir)
                println("=============================================================")
            end

            arrgrowth[i], arrTGLFEP[i], arrMTGLF[i] = TJLFEP.mainsub(input1, input2, printout; use_gpu=use_gpu)
        #catch
        #end
    end

    # IS was set on each arrMTGLF[i] deepcopy by TJLF_map; copy it back to the
    # original profile which is used below for indexing (e.g. profile.AS[..., profile.IS])
    profile.IS = arrMTGLF[1].IS

    Options = arrTGLFEP[1]
    
    kymark_out::Vector{Float64} = fill(NaN, Options.SCAN_N)
    width::Vector{Float64} = fill(NaN, Options.SCAN_N)

    if (!Options.WIDTH_IN_FLAG)
        # Non-MPI:
        # There are only "3" processes in Threads -- 
        for i = 1:n_ir
            width[i] = arrTGLFEP[i].WIDTH_IN
            kymark_out[i] = arrTGLFEP[i].KYMARK
            
        end
    end

    # Options = arrTGLFEP[1]
    outTGLFEP_buffer = String[]
    if (printout)
        push!(outTGLFEP_buffer, "process_in = $(Options.PROCESS_IN)")
        if (Options.PROCESS_IN <= 1) push!(outTGLFEP_buffer, "mode_in = $(Options.MODE_IN)") end
        if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5)) push!(outTGLFEP_buffer, "threshold_flag = $(Options.THRESHOLD_FLAG)") end
        push!(outTGLFEP_buffer, "ky_mode = $(Options.KY_MODEL)")
        push!(outTGLFEP_buffer, "--------------------------------------------------------------")
        push!(outTGLFEP_buffer, "scan_n = $(Options.SCAN_N)")
        push!(outTGLFEP_buffer, "irs = $(Options.IRS)")
        push!(outTGLFEP_buffer, "n_basis = $(Options.N_BASIS)")
        push!(outTGLFEP_buffer, "scan_method = $(Options.SCAN_METHOD)")
        if (Options.WIDTH_IN_FLAG)
            push!(outTGLFEP_buffer, "ir,  width")
            for i = 1:Options.SCAN_N
                push!(outTGLFEP_buffer, "$(Options.IRS+i-1) $(width[i])")
            end
        else
            push!(outTGLFEP_buffer, "ir,  width,  kymark")
            for i = 1:Options.SCAN_N
                push!(outTGLFEP_buffer, "$(Options.IRS+i-1) $(width[i]) $(kymark_out[i])")
            end
        end
        push!(outTGLFEP_buffer, "--------------------------------------------------------------")
        push!(outTGLFEP_buffer, "factor_in_profile = $(Options.FACTOR_IN_PROFILE)")
        if (Options.FACTOR_IN_PROFILE)
            for i = 1:Options.SCAN_N
                push!(outTGLFEP_buffer, string(Options.FACTOR[i]))
            end
        else
            push!(outTGLFEP_buffer, string(Options.FACTOR[1]))
        end
        push!(outTGLFEP_buffer, "width_in_flag = $(Options.WIDTH_IN_FLAG)")
        if (!Options.WIDTH_IN_FLAG) push!(outTGLFEP_buffer, "width_min = $(Options.WIDTH_MIN) width_max = $(Options.WIDTH_MAX)") end
    end
    
    # Initialize output arrays
    # kymark_out and width already defined above
    
    # Now continue on to radii-dependent part:
    if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5))
        if (printout)
            push!(outTGLFEP_buffer, "**************************************************************")
            push!(outTGLFEP_buffer, "************** The critical EP density gradient **************")
            push!(outTGLFEP_buffer, "**************************************************************")
        end

        SFmin = fill(0.0, Options.SCAN_N)
        SFmin_out = fill(0.0, profile.NR)
        dndr_crit = fill(NaN, Options.SCAN_N)
        dndr_crit_out = fill(NaN, profile.NR)
        dpdr_crit = fill(NaN, Options.SCAN_N)
        dpdr_crit_out = fill(NaN, profile.NR)

        if (Options.THRESHOLD_FLAG == 0)
            for i = 1:n_ir
                SFmin[i] = arrTGLFEP[i].FACTOR_IN
            end
            # MPI Original:
            #SFmin[1] = Options.FACTOR_IN
            #println(io3, Options.FACTOR_IN, " factor_in before")
            #println("Before MPI.Recv! for factor_in.")
            #=for i = 1:Options.SCAN_N-1
                buf_factor = [NaN]
                MPI.Recv!(buf_factor, i, i, MPI.COMM_WORLD)
                SFmin[i+1] = buf_factor[1]
                #println(io3, SFmin[i_1], " factor_in after and ", Options.FACTOR_IN, " buf_factor after")
                #println(io3, SFmin) # before buf_factor
                #SFmin[i+1] = buf_factor[1]
                #println(io3, SFmin) # after buf_factor, before comp.out
            
            end=#
            if (printout)
                println("After MPI.Recv! for factor_in")
                push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                push!(outTGLFEP_buffer, "SFmin")
            end
        
        # Next is TGLFEP_complete_output(SFmin, SFmin_out, ir_min, ir_max, l_accept_profile)
        # This function's goal is to determine whether 

            SFmin, SFmin_out, ir_min, ir_max, l_accept_profile = tjlfep_complete_output(SFmin, Options, profile)
        
            if (printout)
                push!(outTGLFEP_buffer, string(SFmin, " SFmin after buf and coutput")) # after comp.out
            end
            #println(io3, Options.FACTOR_MAX_PROFILE)

            # We've received the altered profile (interpolated and accepted or not).
            # If the minimum radius is not the first one...
            if ((ir_min-Options.IRS+1) > 1)
                # Originally, this had no concern for accessing out-of-range values;
                # it does now:

                # Set any scans before this point's factor values to the factor_max_profile values respectively (?? Why so physically)
                # If you're running a normal amount of scans, this will pretty much never be done, right?
                if (ir_min-Options.IRS > Options.SCAN_N)
                    SFmin[1:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]
                else # original alone:
                    SFmin[1:ir_min-Options.IRS] = Options.FACTOR_MAX_PROFILE[1:ir_min-Options.IRS]
                end

                # If the starting radius is greater than 1, set the values before it in the interpolated profile to
                # the first value of the max_profile
                if (Options.IRS > 1) SFmin_out[1:Options.IRS-1] .= Options.FACTOR_MAX_PROFILE[1] end

                # This one doesn't look right...
                # This says that in the interpolated profile, you should set any values from the initial to the first point (minus 1)
                # to the same factor_max_profile. But this factor_max_profile is not an interpolation...
                # Is this being done as a default value? FACTOR_MAX_PROFILE if just scan_n of the same value for the case I've been testing
                # hence it's return of 1.0 (*) when rejected. The problem is that this doesn't make much sense for SFmin_out...

                # The problem is that this ignores defaults again. If factor_max_profile is accessed outside of scan_n, they should be set to 0...
                if (ir_min-Options.IRS > Options.SCAN_N)
                    # The +1 on SFmin_out exists because the original is keeping a spacing of 1 between the two...
                    SFmin_out[Options.IRS:Options.SCAN_N+1] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]

                    # This is a test input:
                    # SFmin_out[Options.SCAN_N+1:ir_min-Options.IRS] = 0.0
                else
                    SFmin_out[Options.IRS:ir_min-1] = Options.FACTOR_MAX_PROFILE[1:ir_min-Options.IRS]
                end
            end

            # Perform a similar maneuver for above the maximum:
            if ((ir_max-Options.IRS+1) < Options.SCAN_N)
                SFmin[ir_max-Options.IRS+2:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[ir_max-Options.IRS+2:Options.SCAN_N]
                if (Options.IRS+Options.SCAN_N-1 < profile.NR) SFmin_out[Options.IRS+Options.SCAN_N:profile.NR] .= Options.FACTOR_MAX_PROFILE[Options.SCAN_N] end
                SFmin_out[ir_max+1:Options.IRS+Options.SCAN_N-1] = Options.FACTOR_MAX_PROFILE[ir_max-Options.IRS+2:Options.SCAN_N]
            end

            if (printout)
                push!(outTGLFEP_buffer, string(SFmin, " SFmin after Max assign"))
            end

            if (printout)
                for i = 1:Options.SCAN_N
                    if (l_accept_profile[i])
                        push!(outTGLFEP_buffer, string(SFmin[i]))
                    else
                        push!(outTGLFEP_buffer, string(SFmin_out[i+Options.IRS-1], "   (*)"))
                    end
                    push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                    push!(outTGLFEP_buffer, string(l_accept_profile))
                    push!(outTGLFEP_buffer, string(ir_min, " : ", ir_max))
                end
            end

            # Calculate the density critical gradient at each of the scanned radii.
            if (Options.INPUT_PROFILE_METHOD == 2)
                dndr_crit .= 10000.0
                for i = 1:Options.SCAN_N
                    # If SFmin[i] is not the default non-rejected value, multiply the scalefactor by the density and the density gradient at that point 
                    # for the energetic ion. 
                    # If SFmin[i] is the default or >= 9k, check if it is one of the factor_max_profile ones, and if so, calculate it with that.
                    # otherwise, leave it at 10k.
                    if (SFmin[i] < 9000.0)
                        dndr_crit[i] = SFmin[i]*ni[Int(Options.IR_EXP[i])]*dlnnidr[Int(Options.IR_EXP[i])]
                    elseif ((i < ir_min-Options.IRS+1) || (i > ir_max-Options.IRS+1))
                        dndr_crit[i] = Options.FACTOR_MAX_PROFILE[i]*ni[Int(Options.IR_EXP[i])]*dlnnidr[Int(Options.IR_EXP[i])]
                    end
                end
                # Interpolate and accept are reject needed values of this profile:
                dndr_crit, dndr_crit_out, ir_dum_1, ir_dum_2, l_accept_profile = tjlfep_complete_output(dndr_crit, Options, profile)
                
                if (printout)
                    io4 = open("alpha_dndr_crit.input", "w")
                    println(io4, "Density critical gradient (10^19/m^4)")
                    println(io4, dndr_crit_out)
                    close(io4)
                end
            end

            if (Options.INPUT_PROFILE_METHOD == 2)
                dpdr_crit .= 10000.0
                dpdr_EP[:] .= ni[:].*Ti[:].*(dlnnidr[:].+dlntidr[:]).*0.16022
                for i = 1:Options.SCAN_N
                    if (SFmin[i] < 9000.0)
                        if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5))
                            case = Options.SCAN_METHOD
                            if (case == 1)
                                dpdr_scale = SFmin[i]
                            elseif (case == 2)
                                dpdr_scale = ((SFmin[i]*dlnnidr[Options.IR_EXP[i]]+dlntidr[Options.IR_EXP[i]]) /
                                (dlnnidr[Options.IR_EXP[i]]+dlntidr[Options.IR_EXP[i]]))
                            end
                            dpdr_crit[i] = dpdr_scale*dpdr_EP[Options.IR_EXP[i]]
                        end # 4 || 5
                    end # < 9000
                end # over scan_n
                dpdr_crit, dpdr_crit_out, ir_dum_1, ir_dum_2, l_accept_profile = tjlfep_complete_output(dpdr_crit, Options, profile)
                
                if (printout)
                    io5 = open("alpha_dpdr_crit.input", "w")
                    println(io5, "Pressure critical gradient (10 kPa/m)")
                    println(io5, dpdr_crit_out)
                    close(io5)
                end
            end # end prof. method 2

            if (printout)
                push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                push!(outTGLFEP_buffer, "The EP density threshold n_EP/n_e (%) for gamma_AE = 0")
                for i = 1:Options.SCAN_N
                    push!(outTGLFEP_buffer, string(SFmin[i]*profile.AS[Options.IRS+i-1, profile.IS]*100.0))
                end
            end
            
            if (printout)
                push!(outTGLFEP_buffer, "--------------------------------------------------------------")
                push!(outTGLFEP_buffer, "The EP beta crit (%) = beta_e*(n_EP_th/n_e)*(T_EP/T_e)")
                for i = 1:Options.SCAN_N
                    if (profile.GEOMETRY_FLAG == 0)
                        push!(outTGLFEP_buffer, string(SFmin[i]*profile.BETAE[Options.IRS+i-1]*100.0*profile.AS[Options.IRS+i-1, profile.IS]*profile.TAUS[Options.IRS+i-1, profile.IS]))
                    else
                        push!(outTGLFEP_buffer, string(SFmin[i]*profile.BETAE[Options.IRS+i-1]*100.0*profile.AS[Options.IRS+i-1, profile.IS]*profile.TAUS[Options.IRS+i-1, profile.IS]*profile.KAPPA[Options.IRS+i-1]^2))
                    end
                end
            end

            # there is a process_in == 4 addition I won't be doing quite yet.
        else # ThreshFlag != 0
            # Skipping for now as I want to test just threshold flag == 0 first
        end # ThreshFlag
    end # process 4 || 5

    # At the very end, write the buffer to file once
    if (printout)
        open("out.TGLFEP", "w") do io
            for line in outTGLFEP_buffer
                println(io, line)
            end
        end
    end

    return width, kymark_out, SFmin, dpdr_crit_out, dndr_crit_out
end  # End of string-based runTHD

"""
runTHD(tglfepfilepath::String, mtglffilepath::String, exprofilepath::String)

inputs: tglfepfilepath, mtglffilepath, exprofilepath

inputs are used in the threads version of the TJLFEP code for a single run
"""
function runTHD(dd::IMAS.dd, rho::AbstractVector{Float64}, OptionsDict::Dict{String, Any}; printout::Bool=false, saveFiles::Bool=false, dir::String="ddFiles", use_gpu::Bool=false)

    # Default values for EXPRO:
    ni = TJLFEP.exproConst.ni
    Ti = TJLFEP.exproConst.Ti
    dlnnidr = TJLFEP.exproConst.dlnnidr
    dlntidr = TJLFEP.exproConst.dlntidr
    cs = TJLFEP.exproConst.cs
    rmin_ex = TJLFEP.exproConst.rmin_ex
    omegaGAM = TJLFEP.exproConst.omegaGAM
    gammaE = TJLFEP.exproConst.gammaE
    gammap = TJLFEP.exproConst.gammap
    # These should be set from the working directory, but these test cases are good for now:

    input_tglfep, extraEP = TJLFEP.InputTGLFEP(dd, rho; is_ep=OptionsDict["IS_EP"])

    println("printing species masses")
    for is = 1:extraEP["NS"]
        println("mass[", is, "] = ", extraEP["MASS"][is])
    end
    ep_slot = extraEP["EP_SLOT"]
    println("EP mass = ", getfield(input_tglfep[1], Symbol("MASS_$ep_slot")))

    prof = TJLFEP.profile{Float64}(extraEP["NR"], extraEP["NS"])
    profile = TJLFEP.populate_tjlfep_profile!(prof, extraEP, input_tglfep, extraEP["NR"], extraEP["NS"])

    Options = TJLFEP.Options{Float64}(OptionsDict["SCAN_N"], OptionsDict["WIDTH_IN_FLAG"], OptionsDict["nn"], extraEP["NR"], OptionsDict["jtscale_max"], OptionsDict["nmodes"])

    if (OptionsDict["KY_MODEL"] == 0)
        Options.NTOROIDAL = 4
    else
        Options.NTOROIDAL = 3
    end
        
    if (OptionsDict["PROCESS_IN"] == 4 || OptionsDict["PROCESS_IN"] == 5)
        Options.NN = OptionsDict["nn"]
    end

    if (!OptionsDict["FACTOR_IN_PROFILE"])
        Options.FACTOR = fill(OptionsDict["FACTOR_IN"], OptionsDict["SCAN_N"])
    end
    Options.FACTOR_MAX_PROFILE = Options.FACTOR

    # populating other fields goes here
    for key in keys(OptionsDict)
        if hasfield(typeof(Options), Symbol(key))
            setfield!(Options, Symbol(key), OptionsDict[key])
        end
    end

    Options.IR_EXP = fill(0, Options.SCAN_N)
    Options.NMODES = OptionsDict["nmodes"]

    # IS_EP in Options must be N_ION+1 so that IS_EP+1 = ep_slot and j_ion=2:IS_EP = thermal ions only
    # Options.IS_EP = extraEP["N_ION"] + 1

    # EP species is always at the last slot (NS = N_ION + 2)
    # ep_slot = extraEP["NS"]
    ep_slot = extraEP["EP_SLOT"]
    Options.IS_EP = ep_slot - 1
    ni = extraEP["DENS_$ep_slot"]
    Ti = extraEP["TEMP_$ep_slot"]
    dlnnidr = extraEP["DLNNDR_$ep_slot"]
    dlntidr = extraEP["DLNTDR_$ep_slot"]
    cs = extraEP["CS"]
    rmin_ex = extraEP["RMIN"]
    gammaE = extraEP["gammaE"]
    gammap = extraEP["gammap"]
    omegaGAM = extraEP["omegaGAM"]

    dpdr_EP = fill(NaN, profile.NR)
    Options.IR_EXP = fill(0, Options.SCAN_N)
    if (Options.INPUT_PROFILE_METHOD == 2)
        # Allotting Ir_exp not from profile.
        Options.IR_EXP = fill(0, Options.SCAN_N)
        for i = 1:Options.SCAN_N
            # if (Options.SCAN_N != 1)
                # jr_exp = profile.IRS + floor((i-1)*(profile.NR-profile.IRS)/(Options.SCAN_N-1))
                # if (i == 1)
                #     jr_exp = 11.0
                # end
                # println("type jr_exp ", typeof(jr_exp))
                # if (i == Options.SCAN_N)
                    # jr_exp = 99.0
                # end
            # else
                # jr_exp = profile.IRS
            # end
            jr_exp = argmin(abs.(extraEP["grid"] .- rho[i]))
            Options.IR_EXP[i] = jr_exp
        end

        ir_exp = Options.IR_EXP
        for i in eachindex(dpdr_EP)
            dpdr_EP[i] = ni[i]*Ti[i]*(dlnnidr[i]+dlntidr[i])# This has some small changes from old main
        end
        #println(Options.FACTOR)
        dpdr_EP_abs = abs.(dpdr_EP)
        dpdr_EP_max = maximum(dpdr_EP_abs)
        dpdr_EP_max_loc = argmax(dpdr_EP_abs)
        n_at_max = ni[dpdr_EP_max_loc]
        if (Options.PROCESS_IN != 5)
            for ir = 1:Options.SCAN_N
                # Options.FACTOR = Options.FACTOR*dpdr_EP_max/dpdr_EP_abs[ir_exp[ir]] 
                # matches fortran
                Options.FACTOR[ir] = Options.FACTOR[ir]*dpdr_EP_max/dpdr_EP_abs[Options.IR_EXP[ir]] 
            end
        end
        Options.FACTOR_MAX_PROFILE .= Options.FACTOR
    end

    Options.F_REAL .= 1.0
    if (Options.REAL_FREQ == 1) 
        Options.F_REAL .= (cs[:]/(rmin_ex[profile.NR]))/(2*pi*1.0e3)
    end

    if (saveFiles)
        # Remap EP data from ep_slot to IS_EP (= ep_slot-1) so the written EXPRO
        # has EP at index IS_EP, matching the Fortran convention.
        for prefix in ["DENS", "TEMP", "DLNNDR", "DLNTDR"]
            extraEP["$(prefix)_$(ep_slot-1)"] = extraEP["$(prefix)_$ep_slot"]
            delete!(extraEP, "$(prefix)_$ep_slot")
        end
        save_all(Options, profile, extraEP, dir)
    end

    # deepcopy is required so as to avoid overwriting of data:
    n_ir = Options.SCAN_N
    Ts = fill(Options, n_ir)
    Ts[1] = deepcopy(Options)
    for i in 2:n_ir
        Ts[i] = deepcopy(Ts[i-1])
    end
    arrTGLFEP = Ts
    arrMTGLF = Vector{typeof(profile)}(undef, n_ir)
    arrMTGLF[1] = deepcopy(profile)
    for i in 2:n_ir
        arrMTGLF[i] = deepcopy(arrMTGLF[i-1])
    end
    arrgrowth = fill(fill(NaN,(5, 10, 10, Options.NMODES)), n_ir)

    stdout_lock = ReentrantLock()

    pmap_outputs = pmap(i -> begin
        arrTGLFEP_i = deepcopy(arrTGLFEP[i])
        arrMTGLF_i = deepcopy(arrMTGLF[i])

        arrTGLFEP_i.IR = arrTGLFEP_i.IR_EXP[i]
        ir = arrTGLFEP_i.IR
        str_r = lpad(string(ir), 3, '0')
        arrTGLFEP_i.SUFFIX = "_r"*str_r

        arrTGLFEP_i.FACTOR_IN = arrTGLFEP_i.FACTOR[i]
        input1 = arrTGLFEP_i
        input2 = arrMTGLF_i

        println("=============================================================")
        println("pre mainsub")
        println("i is ", i, " ir is ", ir)
        println("=============================================================")

        return TJLFEP.mainsub(input1, input2, printout; use_gpu=use_gpu)
    end, 1:n_ir)

    # pmap_outputs is a Vector of 2-tuples: ((growth, tglfep_i, mtglf_i), (scalefactor_buffer, wavebuffer_all))
    results   = [p[1] for p in pmap_outputs]   # Vector of (growth, tglfep_i, mtglf_i)
    all_buffers = [p[2] for p in pmap_outputs] # Vector of (scalefactor_buffer, wavebuffer_all)

    for (i, (growth, tglfep_i, mtglf_i)) in enumerate(results)
        arrgrowth[i] = growth
        arrTGLFEP[i] = tglfep_i
        arrMTGLF[i] = mtglf_i
    end

    # IS was set on each arrMTGLF[i] deepcopy by TJLF_map; copy it back to the
    # original profile which is used below for indexing (e.g. profile.AS[..., profile.IS])
    profile.IS = arrMTGLF[1].IS

    Options = arrTGLFEP[1]

    if printout
        # Write one scalefactor file and wavefunction file(s) per radial point.
        # Each entry in all_buffers[i] is (scalefactor_buffer, wavebuffer_all) for radius i.
        # wavebuffer_all is a Vector of (filename, buffer) pairs from kwscale_scan.
        for i in 1:n_ir
            sf_buf, wf_buf_all = all_buffers[i]
            suffix_i = coalesce(arrTGLFEP[i].SUFFIX, "")

            # Write scalefactor file for this radius
            if sf_buf !== nothing && !isempty(sf_buf)
                open("out.scalefactor" * suffix_i, "w") do io
                    for line in sf_buf
                        println(io, line)
                    end
                end
            end

            # Write wavefunction file(s) for this radius.
            # Each element is a (str_wf_file, buffer) pair produced by kwscale_scan.
            if wf_buf_all !== nothing && !isempty(wf_buf_all)
                for (str_wf_file, wfbuf) in wf_buf_all
                    if wfbuf !== nothing && !isempty(wfbuf)
                        open(str_wf_file, "w") do io
                            for line in wfbuf
                                println(io, line)
                            end
                        end
                    end
                end
            end
        end
    end
    
    kymark_out::Vector{Float64} = fill(NaN, Options.SCAN_N)
    width::Vector{Float64} = fill(NaN, Options.SCAN_N)

    if (!Options.WIDTH_IN_FLAG)
        # Non-MPI:
        # There are only "3" processes in Threads -- 
        for i = 1:n_ir
            width[i] = arrTGLFEP[i].WIDTH_IN
            kymark_out[i] = arrTGLFEP[i].KYMARK
            
        end
    end

    if (printout)
        io2 = open("out.TGLFEP", "w")
        println(io2, "process_in = ", Options.PROCESS_IN)

        if (Options.PROCESS_IN <= 1) println(io2, "mode_in = ", Options.MODE_IN) end
        if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5)) println(io2, "threshold_flag = ", Options.THRESHOLD_FLAG) end

        println(io2, "ky_mode = ", Options.KY_MODEL)
        println(io2, "--------------------------------------------------------------")
        println(io2, "scan_n = ", Options.SCAN_N)
        println(io2, "irs = ", Options.IRS)
        println(io2, "n_basis = ", Options.N_BASIS)
        println(io2, "scan_method = ", Options.SCAN_METHOD)

        if (Options.WIDTH_IN_FLAG)
            println(io2, "ir,  width")
            for i = 1:Options.SCAN_N
                println(io2, Options.IRS+i-1, " ", width[i])
            end
        else
            println(io2, "ir,  width,  kymark")
            for i = 1:Options.SCAN_N
                println(io2, Options.IRS+i-1, " ", width[i], " ", kymark_out[i])
            end
        end

        println(io2, "--------------------------------------------------------------")
        println(io2, "factor_in_profile = ", Options.FACTOR_IN_PROFILE)
        if (Options.FACTOR_IN_PROFILE)
            for i = 1:Options.SCAN_N
                println(io2, Options.FACTOR[i])
            end
        else
            println(io2, Options.FACTOR[1])
        end

        println(io2, "width_in_flag = ", Options.WIDTH_IN_FLAG)
        if (!Options.WIDTH_IN_FLAG) println(io2, "width_min = ", Options.WIDTH_MIN, " width_max = ", Options.WIDTH_MAX) end
        close(io2)
    end
    
    # Initialize output arrays
    # kymark_out and width already defined above
    
    # Now continue on to radii-dependent part:
    if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5))
        
        if (printout)
            io3 = open("out.TGLFEP", "a")
            println(io3, "**************************************************************")
            println(io3, "************** The critical EP density gradient **************")
            println(io3, "**************************************************************")
        end

        SFmin = fill(0.0, Options.SCAN_N)
        SFmin_out = fill(0.0, profile.NR)
        dndr_crit = fill(NaN, Options.SCAN_N)
        dndr_crit_out = fill(NaN, profile.NR)
        dpdr_crit = fill(NaN, Options.SCAN_N)
        dpdr_crit_out = fill(NaN, profile.NR)

        if (Options.THRESHOLD_FLAG == 0)
            for i = 1:n_ir
                SFmin[i] = arrTGLFEP[i].FACTOR_IN
            end
            # MPI Original:
            #SFmin[1] = Options.FACTOR_IN
            #println(io3, Options.FACTOR_IN, " factor_in before")
            #println("Before MPI.Recv! for factor_in.")
            #=for i = 1:Options.SCAN_N-1
                buf_factor = [NaN]
                MPI.Recv!(buf_factor, i, i, MPI.COMM_WORLD)
                SFmin[i+1] = buf_factor[1]
                #println(io3, SFmin[i_1], " factor_in after and ", Options.FACTOR_IN, " buf_factor after")
                #println(io3, SFmin) # before buf_factor
                #SFmin[i+1] = buf_factor[1]
                #println(io3, SFmin) # after buf_factor, before comp.out
            
            end=#
            if (printout)
                println("After MPI.Recv! for factor_in")
                println(io3, "--------------------------------------------------------------")
                println(io3, "SFmin")
            end
        
        # Next is TGLFEP_complete_output(SFmin, SFmin_out, ir_min, ir_max, l_accept_profile)
        # This function's goal is to determine whether 

            SFmin, SFmin_out, ir_min, ir_max, l_accept_profile = tjlfep_complete_output(SFmin, Options, profile)
        
            if (printout)
                println(io3, SFmin, " SFmin after buf and coutput") # after comp.out
            end
            #println(io3, Options.FACTOR_MAX_PROFILE)

            # We've received the altered profile (interpolated and accepted or not).
            # If the minimum radius is not the first one...
            if ((ir_min-Options.IRS+1) > 1)
                # Originally, this had no concern for accessing out-of-range values;
                # it does now:

                # Set any scans before this point's factor values to the factor_max_profile values respectively (?? Why so physically)
                # If you're running a normal amount of scans, this will pretty much never be done, right?
                if (ir_min-Options.IRS > Options.SCAN_N)
                    SFmin[1:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]
                else # original alone:
                    SFmin[1:ir_min-Options.IRS] = Options.FACTOR_MAX_PROFILE[1:ir_min-Options.IRS]
                end

                # If the starting radius is greater than 1, set the values before it in the interpolated profile to
                # the first value of the max_profile
                if (Options.IRS > 1) SFmin_out[1:Options.IRS-1] .= Options.FACTOR_MAX_PROFILE[1] end

                # This one doesn't look right...
                # This says that in the interpolated profile, you should set any values from the initial to the first point (minus 1)
                # to the same factor_max_profile. But this factor_max_profile is not an interpolation...
                # Is this being done as a default value? FACTOR_MAX_PROFILE if just scan_n of the same value for the case I've been testing
                # hence it's return of 1.0 (*) when rejected. The problem is that this doesn't make much sense for SFmin_out...

                # The problem is that this ignores defaults again. If factor_max_profile is accessed outside of scan_n, they should be set to 0...
                if (ir_min-Options.IRS > Options.SCAN_N)
                    # The +1 on SFmin_out exists because the original is keeping a spacing of 1 between the two...
                    SFmin_out[Options.IRS:Options.SCAN_N+1] = Options.FACTOR_MAX_PROFILE[1:Options.SCAN_N]

                    # This is a test input:
                    # SFmin_out[Options.SCAN_N+1:ir_min-Options.IRS] = 0.0
                else
                    SFmin_out[Options.IRS:ir_min-1] = Options.FACTOR_MAX_PROFILE[1:ir_min-Options.IRS]
                end
            end

            # Perform a similar maneuver for above the maximum:
            if ((ir_max-Options.IRS+1) < Options.SCAN_N)
                SFmin[ir_max-Options.IRS+2:Options.SCAN_N] = Options.FACTOR_MAX_PROFILE[ir_max-Options.IRS+2:Options.SCAN_N]
                if (Options.IRS+Options.SCAN_N-1 < profile.NR) SFmin_out[Options.IRS+Options.SCAN_N:profile.NR] .= Options.FACTOR_MAX_PROFILE[Options.SCAN_N] end
                SFmin_out[ir_max+1:Options.IRS+Options.SCAN_N-1] = Options.FACTOR_MAX_PROFILE[ir_max-Options.IRS+2:Options.SCAN_N]
            end

            if (printout)
                println(io3, SFmin, " SFmin after Max assign")
            end

            if (printout)
                for i = 1:Options.SCAN_N
                    if (l_accept_profile[i])
                        println(io3, SFmin[i])
                    else
                        println(io3, SFmin_out[i+Options.IRS-1], "   (*)")
                    end
                    println(io3, "--------------------------------------------------------------")
                    println(io3, l_accept_profile)
                    println(io3, ir_min, " : ", ir_max)
                end
            end

            # Calculate the density critical gradient at each of the scanned radii.
            if (Options.INPUT_PROFILE_METHOD == 2)
                dndr_crit .= 10000.0
                for i = 1:Options.SCAN_N
                    # If SFmin[i] is not the default non-rejected value, multiply the scalefactor by the density and the density gradient at that point 
                    # for the energetic ion. 
                    # If SFmin[i] is the default or >= 9k, check if it is one of the factor_max_profile ones, and if so, calculate it with that.
                    # otherwise, leave it at 10k.
                    if (SFmin[i] < 9000.0)
                        dndr_crit[i] = SFmin[i]*ni[Int(Options.IR_EXP[i])]*dlnnidr[Int(Options.IR_EXP[i])]
                    elseif ((i < ir_min-Options.IRS+1) || (i > ir_max-Options.IRS+1))
                        dndr_crit[i] = Options.FACTOR_MAX_PROFILE[i]*ni[Int(Options.IR_EXP[i])]*dlnnidr[Int(Options.IR_EXP[i])]
                    end
                end
                # Interpolate and accept are reject needed values of this profile:
                dndr_crit, dndr_crit_out, ir_dum_1, ir_dum_2, l_accept_profile = tjlfep_complete_output(dndr_crit, Options, profile)
                
                if (printout)
                    io4 = open("alpha_dndr_crit.input", "w")
                    println(io4, "Density critical gradient (10^19/m^4)")
                    println(io4, dndr_crit_out)
                    close(io4)
                end
            end
            
            if (Options.INPUT_PROFILE_METHOD == 2)
                dpdr_crit .= 10000.0
                dpdr_EP[:] .= ni[:].*Ti[:].*(dlnnidr[:].+dlntidr[:]).*0.16022
                for i = 1:Options.SCAN_N
                    if (SFmin[i] < 9000.0)
                        if ((Options.PROCESS_IN == 4) || (Options.PROCESS_IN == 5))
                            case = Options.SCAN_METHOD
                            if (case == 1)
                                dpdr_scale = SFmin[i]
                            elseif (case == 2)
                                dpdr_scale = ((SFmin[i]*dlnnidr[Options.IR_EXP[i]]+dlntidr[Options.IR_EXP[i]]) /
                                (dlnnidr[Options.IR_EXP[i]]+dlntidr[Options.IR_EXP[i]]))
                            end
                            dpdr_crit[i] = dpdr_scale*dpdr_EP[Options.IR_EXP[i]]
                        end # 4 || 5
                    end # < 9000
                end # over scan_n
                dpdr_crit, dpdr_crit_out, ir_dum_1, ir_dum_2, l_accept_profile = tjlfep_complete_output(dpdr_crit, Options, profile)
                
                if (printout)
                    io5 = open("alpha_dpdr_crit.input", "w")
                    println(io5, "Pressure critical gradient (10 kPa/m)")
                    println(io5, dpdr_crit_out)
                    close(io5)
                end
            end # end prof. method 2
            if (printout)
                println(io3, "--------------------------------------------------------------")
                println(io3, "The EP density threshold n_EP/n_e (%) for gamma_AE = 0")
                for i = 1:Options.SCAN_N
                    println(io3, SFmin[i]*profile.AS[Options.IRS+i-1, profile.IS]*100.0) #percent
                end
            end
            if (printout)
                println(io3, "--------------------------------------------------------------")
                println(io3, "The EP beta crit (%) = beta_e*(n_EP_th/n_e)*(T_EP/T_e)")
                for i = 1:Options.SCAN_N
                    if (profile.GEOMETRY_FLAG == 0)
                        println(io3, SFmin[i]*profile.BETAE[Options.IRS+i-1]*100.0*profile.AS[Options.IRS+i-1, profile.IS]*profile.TAUS[Options.IRS+i-1, profile.IS]) #percent
                    else
                        println(io3, SFmin[i]*profile.BETAE[Options.IRS+i-1]*100.0*profile.AS[Options.IRS+i-1, profile.IS]*profile.TAUS[Options.IRS+i-1, profile.IS]*profile.KAPPA[Options.IRS+i-1]^2) #percent
                    end
                end
            end
            # there is a process_in == 4 addition I won't be doing quite yet.
        else # ThreshFlag != 0
            # Skipping for now as I want to test just threshold flag == 0 first
        end # ThreshFlag
        if (printout)
            close(io3)
        end
    end # process 4 || 5
    return width, kymark_out, SFmin, dpdr_crit_out, dndr_crit_out
end  # End of struct-based runTHD

#=
"""
runMPIs(tglfepfilepaths::Vector{String}, mtglffilepaths::Vector{String}, exprofilepaths::Vector{String})

Options: tglfepfilepaths, mtglffilepaths, exprofilepaths

These Options are vectors of individual filepaths for the MPI version of the TJLFEP code. It will run the code for each input.
"""
function runMPIs(tglfepfilepaths::Vector{String}, mtglffilepaths::Vector{String}, exprofilepaths::Vector{String})
for i = 1:length(tglfepfilepaths)
    runMPI(tglfepfilepaths[i], mtglffilepaths[i], exprofilepaths[i])
end
end
=#

# runTHDs(tglfepfilepaths::Vector{String}, mtglffilepaths::Vector{String}, exprofilepaths::Vector{String})

# Options: tglfepfilepaths, mtglffilepaths, exprofilepaths

# These inputs are vectors of individual filepaths for the threads version of the TJLFEP code. It will run the code for each input.
# It is probably advised to only use this function on runs that use the same # of scans, 
# """
##Have not updated For Fuse
# function runTHDs(tglfepfilepaths::Vector{String}, mtglffilepaths::Vector{String}, exprofilepaths::Vector{String})
# nruns = length(tglfepfilepaths)

# # Set return vectors:
# widths = fill([], nruns)
# kymark_outs = fill([], nruns)
# SFmins = fill([], nruns)
# dpdr_crit_outs = fill([], nruns)
# dndr_crit_outs = fill([], nruns)

# for i = 1:nruns
#     width, kymark_out, SFmin, dpdr_crit_out, dndr_crit_out = runTHD(tglfepfilepaths[i], mtglffilepaths[i], exprofilepaths[i])

#     widths[i] = width
#     kymark_outs[i] = kymark_out
#     SFmins[i] = SFmin
#     dpdr_crit_outs[i] = dpdr_crit_out
#     dndr_crit_outs[i] = dndr_crit_out
# end
# return widths, kymark_outs, SFmins, dpdr_crit_outs, dndr_crit_outs
# end



"""
checkInput(inputTJLF::InputTJLF)

description:
check that the InputTJLF struct is properly populated
"""
function checkInput(inputTJLFEP::InputTJLF)
field_names = fieldnames(InputTJLFEP)
for field_name in field_names
    field_value = getfield(inputTJLFEP, field_name)
    if typeof(field_value)<:Missing
        @assert !ismissing(field_value) "Did not properly populate inputTJLFEP for $field_name = $field_value"
    end
    if typeof(field_value)<:Real
        @assert !isnan(field_value) "Did not properly populate inputTJLFEP for $field_name = $field_value"
    end
    if typeof(field_value)<:Vector && field_name!=:KY_SPECTRUM && field_name!=:EIGEN_SPECTRUM
        for val in field_value
            @assert !isnan(val) "Did not properly populate inputTJLFEP for array $field_name = $val"
        end
    end
end
if !inputTJLFEP.FIND_EIGEN
    @assert !inputTJLFEP.FIND_WIDTH "If FIND_EIGEN false, FIND_WIDTH should also be false"
end
end

function checkInput(inputTJLFEPVector::Vector{InputTJLF})
for inputTJLFEP in inputTJLFEPVector
    field_names = fieldnames(inputTJLFEP)
    for field_name in field_names
        field_value = getfield(inputTJLFEP, field_name)
        if typeof(field_value)<:Missing
            @assert !ismissing(field_value) "Did not properly populate inputTJLFEP for $field_name = $field_value"
        end
        if typeof(field_value)<:Real
            @assert !isnan(field_value) "Did not properly populate inputTJLFEP for $field_name = $field_value"
        end
        if typeof(field_value)<:Vector && field_name!=:KY_SPECTRUM && field_name!=:EIGEN_SPECTRUM
            for val in field_value
                @assert !isnan(val) "Did not properly populate inputTJLFEP for array $field_name = $val"
            end
        end
    end
    if !inputTJLFEP.FIND_EIGEN
        @assert !inputTJLFEP.FIND_WIDTH "If FIND_EIGEN false, FIND_WIDTH should also be false"
    end
end
end