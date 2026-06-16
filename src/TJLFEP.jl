module TJLFEP
# IMAS/FUSE entry points (`runTHD(::IMAS.dd)`, `preprocess_imas_inputs`, ...) live in the
# TJLFEPIMASExt package extension (ext/TJLFEPIMASExt.jl), which Julia loads automatically
# only when IMAS, GACODE, and TurbulentTransport are all present in the environment (e.g.
# under FUSE). Loaded standalone, TJLFEP provides only the file-based path
# (`runTHD(::String,...)` / `runTHD_from_gacode`) and stays light (no IMAS/HDF5/FUSE).
# This replaces the former `TJLFEP_FILE_ONLY` ENV flag, which was fragile because ENV is
# not part of Julia's precompile cache key.

using Base.Threads
using LinearAlgebra
using SparseArrays
using Printf
using StaticArrays
using TJLF
using TJLF: InputTJLF  # consolidated single input type (re-exported below)
using Plots

include("tjlfep_modules.jl")
include("tjlfep_inputs.jl")
include("tjlfep_ad_extensions.jl")
include("tjlfep_read_inputs.jl")
include("tjlfep_debug.jl")
include("EXPROconst.jl")
include("tjlfep_ky.jl")
include("tjlfep_kwscale_scan.jl")
include("tjlfep_ql_extract.jl")
include("mainsub.jl")
include("tjlfep_complete_output.jl")
include("run_tjlfep_file.jl")

include("tjlfep_generate_input.jl")

include("plotCritGrads.jl")

# Generic function bindings for the IMAS-path entry points implemented in
# ext/TJLFEPIMASExt.jl. Declaring them here lets the extension add methods, and lets
# TJLFEP export the names even when the extension is not loaded.
function preprocess_imas_inputs end
function save_imas_preprocessed_inputs end
function remap_extraEP_for_fortran_save! end
# SPMD per-radius entry point for the MPS-team layout (method in ext/TJLFEPIMASExt.jl).
function runTHD_dd_radius end

export profile, Options, InputTJLF  # InputTJLF is TJLF.InputTJLF (single consolidated type)
export readMTGLF, readTGLFEP, TJLF_map, readEXPRO, save_TGLFEP, save_MTGLF, save_EXPRO, save_all
export read_input_profile, readprofile, expro_dict_from_profile, ir_exp_from_scan, setup_fortran_file_inputs, setup_gacode_file_inputs, profile_from_gacode
export expro_vectors_from_gacode, preprocess_gacode_inputs
export expro_species_for_gacode_is_ep, read_gacode_scalar_field, read_gacode_ion_field
export expro_bound_deriv, expro_log_gradients, read_expro_for_alpha, compute_alpha_crit_profiles
export tjlfep_complete_output
export runTHD, runTHD_from_gacode
export gamma_dgamma_dfactor, gamma_grad, marginal_factor, marginal_factor_faithful, critical_factor_grid, critical_factor_optimize, critical_factor_ad_guarded, critical_factor_robust, critical_factor_confirm, critical_factor_profile, gamma_input_sensitivities, critical_factor_ad_f1seed, critical_factor_truth, critical_factor_triggered
export MarginalQLData, extract_marginal_ql, build_alpha_ql_modes
export ql_flux_scan_at_marginal, diff_star_from_D_W
export populate_tjlfep_profile!
export run_gacode_scan_task, finalize_gacode_scan, slurm_array_task_id

export TJLFEP_generate_input, readline_values

# IMAS-path entry points (methods provided by TJLFEPIMASExt when IMAS/GACODE/TurbulentTransport
# are loaded). InputTGLFEP now lives in TurbulentTransport (use `TurbulentTransport.InputTGLFEP`).
export preprocess_imas_inputs, save_imas_preprocessed_inputs, remap_extraEP_for_fortran_save!
export runTHD_dd_radius

export make_crit_grad_plots
export TJLF

end 