# Build the FILE-ONLY GPU sysimage: CUDA + TJLF + TJLFEP (standalone, no FUSE/IMAS), with the
# GPU eigensolve path precompiled. This is the image a TGLF-EP user running the file-based scan
# (run_gacode_scan_task / run_tjlfep on input.gacode + input.TGLFEP) gets before going
# FUSE-native -- leaner and faster-loading than TJLFEP_gpu_generic_sysimage.so because the
# IMAS/FUSE stack is NOT baked.
#
# In the current extension model there is no TJLFEP_FILE_ONLY const: TJLFEP is "file-only"
# simply by being loaded standalone (the TJLFEPIMASExt extension only loads when IMAS/GACODE/
# TurbulentTransport are all present). The precompile workload `using`s only CUDA/TJLF/TJLFEP,
# so the extension stays dormant and FUSE/IMAS are never pulled into the image.
#
# Run on a GPU node (the precompile workload needs a functional GPU).

using Pkg

const TJLFEP_ROOT = normpath(@__DIR__, "..", "..")

Pkg.activate(TJLFEP_ROOT)
Pkg.instantiate()
using PackageCompiler

create_sysimage(
    [:CUDA, :TJLF, :TJLFEP];
    sysimage_path = normpath(@__DIR__, "..", "TJLFEP_gpu_sysimage.so"),
    precompile_execution_file = normpath(@__DIR__, "precompile_gpu_workload_fileonly.jl"),
    project = TJLFEP_ROOT,
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
