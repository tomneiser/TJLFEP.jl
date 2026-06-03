# Build the GPU-worker sysimage: CUDA + TJLF + TJLFEP (file-only), with the GPU eigensolve
# path precompiled via precompile_gpu_workload.jl. This bakes in the ~110 s/team JIT that the
# cold MPS workers currently pay on every radius. FUSE/IMAS/TurbulentTransport are excluded
# (TJLFEP_FILE_ONLY=1 must be set when TJLFEP is precompiled, which the batch script does).
#
# Run on a GPU node (the precompile workload needs a functional GPU).

ENV["TJLFEP_FILE_ONLY"] = "1"
using Pkg
Pkg.activate(normpath(@__DIR__, ".."))
Pkg.instantiate()
using PackageCompiler

create_sysimage(
    [:CUDA, :TJLF, :TJLFEP];
    sysimage_path = normpath(@__DIR__, "TJLFEP_gpu_sysimage.so"),
    precompile_execution_file = normpath(@__DIR__, "precompile_gpu_workload.jl"),
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
