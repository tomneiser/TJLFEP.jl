# Build the GENERIC GPU sysimage: CUDA + TJLF + TJLFEP (+ the TJLFEPIMASExt IMAS extension)
# + the FUSE/IMAS stack, with the GPU eigensolve path precompiled. Works for both the
# file-based scan (run_gacode_scan_task) and the IMAS/FUSE actor path on GPU. Because FUSE/IMAS
# are baked, the image still loads fast (the FUSE cost is compilation, which the image removes).
#
# IMPORTANT (post-extension refactor): IMAS/GACODE/TurbulentTransport are now *weak* deps of
# TJLFEP and FUSE is not (and cannot be) a TJLFEP dep, so the full stack can only be baked from
# a project that has all of them as direct deps -- that is the FUSE project. We therefore build
# against FUSE_ROOT and stack TJLFEP_ROOT on LOAD_PATH so PackageCompiler (a TJLFEP dep) is
# loadable here. CUDA is pulled in transitively (TJLF/TJLFEP) so it need not be listed.
# Listing IMAS/GACODE/TurbulentTransport makes Julia load them during the build, which bakes the
# TJLFEPIMASExt extension (the dd/FUSE actor entry points) into the image automatically.

using Pkg

const TJLFEP_ROOT = normpath(@__DIR__, "..", "..")
const FUSE_ROOT = get(ENV, "FUSE_ROOT", normpath(TJLFEP_ROOT, "..", "FUSE"))

Pkg.activate(FUSE_ROOT)
Pkg.instantiate()

# Make PackageCompiler (a TJLFEP dep, not a FUSE dep) resolvable without touching FUSE's deps.
push!(LOAD_PATH, TJLFEP_ROOT)
using PackageCompiler

create_sysimage(
    [:TJLF, :TJLFEP, :FUSE, :IMAS, :GACODE, :TurbulentTransport];
    sysimage_path = normpath(@__DIR__, "..", "TJLFEP_gpu_generic_sysimage.so"),
    precompile_execution_file = normpath(@__DIR__, "precompile_gpu_workload_generic.jl"),
    project = FUSE_ROOT,
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
