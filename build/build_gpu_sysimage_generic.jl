# Build the GENERIC GPU sysimage: CUDA + TJLF + TJLFEP (full, NOT file-only) + the FUSE/IMAS
# stack, with the GPU eigensolve path precompiled. Works for both the file-based scan
# (run_gacode_scan_task) and the IMAS/FUSE actor path on GPU. Larger than the file-only image
# but, because FUSE/IMAS are baked, it still loads fast (the FUSE cost is compilation, which
# the image eliminates) -- the only worker penalty vs file-only is the larger .so on disk.
#
# Run on a GPU node. The batch script force-recompiles TJLFEP with FILE_ONLY=0 first so the
# baked _FILE_ONLY const is false (ENV changes alone do not invalidate Julia's precompile cache).

using Pkg
Pkg.activate(normpath(@__DIR__, ".."))
Pkg.instantiate()
using PackageCompiler

# Ensure the TJLFEP baked into the image is the FULL (non-file-only) variant.
# `_FILE_ONLY` is a const evaluated from ENV at precompile time, but ENV is NOT part of
# Julia's precompile cache key, and TJLFEP has multiple cache slots (one per set of compile
# flags). create_sysimage builds with its own flags, so it can otherwise snapshot a stale
# file-only slot left over from the file-only image build. Delete ALL TJLFEP cache slots and
# clear the ENV in this process tree so the create_sysimage subprocess must compile TJLFEP
# fresh with _FILE_ONLY=false (and thus include run_tjlfep_imas/context/actor_context).
delete!(ENV, "TJLFEP_FILE_ONLY")
let cdir = joinpath(first(DEPOT_PATH), "compiled", "v$(VERSION.major).$(VERSION.minor)", "TJLFEP")
    if isdir(cdir)
        @info "Removing stale TJLFEP precompile cache slots" cdir
        rm(cdir; recursive=true, force=true)
    end
end

create_sysimage(
    [:CUDA, :TJLF, :TJLFEP, :FUSE, :IMAS, :GACODE, :IMASdd, :TurbulentTransport];
    sysimage_path = normpath(@__DIR__, "TJLFEP_gpu_generic_sysimage.so"),
    precompile_execution_file = normpath(@__DIR__, "precompile_gpu_workload_generic.jl"),
    cpu_target = PackageCompiler.default_app_cpu_target(),
)
