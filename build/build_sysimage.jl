using Pkg
Pkg.activate("..")
# Pkg.update()
# Pkg.resolve()
# Pkg.instantiate()
using PackageCompiler
# create_sysimage(
#     [:FUSE, :IMAS, :TJLF],
#     sysimage_path="TJLFEP_sysimage.so",
#     precompile_execution_file="../src/main.jl"  # optional but helps
# )
create_sysimage(
    [:FUSE, :IMAS, :TJLF],
    sysimage_path="TJLFEP_sysimage.so"
)
# create_sysimage(
#     [:FUSE, :IMAS],
#     sysimage_path="FUSE_IMAS_ONLY_TJLFEP_sysimage.so"
# )