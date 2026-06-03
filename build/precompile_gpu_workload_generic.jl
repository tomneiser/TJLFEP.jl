# Precompile workload for the GENERIC GPU sysimage (TJLFEP full / not file-only). Same GPU
# compute trace as precompile_gpu_workload.jl (one nb6 radius on the GPU -> tjlf_LS eigenvalue
# + eigenvector CUSOLVER paths), but does NOT set TJLFEP_FILE_ONLY, so it runs in the full
# module context (IMAS/FUSE/TurbulentTransport loaded). The expensive JIT is the GPU per-combo
# path, which this traces; IMAS/FUSE glue is loaded (baked) but compiles on first actual use.

using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using CUDA
using TJLF
using TJLFEP
using LinearAlgebra
BLAS.set_num_threads(1)

if !CUDA.functional()
    error("precompile_gpu_workload_generic.jl must run on a GPU node (CUDA.functional() == false)")
end
@info "generic precompile workload GPU" name=CUDA.name(first(CUDA.devices()))

const ROOT   = normpath(@__DIR__, "..")
const GACODE = joinpath(ROOT, "examples", "DIIID_202017C42_500ms_v3.1", "input.gacode")
const TGLFEP = joinpath(ROOT, "examples", "DIIID_202017C42_500ms_v3.1", "input_singleradius_nb6.TGLFEP")   # N_BASIS=6, SCAN_N=1

@assert isfile(GACODE) "missing $GACODE"
@assert isfile(TGLFEP) "missing $TGLFEP"

mktempdir() do tmp
    res = run_gacode_scan_task(GACODE, TGLFEP, 1;
        out_dir=tmp, use_gpu=true, printout=false, inner=:threads, team=nothing)
    @info "generic precompile workload done" scan_index=res.scan_index ir=res.ir sfmin=res.sfmin
end
