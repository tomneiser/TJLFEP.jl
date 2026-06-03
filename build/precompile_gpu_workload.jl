# Precompile workload for the GPU-worker sysimage: run ONE nb6 radius on the GPU so that
# PackageCompiler traces and bakes in the hot per-combo path -- tjlf_LS -> tjlf_eigensolver ->
# _standard_eigenvalues_via_solve (CUSOLVER getrf/getrs/Xgeev) and the GPU eigenvector solve
# _gpu_lu_solve! (CUSOLVER getrf/getrs). Method specializations depend on TYPES (Float64 /
# ComplexF64), not matrix size, so nb6 (small, fast) covers the nb32 production case.
#
# Runs inner=:threads in a single process: the MPS team just distributes the same per-combo
# tjlf_LS calls across workers, so tracing the threaded path captures the compilation that
# every worker would otherwise pay (~110 s) on a cold spawn.

ENV["TJLFEP_FILE_ONLY"] = "1"
using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using CUDA
using TJLF
using TJLFEP
using LinearAlgebra
BLAS.set_num_threads(1)

if !CUDA.functional()
    error("precompile_gpu_workload.jl must run on a GPU node (CUDA.functional() == false)")
end
@info "precompile workload GPU" name=CUDA.name(first(CUDA.devices()))

const ROOT   = normpath(@__DIR__, "..")
const GACODE = joinpath(ROOT, "src", "DIIIDfiles", "202017C42_500ms_v3.1", "input.gacode")
const TGLFEP = joinpath(ROOT, "build", "debug_nb6", "input.TGLFEP")   # N_BASIS=6, SCAN_N=1

@assert isfile(GACODE) "missing $GACODE"
@assert isfile(TGLFEP) "missing $TGLFEP"

# Exercise the full GPU per-combo path once (SFmin value irrelevant here; we only need the trace).
mktempdir() do tmp
    res = run_gacode_scan_task(GACODE, TGLFEP, 1;
        out_dir=tmp, use_gpu=true, printout=false, inner=:threads, team=nothing)
    @info "precompile workload done" scan_index=res.scan_index ir=res.ir sfmin=res.sfmin
end
