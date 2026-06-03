#!/bin/bash -l
# Build the GENERIC GPU sysimage (CUDA + TJLF + TJLFEP full + FUSE/IMAS stack) on a GPU node.
# Works for both the file-based scan and the IMAS/FUSE actor path on GPU.
# Output: build/TJLFEP_gpu_generic_sysimage.so
#
#   sbatch batch_build_gpu_sysimage_generic.sh
#
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_gpu_gen_sysimg
#SBATCH -o build_gpu_generic_sysimage_%j.out
#SBATCH -e build_gpu_generic_sysimage_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -euo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
mkdir -p "${JULIA_DEPOT_PATH}/compiled"

# GENERIC image: TJLFEP_FILE_ONLY must be UNSET/0 so the baked _FILE_ONLY const is false and
# IMAS/FUSE/TurbulentTransport are loaded into the image. JULIA_CUDA_USE_COMPAT=false matches
# the runtime env.
unset TJLFEP_FILE_ONLY || true
export JULIA_CUDA_USE_COMPAT=false

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}/build"

echo "=== TJLFEP GENERIC GPU sysimage build ==="
echo "host: $(hostname)  date: $(date)"
julia --version
nvidia-smi -L 2>/dev/null | head -1 || true
echo "TJLF: $(git -C "${TJLFEP_ROOT}/../TJLF" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "${TJLFEP_ROOT}/../TJLF" log -1 --oneline 2>/dev/null)"
echo "TJLFEP: $(git -C "${TJLFEP_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "${TJLFEP_ROOT}" log -1 --oneline 2>/dev/null)"

# 1) Instantiate + precompile dependencies (FUSE/IMAS/CUDA/...).
# 2) FORCE-recompile TJLFEP so its baked _FILE_ONLY const is false. An ENV change alone does
#    NOT invalidate Julia's precompile cache, and the file-only build above left the depot's
#    TJLFEP .ji at FILE_ONLY=true, so we must force a fresh compile here. This also restores
#    the depot to the generic (dev-friendly) TJLFEP cache afterward.
julia --project="${TJLFEP_ROOT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project="${TJLFEP_ROOT}" -e 'Base.compilecache(Base.identify_package("TJLFEP")); using TJLFEP; @assert !TJLFEP._FILE_ONLY "TJLFEP still compiled file-only; generic build aborted"; println("TJLFEP _FILE_ONLY=", TJLFEP._FILE_ONLY)'

julia --project="${TJLFEP_ROOT}" -t "${SLURM_CPUS_PER_TASK:-32}" build_gpu_sysimage_generic.jl

SO="${TJLFEP_ROOT}/build/TJLFEP_gpu_generic_sysimage.so"
if [[ ! -f "${SO}" ]]; then
    echo "ERROR: sysimage not found at ${SO}"
    exit 1
fi
ls -lh "${SO}"

# Self-check: load the freshly built image with a clean ENV and assert it is GENERIC
# (TJLFEP._FILE_ONLY == false, FUSE baked, IMAS actor entry points present). Fail loudly
# otherwise so we never ship a silently-file-only "generic" image again.
echo "=== verifying generic image (clean env) ==="
env -u TJLFEP_FILE_ONLY julia --startup-file=no --sysimage="${SO}" --project="${TJLFEP_ROOT}" -e '
    fuse = Base.PkgId(Base.UUID("e64856f0-3bb8-4376-b4b7-c03396503992"), "FUSE")
    @assert !TJLFEP._FILE_ONLY  "GENERIC build FAILED: baked TJLFEP._FILE_ONLY=true (file-only variant)"
    @assert haskey(Base.loaded_modules, fuse)  "GENERIC build FAILED: FUSE not baked into image"
    @assert isdefined(TJLFEP, :InputTGLFEP)  "GENERIC build FAILED: TJLFEP.InputTGLFEP missing (IMAS/FUSE actor path not compiled in)"
    println("generic image OK: _FILE_ONLY=", TJLFEP._FILE_ONLY, "  FUSE baked + InputTGLFEP present")'

echo "=== GENERIC GPU sysimage build OK: ${SO} ==="
