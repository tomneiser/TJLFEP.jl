#!/bin/bash -l
# Build the GPU-worker sysimage (CUDA + TJLF + TJLFEP file-only) on a GPU node so the GPU
# eigensolve path is traced and baked in. Eliminates the ~110 s/team JIT the cold MPS workers
# pay per radius. Output: build/TJLFEP_gpu_sysimage.so
#
#   sbatch batch_build_gpu_sysimage.sh
#
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_gpu_sysimg
#SBATCH -o build_gpu_sysimage_%j.out
#SBATCH -e build_gpu_sysimage_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -euo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
mkdir -p "${JULIA_DEPOT_PATH}/compiled"

# FILE_ONLY must be set when TJLFEP is (pre)compiled so the const bakes in true and FUSE/IMAS
# are excluded from the image. JULIA_CUDA_USE_COMPAT=false to match the runtime worker env.
export TJLFEP_FILE_ONLY=1
export JULIA_CUDA_USE_COMPAT=false

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}/build"

echo "=== TJLFEP GPU sysimage build ==="
echo "host: $(hostname)  date: $(date)"
julia --version
nvidia-smi -L 2>/dev/null | head -1 || true
echo "TJLF: $(git -C "${TJLFEP_ROOT}/../TJLF" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "${TJLFEP_ROOT}/../TJLF" log -1 --oneline 2>/dev/null)"
echo "TJLFEP: $(git -C "${TJLFEP_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "${TJLFEP_ROOT}" log -1 --oneline 2>/dev/null)"

# Precompile the env with FILE_ONLY=1 so TJLFEP's .ji bakes _FILE_ONLY=true (no FUSE/IMAS).
julia --project="${TJLFEP_ROOT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia --project="${TJLFEP_ROOT}" -t "${SLURM_CPUS_PER_TASK:-32}" build_gpu_sysimage.jl

SO="${TJLFEP_ROOT}/build/TJLFEP_gpu_sysimage.so"
if [[ ! -f "${SO}" ]]; then
    echo "ERROR: sysimage not found at ${SO}"
    exit 1
fi
ls -lh "${SO}"
echo "=== GPU sysimage build OK: ${SO} ==="
