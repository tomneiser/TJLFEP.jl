#!/bin/bash -l
#SBATCH -A m3739
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_smoke
#SBATCH -o smoke_test_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7

export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
# Optional GPU sysimage (build once with batch_build_gpu_sysimage_generic.sh). Missing -> JIT.
SYSIMAGE="${TJLFEP_GPU_SYSIMAGE:-${TJLFEP_ROOT}/build/TJLFEP_gpu_generic_sysimage.so}"

cd "${TJLFEP_ROOT}/build"
echo "=== TJLFEP smoke test ==="
echo "host: $(hostname)"
echo "date: $(date)"

if [[ -f "${SYSIMAGE}" ]]; then
    ls -lh "${SYSIMAGE}"
    SYSIMG_ARGS=(--sysimage="${SYSIMAGE}")
else
    echo "sysimage not found at ${SYSIMAGE} -> running with JIT"
    SYSIMG_ARGS=()
fi

julia --project="${TJLFEP_ROOT}" \
    "${SYSIMG_ARGS[@]}" \
    -t "${SLURM_CPUS_PER_TASK:-32}" \
    smoke_test.jl

echo "=== smoke test finished ==="
