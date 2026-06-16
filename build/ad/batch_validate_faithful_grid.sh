#!/bin/bash -l
# Diagnose AD-scan sfmin overestimation at strong-drive DIII-D radii (ir=33,38,43) at N_BASIS=32.
# Runs JIT (NO sysimage) to pick up current critical_factor_optimize source.
#   cd build && sbatch ad/batch_diag_outlier_radii.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -J ad_faithful_grid
#SBATCH -o ad/faithful_grid_%j.out
#SBATCH -e ad/faithful_grid_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
export JULIA_CUDA_USE_COMPAT=false
export TJLFEP_FILE_ONLY=1

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project=. -t 8 build/ad/validate_faithful_grid.jl
echo "=== diag_outlier_radii done (exit $?) ==="
