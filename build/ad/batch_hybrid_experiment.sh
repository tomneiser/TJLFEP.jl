#!/bin/bash -l
# Offline accuracy-per-eval experiment for the critical_factor_robust (ky,w) outer solver:
# refine0 vs refine2 grid-zoom vs outer=:hybrid vs a dense faithful-grid continuous truth.
# Runs from LIVE SOURCE (no sysimage — outer=:hybrid postdates the baked image) on ONE GPU so
# eigensolves are fast; JITs once (~15 min) then runs the radius sweep. Accuracy (eval COUNTS,
# sfmin error) is hardware-independent; GPU is just for turnaround.
#   cd build && sbatch ad/batch_hybrid_experiment.sh
#SBATCH -A m3739_g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J tjlfep_hybrid_exp
#SBATCH -o hybrid_experiment_%j.out
#SBATCH -e hybrid_experiment_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"
export JULIA_CUDA_USE_COMPAT=false

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}/build"

# Focused, tractable defaults: a dense 10x20 faithful truth at nb=32 is ~3500 sweeps/radius and
# blew the 1h walltime after one radius. Use a smaller 6x10 dense truth and a discriminating
# radius set — 22 (easy interior control), 38 (strong-drive floor), 48 & 95 (the AD-threads
# spikes where refinement actually matters) — so each accuracy case is exercised within walltime.
export USE_GPU="${USE_GPU:-1}"
export NB="${NB:-32}"
export RADII="${RADII:-22,38,48,95}"
export DENSE="${DENSE:-1}"
export DENSE_NKY="${DENSE_NKY:-6}"
export DENSE_NW="${DENSE_NW:-10}"

echo "=== hybrid (ky,w) outer-solver experiment ==="
echo "host: $(hostname)  date: $(date)  USE_GPU=${USE_GPU} NB=${NB} RADII=${RADII} DENSE=${DENSE} (${DENSE_NKY}x${DENSE_NW})"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    -t "${SLURM_CPUS_PER_TASK:-32}" ad/hybrid_experiment.jl

echo "=== hybrid experiment job done ==="
