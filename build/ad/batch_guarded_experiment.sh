#!/bin/bash -l
# Threads-timed validation of the guarded :ad path (critical_factor_ad_guarded): does detect-and-
# reseed / multistart recover DIRECT-class accuracy at the IR=48 off-node spike while keeping ~:ad
# wallclock? Runs from LIVE SOURCE (critical_factor_ad_guarded postdates the baked image) on ONE
# GPU, single process, :threads (no MPS team — :ad-class workloads favor threads). JITs once (~15
# min) then sweeps the radii. Compares against grid/dense/DIRECT references baked into the .jl.
#   cd build && sbatch ad/batch_guarded_experiment.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -J tjlfep_guarded_exp
#SBATCH -o guarded_experiment_%j.out
#SBATCH -e guarded_experiment_%j.err
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

export USE_GPU="${USE_GPU:-1}"
export NB="${NB:-32}"
export RADII="${RADII:-22,38,48,95}"

echo "=== guarded :ad (threads) validation ==="
echo "host: $(hostname)  date: $(date)  USE_GPU=${USE_GPU} NB=${NB} RADII=${RADII}"
nvidia-smi -L 2>/dev/null | head -1 || true

stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    -t "${SLURM_CPUS_PER_TASK:-32}" ad/guarded_experiment.jl

echo "=== guarded experiment job done ==="
