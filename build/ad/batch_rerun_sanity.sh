#!/bin/bash -l
# (A) grid-floor-guarded adf1 vs lean DIRECT-20 on 4 radii + (B) IR=95 fine-faithful-grid sanity.
# ONE GPU, MPS team=4, live source.  cd build && sbatch ad/batch_rerun_sanity.sh
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J tjlfep_rerun_sanity
#SBATCH -o rerun_sanity_%j.out
#SBATCH -e rerun_sanity_%j.err
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
export DIRECT20_EVALS="${DIRECT20_EVALS:-20}"
export FINE_NKY="${FINE_NKY:-12}"
export FINE_NW="${FINE_NW:-20}"
export INNER="${INNER:-mps_team}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

echo "=== (A) rerun grid-floor adf1 vs DIRECT-20 + (B) IR=95 fine-grid sanity ==="
echo "host: $(hostname)  date: $(date)  NB=${NB} RADII=${RADII} D20=${DIRECT20_EVALS} FINE=${FINE_NKY}x${FINE_NW} INNER=${INNER} MPS_TEAM=${MPS_TEAM}"
nvidia-smi -L 2>/dev/null | head -1 || true

if [[ "${INNER}" == "mps_team" ]]; then
    export GPUS_PER_RADIUS=1
    srun -n 1 --ntasks-per-node=1 --gpus-per-node=1 \
        common/mps-scan-wrapper.sh \
        julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t 4 ad/rerun_sanity_experiment.jl
else
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t "${SLURM_CPUS_PER_TASK:-32}" ad/rerun_sanity_experiment.jl
fi

echo "=== rerun+sanity job done ==="
