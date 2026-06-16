#!/bin/bash -l
# Offline accuracy-per-eval experiment for a DIRECT (DIviding RECTangles) global (ky,w) search
# vs the fixed 4x8 grid methods (robust refine=0, critical_factor_confirm) and a dense faithful
# truth. Key question: does DIRECT's adaptive resolution recover the OFF-NODE basin that every
# fixed 4x8 method missed at DIII-D IR=48 (~+90% vs the 6x10 dense truth), within a fixed cheap-
# eval budget? Runs from LIVE SOURCE (no sysimage — critical_factor_direct/NLopt postdate the
# baked image) on ONE GPU so eigensolves are fast; JITs once (~15 min) then runs the radius sweep.
# Accuracy (eval COUNTS, sfmin error) is hardware-independent; GPU is just for turnaround.
#   cd build && sbatch ad/batch_direct_experiment.sh
#SBATCH -A m3739_g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -J tjlfep_direct_exp
#SBATCH -o direct_experiment_%j.out
#SBATCH -e direct_experiment_%j.err
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

# Discriminating radii: 22 (easy interior control), 38 (strong-drive floor), 48 & 95 (the AD-
# threads spikes where the coarse grid mis-locates the basin). 6x10 dense truth is the off-grid
# reference. DIRECT_EVALS is the cheap-eval budget (each is a DIRECT_NEIG-pt IFLUX=false hull scan).
export USE_GPU="${USE_GPU:-1}"
export NB="${NB:-32}"
export RADII="${RADII:-22,38,48,95}"
export DENSE="${DENSE:-1}"
export DENSE_NKY="${DENSE_NKY:-6}"
export DENSE_NW="${DENSE_NW:-10}"
export DIRECT_EVALS="${DIRECT_EVALS:-40}"
export DIRECT_NEIG="${DIRECT_NEIG:-24}"
export INNER="${INNER:-threads}"
export MPS_TEAM="${MPS_TEAM:-4}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"

echo "=== DIRECT (ky,w) global-search experiment ==="
echo "host: $(hostname)  date: $(date)"
echo "USE_GPU=${USE_GPU} NB=${NB} RADII=${RADII} DENSE=${DENSE} (${DENSE_NKY}x${DENSE_NW}) DIRECT_EVALS=${DIRECT_EVALS} NEIG=${DIRECT_NEIG} INNER=${INNER} MPS_TEAM=${MPS_TEAM}"
nvidia-smi -L 2>/dev/null | head -1 || true

if [[ "${INNER}" == "mps_team" ]]; then
    # MPS path: launch one task under the wrapper (starts the node MPS daemon, pins 1 GPU). The
    # master spawns MPS_TEAM client workers; their Xgeev eigensolves overlap on the GPU via Hyper-Q.
    export GPUS_PER_RADIUS=1
    srun -n 1 --ntasks-per-node=1 --gpus-per-node=1 \
        common/mps-scan-wrapper.sh \
        julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t 4 ad/direct_experiment.jl
else
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
        -t "${SLURM_CPUS_PER_TASK:-32}" ad/direct_experiment.jl
fi

echo "=== direct experiment job done ==="
