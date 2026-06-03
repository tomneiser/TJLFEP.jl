#!/bin/bash -l
# Production layout candidate 10N (also the FAIR head-to-head vs the 10-node Fortran CPU scan):
# SCAN_N=20 on 10 GPU nodes, 2 radii/node, 2 A100s/radius, MPS team of 16 workers (8/GPU) x 2
# threads. All 20 radii run in one parallel wave. SCAN_INDEX = global procid + 1.
#
#   sbatch batch_run_scan20_10N.sh
#
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -n 20
#SBATCH -t 00:45:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_s20_10N
#SBATCH -o gacode_scan20_10N_%j.out
#SBATCH -e gacode_scan20_10N_%j.err
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TJLFEP_FILE_ONLY=1 USE_GPU=1 TJLFEP_DEBUG=0 TJLFEP_PRINTOUT=0
export INNER=mps_team
export GPUS_PER_RADIUS=2
export MPS_TEAM="${MPS_TEAM:-16}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export TJLFEP_PROBE="${TJLFEP_PROBE:-0}"

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/src/DIIIDfiles/202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${TJLFEP_ROOT}/build/debug_nb32/input_scan20.TGLFEP}"
export OUT_DIR="${TJLFEP_ROOT}/build/gacode_scan20_10N_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

cd "${TJLFEP_ROOT}/build"
echo "=== 10N: SCAN_N=20 on ${SLURM_NNODES:-10} nodes, 2 radii/node, 2 GPUs/radius, MPS_TEAM=${MPS_TEAM} (8/GPU) x ${JULIA_WORKER_THREADS}t ==="
t_start=$(date +%s)

srun --export=ALL --label -n "${SLURM_NTASKS:-20}" --ntasks-per-node=2 --cpu-bind=cores \
    ./mps-scan-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    -t "${JULIA_WORKER_THREADS}" run_gacode_scan20_mps_task.jl

t_end=$(date +%s)
echo "=== all tasks done in $((t_end - t_start)) s (incl. spawn+load); quitting MPS daemons + merging ==="
srun --export=ALL -n "${SLURM_NNODES:-10}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
source "${TJLFEP_ROOT}/build/julia_sysimage.inc.sh"
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${JULIA_SYSIMAGE_ARGS[@]}" -t 8 merge_gacode_scan20_array.jl

echo "=== 10N scan + merge done; outputs in ${OUT_DIR} ==="
