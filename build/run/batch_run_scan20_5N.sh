#!/bin/bash -l
# Production layout candidate 5N: SCAN_N=20 on 5 GPU nodes, 4 radii/node, 1 A100/radius, MPS
# team of 8 workers x 2 threads per radius (8 clients/GPU). Densest packing -> lowest node-hours
# if wall stays acceptable. All 20 radii run in one parallel wave. SCAN_INDEX = global procid+1.
#
#   cd build && sbatch run/batch_run_scan20_5N.sh
#
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -n 20
#SBATCH -t 00:45:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_s20_5N
#SBATCH -o gacode_scan20_5N_%j.out
#SBATCH -e gacode_scan20_5N_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TJLFEP_FILE_ONLY=1 USE_GPU=1 TJLFEP_DEBUG=0 TJLFEP_PRINTOUT=0
export INNER=mps_team
export GPUS_PER_RADIUS=1
export MPS_TEAM="${MPS_TEAM:-8}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"
export JULIA_CUDA_USE_COMPAT=false
export TJLFEP_PROBE="${TJLFEP_PROBE:-0}"

# Optional GPU sysimage for BOTH master + workers. Workers pick it up via TJLFEP_GPU_SYSIMAGE
# in run_gacode_scan20_mps_task.jl; the master gets the flag below. Unset -> JIT (reproducible
# baseline). Set to build/TJLFEP_gpu_generic_sysimage.so to skip ~110 s/team of cold JIT.
GPU_SYSIMG="${TJLFEP_GPU_SYSIMAGE:-}"
if [[ -n "${GPU_SYSIMG}" && -f "${GPU_SYSIMG}" ]]; then
    export TJLFEP_GPU_SYSIMAGE="${GPU_SYSIMG}"
    MASTER_SYSIMG_ARGS=(--sysimage="${GPU_SYSIMG}")
    echo "GPU sysimage (master+workers): ${GPU_SYSIMG}"
else
    MASTER_SYSIMG_ARGS=()
    echo "GPU sysimage: none (JIT)"
fi

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${CASE_DIR}/input_scan20_nb32.TGLFEP}"
export OUT_DIR="${TJLFEP_ROOT}/build/gacode_scan20_5N_${SLURM_JOB_ID}_tasks"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

cd "${TJLFEP_ROOT}/build"
echo "=== 5N: SCAN_N=20 on ${SLURM_NNODES:-5} nodes, 4 radii/node, 1 GPU/radius, MPS_TEAM=${MPS_TEAM} (8/GPU) x ${JULIA_WORKER_THREADS}t ==="
t_start=$(date +%s)

srun --export=ALL --label -n "${SLURM_NTASKS:-20}" --ntasks-per-node=4 --cpu-bind=cores \
    common/mps-scan-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t "${JULIA_WORKER_THREADS}" common/run_gacode_scan20_mps_task.jl

t_end=$(date +%s)
echo "=== all tasks done in $((t_end - t_start)) s (incl. spawn+load); quitting MPS daemons + merging ==="
srun --export=ALL -n "${SLURM_NNODES:-5}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
source "${TJLFEP_ROOT}/build/common/julia_sysimage.inc.sh"
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${JULIA_SYSIMAGE_ARGS[@]}" -t 8 common/merge_gacode_scan20_array.jl

echo "=== 5N scan + merge done; outputs in ${OUT_DIR} ==="
