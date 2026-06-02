#!/bin/bash -l
# SCAN_N=20 on 10 GPU nodes, 2 radii/node, 2 A100s/radius, with an MPS team of MPS_TEAM
# workers per radius split across its 2 GPUs (MPS_TEAM/2 clients per GPU). This stacks the
# two proven levers:
#   - data parallel across 2 GPUs/radius  (~1.7x, validated: 359 -> 211 s/radius)
#   - MPS within each GPU (8 worker contexts overlap Xgeev via Hyper-Q, ~2.9x single-GPU)
# Single-GPU 8-worker validation: 439.9 s -> 150.2 s/radius (2.93x), SFmin bit-exact.
# Footprint matches the 10-node Fortran CPU scan for a fair head-to-head.
#
#   sbatch batch_run_gacode_scan20_mps_10node_2gpu.sh
#
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -n 20
#SBATCH -t 01:30:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_s20_mps10x2
#SBATCH -o gacode_scan20_mps10x2_%j.out
#SBATCH -e gacode_scan20_mps10x2_%j.err
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -uo pipefail

# cudatoolkit MUST be >= 12.6 (cusolverDnXgeev). Keep 12.9.
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TJLFEP_FILE_ONLY=1
export USE_GPU=1
export TJLFEP_DEBUG=0
export TJLFEP_PRINTOUT=0
export INNER=mps_team
export MPS_TEAM="${MPS_TEAM:-16}"            # 16 workers/radius = 8 per GPU (2 GPUs/radius)
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-2}"   # 16 workers x 2 = 32 cpus/radius
# REQUIRED for MPS clients: avoids CUDA.jl forward-compat cuInit hang under MPS.
export JULIA_CUDA_USE_COMPAT=false
# Do NOT set JULIA_CUDA_MEMORY_POOL=none: the default stream-ordered pool scales fine under
# MPS (fresh-alloc and reuse both hit 3.2x in the micro-benchmark); pool=none serialises.
export TJLFEP_PROBE="${TJLFEP_PROBE:-0}"

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/src/DIIIDfiles/202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${TJLFEP_ROOT}/build/debug_nb32/input_scan20.TGLFEP}"
export OUT_DIR="${TJLFEP_ROOT}/build/gacode_scan20_${SLURM_JOB_ID}_tasks"

# Per-job MPS pipe/log dirs (each node's wrapper starts its own daemon here).
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"

cd "${TJLFEP_ROOT}/build"

echo "=== SCAN_N=20 on ${SLURM_NNODES:-?} nodes, 2 radii/node, 2 GPUs/radius, MPS_TEAM=${MPS_TEAM} (8/GPU) ==="
echo "host=$(hostname) OUT_DIR=${OUT_DIR} TGLFEP_FILE=${TGLFEP_FILE}"
nvidia-smi -L 2>/dev/null | head -4 || true

# 20 radii, 2/node. mps-pair-wrapper starts the per-node daemon (barrier on control pipe),
# assigns each task its GPU pair + SCAN_INDEX, then runs the coordinator which addprocs the
# MPS team across both GPUs.
srun --export=ALL --label -n "${SLURM_NTASKS:-20}" --ntasks-per-node=2 --cpu-bind=cores \
    ./mps-pair-wrapper.sh \
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    -t 2 run_gacode_scan20_mps_task.jl

echo "=== all ${SLURM_NTASKS:-20} tasks done; quitting MPS daemons + merging on CPU ==="
srun --export=ALL -n "${SLURM_NNODES:-10}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

export USE_GPU=0
# shellcheck source=julia_sysimage.inc.sh
source "${TJLFEP_ROOT}/build/julia_sysimage.inc.sh"
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${JULIA_SYSIMAGE_ARGS[@]}" \
    -t 8 merge_gacode_scan20_array.jl

echo "=== scan + merge done; outputs in ${OUT_DIR} ==="
