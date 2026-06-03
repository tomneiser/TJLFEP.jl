#!/bin/bash -l
# A/B the GPU-worker sysimage on one node (1 radius, 4 GPUs, 16 workers x 4 threads, MPS team):
#   run A: no sysimage  (workers JIT-compile the per-combo path -> ~145 s/radius cold)
#   run B: --sysimage=TJLFEP_gpu_sysimage.so (baked -> expect ~35-60 s/radius)
# Same node + shared MPS daemon, so the only difference is the worker sysimage. The
# per-radius "OK ... in Xs" line is timed after worker spawn, so it isolates the JIT.
# SCAN_INDEX=2, golden SFmin=0.6249450209778226.
#
#   sbatch batch_test_sysimage_ab_4gpu.sh
#
#SBATCH -A m3739_g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -C gpu
#SBATCH -J sysimg_ab
#SBATCH -o sysimg_ab_%j.out
#SBATCH -e sysimg_ab_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4

set -uo pipefail

module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia"

export TJLFEP_FILE_ONLY=1 USE_GPU=1 TJLFEP_DEBUG=0 TJLFEP_PRINTOUT=0
export INNER=mps_team
export MPS_TEAM="${MPS_TEAM:-16}"
export JULIA_WORKER_THREADS="${JULIA_WORKER_THREADS:-4}"
export JULIA_CUDA_USE_COMPAT=false
export TEAM_GPUS="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SCAN_INDEX="${SCAN_INDEX:-2}"

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
export CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/src/DIIIDfiles/202017C42_500ms_v3.1}"
export GACODE_FILE="${GACODE_FILE:-${CASE_DIR}/input.gacode}"
export TGLFEP_FILE="${TGLFEP_FILE:-${TJLFEP_ROOT}/build/debug_nb32/input_scan20.TGLFEP}"
SYSIMG="${TJLFEP_ROOT}/build/TJLFEP_gpu_sysimage.so"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cd "${TJLFEP_ROOT}/build"
echo "=== sysimage A/B: 1 node, 1 radius (SCAN_INDEX=${SCAN_INDEX}), 4 GPUs, 16w x 4t ==="

CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-cuda-mps-control -d
sleep 5
echo "MPS server up: $(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null)"

echo "----- run A: NO sysimage (JIT) -----"
unset TJLFEP_GPU_SYSIMAGE || true
export OUT_DIR="${TJLFEP_ROOT}/build/sysimg_ab_${SLURM_JOB_ID}_nosys_tasks"
ta0=$(date +%s)
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" -t 4 run_gacode_scan20_mps_task.jl 2>&1 \
    | grep -E "OK scan_index|worker sysimage|team="
echo "  run A total (incl spawn+load): $(( $(date +%s) - ta0 )) s"

echo "----- run B: WITH sysimage -----"
export TJLFEP_GPU_SYSIMAGE="${SYSIMG}"
export OUT_DIR="${TJLFEP_ROOT}/build/sysimg_ab_${SLURM_JOB_ID}_sys_tasks"
tb0=$(date +%s)
stdbuf -oL -eL julia --startup-file=no --sysimage="${SYSIMG}" --project="${TJLFEP_ROOT}" -t 4 run_gacode_scan20_mps_task.jl 2>&1 \
    | grep -E "OK scan_index|worker sysimage|team="
echo "  run B total (incl spawn+load): $(( $(date +%s) - tb0 )) s"

echo quit | nvidia-cuda-mps-control 2>/dev/null || true
echo "=== compare OK-line (per-radius compute): A(JIT ~145s) vs B(sysimage). total includes spawn+load. ==="
