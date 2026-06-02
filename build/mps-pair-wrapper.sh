#!/bin/bash
# Per-node MPS launch + 2-GPU-per-radius pinning wrapper for the 10-node production scan.
#
# Layout (per node): 2 SLURM tasks (radii), 4 A100s. Task localid 0 -> GPUs 0,1;
# localid 1 -> GPUs 2,3. Each task is a coordinator that addprocs MPS_TEAM workers
# round-robined across its 2 GPUs (TEAM_GPUS), so each GPU hosts MPS_TEAM/2 MPS clients
# whose Xgeev solves overlap via Hyper-Q.
#
# LAUNCH-ORDER FIX (critical): the MPS control daemon MUST be up before ANY client calls
# cuInit, or clients fall back to private contexts and time-slice (this silently caps the
# whole speedup at ~1.3x). Node-local rank 0 starts the daemon; ALL ranks then BLOCK on a
# poll of "${CUDA_MPS_PIPE_DIRECTORY}/control" before exec'ing the (CUDA-initializing) cmd.
#
#   srun -n <radii> --ntasks-per-node=2 [--gpus-per-node=4, NO --gpus-per-task] mps-pair-wrapper.sh <cmd>...

set -uo pipefail

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"
CONTROL="${CUDA_MPS_PIPE_DIRECTORY}/control"

LOCALID="${SLURM_LOCALID:-0}"

# Node-local rank 0 starts the daemon with ALL node GPUs visible (the control daemon spawns
# one server per GPU on first client connection).
if [ "${LOCALID}" -eq 0 ]; then
    if [ ! -e "${CONTROL}" ]; then
        mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
        CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-0,1,2,3}}" \
            nvidia-cuda-mps-control -d
    fi
fi

# BARRIER: every rank waits until the control pipe exists (daemon up) before any CUDA init.
for _ in $(seq 1 120); do
    [ -e "${CONTROL}" ] && break
    sleep 0.5
done
if [ ! -e "${CONTROL}" ]; then
    echo "[mps-pair-wrapper localid=${LOCALID}] ERROR: MPS control pipe ${CONTROL} never appeared" >&2
    exit 1
fi
sleep 3   # small extra margin so the daemon is fully ready to accept clients

# Assign this task's GPU pair from its node-local id (global device indices on this node).
case "${LOCALID}" in
    0) PAIR="0,1" ;;
    1) PAIR="2,3" ;;
    *) PAIR="$(( LOCALID*2 )),$(( LOCALID*2 + 1 ))" ;;
esac
# Coordinator sees both of its GPUs; workers will each pin to one via TEAM_GPUS round-robin.
export CUDA_VISIBLE_DEVICES="${PAIR}"
export TEAM_GPUS="${PAIR}"
# One radius per global task.
export SCAN_INDEX="$(( ${SLURM_PROCID:-0} + 1 ))"

echo "[mps-pair-wrapper] node=$(hostname) localid=${LOCALID} procid=${SLURM_PROCID:-0} -> SCAN_INDEX=${SCAN_INDEX} GPUs=${PAIR}"
exec "$@"
