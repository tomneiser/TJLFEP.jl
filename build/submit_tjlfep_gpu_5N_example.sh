#!/bin/bash -l
# =============================================================================
# EXAMPLE: TJLFEP (Julia, GPU) SCAN_N=20 submit for 5 GPU nodes, with sysimage.
#
# This is the GPU/Julia analogue of the Fortran TGLF-EP submit
# (see batch_debug_nb6_fortran_scan20_10n.sh). Side-by-side:
#
#   Fortran TGLF-EP (CPU, 10 nodes)          TJLFEP (Julia, GPU, 5 nodes)
#   -------------------------------          ----------------------------
#   DRIVER=$TGLFEP_DIR/TGLFEP_driver         DRIVER=$TJLFEP_ROOT/build/run_gacode_scan20_mps_task.jl
#   srun -n 1280 "$DRIVER"                   srun -n 20 mps-scan-wrapper.sh julia ... "$DRIVER"
#   input.TGLFEP + input.gacode (CASE_DIR)   input.TGLFEP + input.gacode (CASE_DIR)  [same files]
#   out.TGLFEP  (final SFmin profile)        finalize_gacode_scan -> merged SFmin profile in OUT_DIR
#
# Layout: 20 radii on 5 nodes = 4 radii/node, 1 A100/radius, an MPS team of 8 worker
# processes per GPU x 2 CPU threads each. All 20 radii run in one parallel wave;
# SCAN_INDEX = (SLURM procid + 1). SCAN_N in input.TGLFEP MUST be 20 to match `-n 20`.
#
# Usage:  edit the CONFIG paths below, then:  sbatch submit_tjlfep_gpu_5N_example.sh
#         (or override per-run, e.g.  TJLFEP_ROOT=... CASE_DIR=... sbatch ... )
# =============================================================================
#SBATCH -A m3739_g                 # <-- your GPU allocation (NERSC GPU projects end in "_g")
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -n 20
#SBATCH -t 00:45:00
#SBATCH -C gpu
#SBATCH -J TJLFEP_gpu_5N
#SBATCH -o tjlfep_gpu_5N_%j.out
#SBATCH -e tjlfep_gpu_5N_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

set -uo pipefail

# -----------------------------------------------------------------------------
# CONFIG -- edit these for your environment (the paths below are hypothetical).
# -----------------------------------------------------------------------------
# 1) Your TJLFEP.jl checkout (contains build/, src/, Project.toml).
TJLFEP_ROOT="${TJLFEP_ROOT:-/global/cfs/cdirs/m3739/$USER/TJLFEP}"

# 2) Prebuilt GPU sysimage (file-only). Build it ONCE with batch_build_gpu_sysimage_generic.sh,
#    then keep the .so on a shared, non-purged path (CFS is good; $PSCRATCH is purged).
#    The same .so works for any node count -- nothing about the layout is baked in.
#    Leave empty (SYSIMAGE="") to fall back to JIT (~110 s/team slower per radius).
SYSIMAGE="${TJLFEP_GPU_SYSIMAGE:-/global/cfs/cdirs/m3739/$USER/sysimages/TJLFEP_gpu_generic_sysimage.so}"

# 3) Case directory holding input.gacode (equilibrium + profiles).
CASE_DIR="${CASE_DIR:-${TJLFEP_ROOT}/examples/DIIID_202017C42_500ms_v3.1}"

# 4) TGLF-EP scan-control input (input.TGLFEP with SCAN_N=20) -- the same file the
#    Fortran TGLFEP_driver reads.
TGLFEP_INPUT="${TGLFEP_INPUT:-${CASE_DIR}/input_scan20_nb32.TGLFEP}"

# The Julia "driver" (analogue of $TGLFEP_DIR/TGLFEP_driver), its per-node launch
# wrapper (starts MPS + pins GPUs + sets SCAN_INDEX), and the result-merge driver.
DRIVER="${TJLFEP_ROOT}/build/run_gacode_scan20_mps_task.jl"
WRAPPER="${TJLFEP_ROOT}/build/mps-scan-wrapper.sh"
MERGE="${TJLFEP_ROOT}/build/merge_gacode_scan20_array.jl"

# -----------------------------------------------------------------------------
# Runtime environment.
# -----------------------------------------------------------------------------
module load cudatoolkit/12.9
module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"

export TJLFEP_FILE_ONLY=1 USE_GPU=1 TJLFEP_DEBUG=0 TJLFEP_PRINTOUT=0
export INNER=mps_team
export GPUS_PER_RADIUS=1            # 1 A100 per radius => 4 radii/node on 5 nodes
export MPS_TEAM=8                   # MPS worker processes sharing each GPU
export JULIA_WORKER_THREADS=2       # CPU threads per worker (matrix assembly)
export JULIA_CUDA_USE_COMPAT=false

# Inputs consumed by the driver + merge.
export CASE_DIR
export GACODE_FILE="${CASE_DIR}/input.gacode"
export TGLFEP_FILE="${TGLFEP_INPUT}"
# Outputs only -- never written into CASE_DIR (keeps the case dir read-only/archival).
export OUT_DIR="${TJLFEP_ROOT}/build/tjlfep_gpu_5N_${SLURM_JOB_ID:-local}_tasks"

# Per-job MPS pipe/log dirs (the wrapper starts one MPS daemon per node).
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps.${SLURM_JOB_ID:-$$}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log.${SLURM_JOB_ID:-$$}"

# Sysimage applied to BOTH the master task and its MPS workers (skips JIT).
if [[ -n "${SYSIMAGE}" && -f "${SYSIMAGE}" ]]; then
    export TJLFEP_GPU_SYSIMAGE="${SYSIMAGE}"
    MASTER_SYSIMG_ARGS=(--sysimage="${SYSIMAGE}")
    echo "GPU sysimage (master+workers): ${SYSIMAGE}"
else
    MASTER_SYSIMG_ARGS=()
    echo "GPU sysimage: none found at '${SYSIMAGE}' -> running with JIT"
fi

# -----------------------------------------------------------------------------
# Validate inputs (mirrors the Fortran submit's pre-flight checks).
# -----------------------------------------------------------------------------
for f in "${DRIVER}" "${WRAPPER}" "${MERGE}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}"; exit 1; }
done
[[ -f "${GACODE_FILE}" ]] || { echo "ERROR: missing input.gacode at ${GACODE_FILE}"; exit 1; }
[[ -f "${TGLFEP_FILE}"  ]] || { echo "ERROR: missing input.TGLFEP at ${TGLFEP_FILE}"; exit 1; }
mkdir -p "${OUT_DIR}"

echo "=== TJLFEP GPU SCAN_N=20 | 5 nodes x 4 radii, 1 A100/radius, MPS_TEAM=${MPS_TEAM} x ${JULIA_WORKER_THREADS}t ==="
echo "host: $(hostname)   date: $(date)"
echo "TJLFEP_ROOT=${TJLFEP_ROOT}"
echo "DRIVER=${DRIVER}"
echo "CASE_DIR=${CASE_DIR}"
echo "GACODE_FILE=${GACODE_FILE}"
echo "TGLFEP_FILE=${TGLFEP_FILE}"
echo "OUT_DIR=${OUT_DIR}"

cd "${TJLFEP_ROOT}/build"
t_start=$(date +%s)

# Launch: 20 SLURM tasks (1 radius each). mps-scan-wrapper.sh starts the per-node MPS
# control daemon, pins this task's GPU(s), sets SCAN_INDEX, then exec's the Julia driver.
# This is the analogue of `srun -n 1280 "$TGLFEP_DIR/TGLFEP_driver"` in the Fortran submit.
srun --export=ALL --label -n "${SLURM_NTASKS:-20}" --ntasks-per-node=4 --cpu-bind=cores \
    "${WRAPPER}" \
    stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" \
    -t "${JULIA_WORKER_THREADS}" "${DRIVER}"

t_end=$(date +%s)
echo "=== scan done in $((t_end - t_start)) s (incl. worker spawn + image load) ==="

# Stop the per-node MPS daemons.
srun --export=ALL -n "${SLURM_NNODES:-5}" --ntasks-per-node=1 \
    bash -c 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' || true

# Merge the 20 per-radius outputs into the final SFmin profile (the out.TGLFEP analogue).
export USE_GPU=0
stdbuf -oL -eL julia --startup-file=no --project="${TJLFEP_ROOT}" \
    "${MASTER_SYSIMG_ARGS[@]}" -t 8 "${MERGE}"

echo "=== done; per-radius outputs + merged SFmin profile in ${OUT_DIR} ==="
