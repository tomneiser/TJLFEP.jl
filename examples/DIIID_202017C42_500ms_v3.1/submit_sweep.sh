#!/bin/bash -l
# Usage: bash submit_sweep.sh [gpu|cpu] [N_BASIS values...]
# Example: bash submit_sweep.sh gpu 4 8 12 16 32
#          bash submit_sweep.sh cpu 4 8
# N_BASES=(4 8 12 16 24 32)
N_BASES=(12)
NODE=cpu
SCAN_N=20
# SCAN_N=1
# customTag=none
customTag=185
# submitType=regular
submitType=premium
# submitType=debug

DEVICE="${1:-$NODE}"
DEVICE="${DEVICE,,}"   # lowercase
shift 2>/dev/null || true

TIME="00:30:00"
if [[ "$DEVICE" == "cpu" ]]; then
    # TIMES=("00:10:00" "00:15:00" "00:30:00" "01:59:00" "03:59:00" "06:59:00")
    TIMES=("03:59:00" "06:59:00")
    N_NODES=10
else
    TIME="00:39:00"
    N_NODES=$(((SCAN_N + 3) / 4))   # 20 workers / 4 GPUs per node = 5 nodes (+3 for ceiling)
    # skeptical about this one, more nodes still gives us more threads which we need
    # only helps if threads are shared between nodes, but I thought NERSC does that
fi

if [[ $# -gt 0 ]]; then
    N_BASIS_LIST=("$@")
else
    N_BASIS_LIST=("${N_BASES[@]}")
fi

[[ -n "$customTag" && "$customTag" != "none" ]] && TAG_PART="_${customTag}" || TAG_PART=""

t=0
for N_BASIS in "${N_BASIS_LIST[@]}"; do
    # Use a unique temp name (timestamp+PID) so concurrent submissions don't collide
    UNIQUE_ID="$(date +%s)_$$"
    DEVICE_UPPER="${DEVICE^^}"
    OUTDIR_TMP="${DEVICE_UPPER}_n${N_BASIS}${TAG_PART}_${SCAN_N}_${UNIQUE_ID}"
    JL_FILE="DIIID_juliaValidation_${DEVICE}_n${N_BASIS}${TAG_PART}_${SCAN_N}_${UNIQUE_ID}.jl"

    # Create the output dir early so the script can live inside it
    mkdir -p "$OUTDIR_TMP"

    # Patch N_BASIS and write directly into the output dir
    sed "s/^N_BASIS = .*/N_BASIS = ${N_BASIS}/" DIIID_juliaValidation.jl > "${OUTDIR_TMP}/${JL_FILE}"

    # Patch customTag
    sed -i "s|^customTag = .*|customTag = \"${customTag}\"|" "${OUTDIR_TMP}/${JL_FILE}"

    # Enable or disable CUDA loading on workers
    if [[ "$DEVICE" == "gpu" ]]; then
        sed -i "s|^# @everywhere @time using CUDA|@everywhere @time using CUDA|" "${OUTDIR_TMP}/${JL_FILE}"
        sed -i "s|^use_gpu = .*|use_gpu = true|" "${OUTDIR_TMP}/${JL_FILE}"
    else
        sed -i "s|^@everywhere @time using CUDA|# @everywhere @time using CUDA|" "${OUTDIR_TMP}/${JL_FILE}"
        sed -i "s|^use_gpu = .*|use_gpu = false|" "${OUTDIR_TMP}/${JL_FILE}"
        TIME=${TIMES[$t]}
        ((t++))
    fi

    # Symlink the script into DIIIDfiles root so the batch job can find it by name
    ln -sf "${OUTDIR_TMP}/${JL_FILE}" "$JL_FILE"

    # Create a temp batch script pointing to this Julia file, with correct -C constraint
    TMP_BATCH="batchRun_${DEVICE}_n${N_BASIS}_${SCAN_N}_${UNIQUE_ID}.sh"
    sed -e "s/DIIID_juliaValidation\.jl/${JL_FILE}/" \
        -e "s/^#SBATCH -C .*/#SBATCH -C ${DEVICE}/" \
        -e "s/^#SBATCH -t .*/#SBATCH -t ${TIME}/" \
        -e "s/^#SBATCH -n .*/#SBATCH -n ${SCAN_N}/" \
        -e "s/^#SBATCH -N .*/#SBATCH -N ${N_NODES}/" \
        -e "s/^#SBATCH -q .*/#SBATCH -q ${submitType}/" \
        batchRun.sh > "$TMP_BATCH"
    # total number of threads set inside Julia script,
    # -n is the total number of tasks, SlurmManager makes each one a Julia worker

    if [[ "$DEVICE" == "gpu" ]]; then
        sed -i "/#SBATCH -C gpu/a #SBATCH --gpus-per-task=1" "$TMP_BATCH"
    fi

    SUBMIT_OUT=$(sbatch "$TMP_BATCH")
    JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
    rm "$TMP_BATCH"

    # Rename the output dir to the actual job ID
    OUTDIR_FINAL="${DEVICE_UPPER}_n${N_BASIS}${TAG_PART}_${SCAN_N}_${JOB_ID}"
    mv "$OUTDIR_TMP" "$OUTDIR_FINAL"

    # Update the symlink to point into the renamed dir
    ln -sf "${OUTDIR_FINAL}/${JL_FILE}" "$JL_FILE"

    echo "Submitted ${DEVICE_UPPER} N_BASIS=${N_BASIS} job=${JOB_ID} outdir=${OUTDIR_FINAL}"
done
