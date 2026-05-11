#!/bin/bash -l
# Usage: bash submit_sweep.sh [gpu|cpu] [N_BASIS values...]
# Example: bash submit_sweep.sh gpu 4 8 12 16 32
#          bash submit_sweep.sh cpu 4 8
# N_BASES=(4 8 12 16 24 32)
N_BASES=(32)
NODE=cpu
SCAN_N=20
runType=regular
# runType=premium

DEVICE="${1:-$NODE}"
DEVICE="${DEVICE,,}"   # lowercase
shift 2>/dev/null || true

TIME="00:30:00"
TOT_THREADS=0
if [[ "$DEVICE" == "cpu" ]]; then
    TIME="07:59:00"
    TOT_THREADS=1280
else
    TIME="05:59:00"
    TOT_THREADS=40
fi

if [[ $# -gt 0 ]]; then
    N_BASIS_LIST=("$@")
else
    N_BASIS_LIST=("${N_BASES[@]}")
fi

for N_BASIS in "${N_BASIS_LIST[@]}"; do
    # Use a unique temp name (timestamp+PID) so concurrent submissions don't collide
    UNIQUE_ID="$(date +%s)_$$"
    DEVICE_UPPER="${DEVICE^^}"
    OUTDIR_TMP="${DEVICE_UPPER}_n${N_BASIS}_${SCAN_N}_${UNIQUE_ID}"
    JL_FILE="DIIID_juliaValidation_${DEVICE}_n${N_BASIS}_${SCAN_N}_${UNIQUE_ID}.jl"

    # Create the output dir early so the script can live inside it
    mkdir -p "$OUTDIR_TMP"

    # Patch N_BASIS and write directly into the output dir
    sed "s/^N_BASIS = .*/N_BASIS = ${N_BASIS}/" DIIID_juliaValidation.jl > "${OUTDIR_TMP}/${JL_FILE}"
    # Patch nthreads
    sed -i "s/^tot = .*/tot = ${TOT_THREADS}/" "${OUTDIR_TMP}/${JL_FILE}"

    # Enable or disable CUDA loading on workers
    if [[ "$DEVICE" == "gpu" ]]; then
        sed -i "s|^# @everywhere @time using CUDA|@everywhere @time using CUDA|" "${OUTDIR_TMP}/${JL_FILE}"
    else
        sed -i "s|^@everywhere @time using CUDA|# @everywhere @time using CUDA|" "${OUTDIR_TMP}/${JL_FILE}"
    fi

    # Symlink the script into DIIIDfiles root so the batch job can find it by name
    ln -sf "${OUTDIR_TMP}/${JL_FILE}" "$JL_FILE"

    # Create a temp batch script pointing to this Julia file, with correct -C constraint
    TMP_BATCH="batchRun_${DEVICE}_n${N_BASIS}_${SCAN_N}_${UNIQUE_ID}.sh"
    sed -e "s/DIIID_juliaValidation\.jl/${JL_FILE}/" \
        -e "s/^#SBATCH -C .*/#SBATCH -C ${DEVICE}/" \
        -e "s/^#SBATCH -t .*/#SBATCH -t ${TIME}/" \
        -e "s/^#SBATCH -n .*/#SBATCH -n ${TOT_THREADS}/" \
        -e "s/^#SBATCH -N .*/#SBATCH -N 10/" \
        -e "s/^#SBATCH -q .*/#SBATCH -q ${runType}/" \
        batchRun.sh > "$TMP_BATCH"

    SUBMIT_OUT=$(sbatch "$TMP_BATCH")
    JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
    rm "$TMP_BATCH"

    # Rename the output dir to the actual job ID
    OUTDIR_FINAL="${DEVICE_UPPER}_n${N_BASIS}_${SCAN_N}_${JOB_ID}"
    mv "$OUTDIR_TMP" "$OUTDIR_FINAL"

    # Update the symlink to point into the renamed dir
    ln -sf "${OUTDIR_FINAL}/${JL_FILE}" "$JL_FILE"

    echo "Submitted ${DEVICE_UPPER} N_BASIS=${N_BASIS} job=${JOB_ID} outdir=${OUTDIR_FINAL}"
done
