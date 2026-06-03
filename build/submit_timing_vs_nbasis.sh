#!/bin/bash -l
# Submit the timing-vs-nbasis sweep with the latest improvements + sysimages:
#   * Julia GPU : 5 nodes, MPS team + GPU sysimage (batch_time_scan20_julia_gpu.sh)
#   * Julia CPU : 10 nodes, SlurmClusterManager + full CPU sysimage (batch_time_scan20_julia_cpu.sh)
#   * Fortran   : 10 nodes CPU -- REUSED from prior runs (not resubmitted here).
#
# Output logs are named time_scan20_nb${nb}_julia_{cpu,gpu}_<jobid>.out so collect_scan20_timing.jl
# picks them up per nbasis. Job ids are recorded in timing_runs/last_timing_vs_nbasis.txt.
#
# Usage:
#   # CPU sysimage already built:
#   ./submit_timing_vs_nbasis.sh
#   # CPU sysimage still building (chain CPU jobs afterok the build job):
#   CPU_SYSIMAGE_BUILD_JOB=53858228 ./submit_timing_vs_nbasis.sh
#   # subset / GPU-only:
#   NB_LIST="6 32" ./submit_timing_vs_nbasis.sh
#   GPU_ONLY=1 ./submit_timing_vs_nbasis.sh

set -uo pipefail
cd "$(dirname "$0")"

NB_LIST="${NB_LIST:-6 8 16 32}"
TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
CPU_SYSIMG="${TJLFEP_ROOT}/build/TJLFEP_cpu_sysimage.so"
GPU_SYSIMG="${TJLFEP_ROOT}/build/TJLFEP_gpu_sysimage.so"
REC="${TJLFEP_ROOT}/build/timing_runs/last_timing_vs_nbasis.txt"
mkdir -p "$(dirname "${REC}")"
: > "${REC}"

[[ -f "${GPU_SYSIMG}" ]] || echo "WARNING: ${GPU_SYSIMG} missing -> GPU jobs would JIT (slower)."

# CPU jobs need the full CPU sysimage. If it isn't on disk yet, require a build job id to
# chain afterok so the CPU timing only runs once a verified image exists.
CPU_DEP_ARG=()
SUBMIT_CPU=1
if [[ "${GPU_ONLY:-0}" == "1" ]]; then
    SUBMIT_CPU=0
    echo "GPU_ONLY=1 -> skipping Julia CPU submits"
elif [[ ! -f "${CPU_SYSIMG}" ]]; then
    if [[ -n "${CPU_SYSIMAGE_BUILD_JOB:-}" ]]; then
        CPU_DEP_ARG=(--dependency=afterok:${CPU_SYSIMAGE_BUILD_JOB})
        echo "CPU sysimage not built yet -> CPU jobs depend on afterok:${CPU_SYSIMAGE_BUILD_JOB}"
    else
        SUBMIT_CPU=0
        echo "WARNING: ${CPU_SYSIMG} missing and no CPU_SYSIMAGE_BUILD_JOB set -> skipping CPU submits (would JIT)."
    fi
fi

for nb in ${NB_LIST}; do
    g=$(sbatch --parsable \
        --output="time_scan20_nb${nb}_julia_gpu_%j.out" \
        --error="time_scan20_nb${nb}_julia_gpu_%j.err" \
        --export=ALL,NB=${nb} batch_time_scan20_julia_gpu.sh)
    echo "nb=${nb} GPU  job=${g}"
    echo "nb=${nb} gpu ${g}" >> "${REC}"

    if [[ "${SUBMIT_CPU}" == "1" ]]; then
        c=$(sbatch --parsable "${CPU_DEP_ARG[@]}" \
            --output="time_scan20_nb${nb}_julia_cpu_%j.out" \
            --error="time_scan20_nb${nb}_julia_cpu_%j.err" \
            --export=ALL,NB=${nb} batch_time_scan20_julia_cpu.sh)
        echo "nb=${nb} CPU  job=${c}"
        echo "nb=${nb} cpu ${c}" >> "${REC}"
    fi
done

echo "=== recorded job ids -> ${REC} ==="
cat "${REC}"
echo "=== after runs complete: julia --project=${TJLFEP_ROOT} collect_scan20_timing.jl && ./plot_scan20_timing.sh ==="
