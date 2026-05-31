#!/bin/bash -l
# Submit N_BASIS=8 SCAN_N=20 timing: Fortran + Julia CPU (10n premium) + Julia GPU (5n).
set -euo pipefail

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}/build"
mkdir -p timing_runs

FRT=$(sbatch --parsable batch_time_scan20_nb8_fortran.sh)
CPU=$(sbatch --parsable batch_time_scan20_nb8_julia_cpu.sh)
GPU=$(sbatch --parsable batch_time_scan20_nb8_julia_gpu.sh)

echo "Submitted N_BASIS=8 SCAN_N=20 timing:"
echo "  Fortran 10n/20t (premium):   ${FRT}  -> time_scan20_nb8_fortran_${FRT}.out"
echo "  Julia CPU 10n/20t (premium): ${CPU}  -> time_scan20_nb8_julia_cpu_${CPU}.out"
echo "  Julia GPU 5n/20t:            ${GPU}  -> time_scan20_nb8_julia_gpu_${GPU}.out"
echo "${FRT} ${CPU} ${GPU}" > timing_runs/last_time_scan20_nb8_jobs.txt
