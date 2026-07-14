#!/bin/bash -l
# Build the full CPU sysimage (TJLF + TJLFEP, file-only) on a CPU node, with the CPU
# eigensolve path precompiled. Output: build/TJLFEP_cpu_sysimage.so
#
#   cd build && sbatch sysimage/batch_build_cpu_sysimage.sh
#
#SBATCH -A m3739
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -C cpu
#SBATCH -J TJLFEP_cpu_sysimg
#SBATCH -o build_cpu_sysimage_%j.out
#SBATCH -e build_cpu_sysimage_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

set -euo pipefail

module load julia/1.11.7
export JULIA_DEPOT_PATH="${PSCRATCH}/.julia${JULIA_DEPOT_PATH:+:${JULIA_DEPOT_PATH}}"
mkdir -p "${JULIA_DEPOT_PATH%%:*}/compiled"

# File-only image: TJLFEP_FILE_ONLY=1 so TJLFEP's baked _FILE_ONLY const is true and the
# FUSE/IMAS stack is NOT pulled in (the timing scan uses the file/gacode path).
export TJLFEP_FILE_ONLY=1

TJLFEP_ROOT="${TJLFEP_ROOT:-/pscratch/sd/t/tneiser/.julia/dev/TJLFEP}"
cd "${TJLFEP_ROOT}/build"

echo "=== TJLFEP full CPU sysimage build (TJLF + TJLFEP) ==="
echo "host: $(hostname)  date: $(date)"
julia --version
echo "TJLF: $(git -C "${TJLFEP_ROOT}/../TJLF" log -1 --oneline 2>/dev/null)"
echo "TJLFEP: $(git -C "${TJLFEP_ROOT}" log -1 --oneline 2>/dev/null)"

julia --project="${TJLFEP_ROOT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project="${TJLFEP_ROOT}" -t "${SLURM_CPUS_PER_TASK:-128}" sysimage/build_cpu_sysimage.jl

SO="${CPU_SYSIMAGE_OUT:-${TJLFEP_ROOT}/build/TJLFEP_cpu_sysimage.so}"
if [[ ! -f "${SO}" ]]; then
    echo "ERROR: sysimage not found at ${SO}"
    exit 1
fi
ls -lh "${SO}"

# Self-check: load with the image and confirm TJLF + TJLFEP are baked (instant load).
echo "=== verifying CPU image ==="
julia --startup-file=no --sysimage="${SO}" --project="${TJLFEP_ROOT}" -e '
    t=time(); using TJLF, TJLFEP
    println("load(using TJLF,TJLFEP)=", round(time()-t;digits=3), "s")
    @assert isdefined(TJLF, :run_tjlf) "TJLF not baked"
    @assert isdefined(TJLFEP, :run_gacode_scan_task) "TJLFEP not baked"
    heavy = filter(m -> nameof(m) in (:FUSE, :IMAS), Base.loaded_modules_array())
    println("file-only check: heavy deps loaded = ", isempty(heavy) ? "none" : nameof.(heavy))
    @assert isempty(heavy) "file-only image pulled in FUSE/IMAS"
    println("CPU sysimage OK: TJLF + TJLFEP baked (file-only)")'

echo "=== full CPU sysimage build OK: ${SO} ==="
