#!/bin/bash

#SBATCH --qos=regular
#SBATCH -A m3739
#SBATCH --constraint=cpu
#SBATCH -o ./run.out.TGLFEP
#SBATCH -e ./run.err.TGLFEP
#SBATCH --ntasks-per-node=128
#SBATCH --nodes=10
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH -J adaptive

srun -n 1280 $TGLFEP_DIR/TGLFEP_driver
