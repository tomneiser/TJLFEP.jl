#!/bin/bash

#SBATCH --qos=regular
#SBATCH -A m808
#SBATCH --constraint=cpu
#SBATCH -o ./run.out.TGLFEP
#SBATCH -e ./run.err.TGLFEP
#SBATCH --ntasks-per-node=128
#SBATCH --nodes=40
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH -J adaptive

srun -n 5000 $TGLFEP_DIR/TGLFEP_driver
