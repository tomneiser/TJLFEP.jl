#!/bin/bash

#SBATCH --qos=debug
#SBATCH -A m3739
#SBATCH --constraint=cpu
#SBATCH -o ./run.out.Alpha
#SBATCH -e ./run.err.Alpha
#SBATCH --ntasks-per-node=125
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH -J Alpha

srun -n 1 $ALPHA_DIR/Alpha_driver
