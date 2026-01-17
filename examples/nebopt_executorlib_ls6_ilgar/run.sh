#!/bin/sh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -o ll_out
#SBATCH -p gpu-a100-dev
#SBATCH -t 00:15:00
#SBATCH -A CHE23004

module unload impi python3
module load cuda/12.8

srun -n $SLURM_NNODES flux start python -u -m tsearch
