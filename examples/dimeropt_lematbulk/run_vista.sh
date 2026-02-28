#!/bin/sh
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -o ll_out
#SBATCH -p gh
#SBATCH -t 48:00:00
#SBATCH -A CHE23004

export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m tsearch
