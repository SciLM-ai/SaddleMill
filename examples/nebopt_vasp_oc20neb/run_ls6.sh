#!/bin/bash

##SBATCH -p normal
#SBATCH -p development
##SBATCH -N 64
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
##SBATCH -t 48:00:00
#SBATCH -t 02:00:00
#SBATCH -J oc20nebVASP
#SBATCH -A CHE23004

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate tsearch

ml unload xalt python3
ml load impi cuda/12.8

srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m tsearch
